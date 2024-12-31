import logging
import os
from time import time
import traceback
from typing import List, Optional, Union
import pickle

from pathlib_abc import PathBase
from torch import multiprocessing as mp
from torch.distributed.checkpoint.filesystem import (
    _write_item,
)
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.storage import WriteResult

from tenacity import retry, wait_random_exponential, before_sleep_log, stop_after_attempt, retry_if_exception_type
from s3torchconnectorclient._mountpoint_s3_client import S3Exception

from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
    _disable_gc,
    _process_memory,
    WriteBucket
)
from megatron.core.dist_checkpointing.strategies.s3.s3_filesystem import VirtualFileSystem


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VirtualWriterAsync(FileSystemWriterAsync):
    """
    Async-enabled implementation of FileSystemWriter using S3
    
    Flow:
    1. Call `write_data`
    2. Externally start async process with `get_save_function_and_args` function and args
    3. The async function to call is `writer_proxy_func` which calls
       `write_preloaded_data` in multiple processes

    After saving is finalized on all ranks:
    4. Call `super().finish` with the results gathered in `self.writer_result`
    """

    def __init__(self, path: Union[str, PathBase, os.PathLike], *args, **kwargs):
        super().__init__(path="", *args, **kwargs)
        if not self.single_file_per_rank:
            raise NotImplementedError("single_file_per_rank flag not supported for S3WriterAsync")
        self.fs = VirtualFileSystem()
        self.path = self.fs.init_path(path)

        # Intermediate state between preparation and finalization
        self.write_buckets: Optional[List[WriteBucket]] = None
        self.results_queue: Optional[mp.Queue] = None

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_multiproc(
        write_buckets: List[WriteBucket], global_results_queue: mp.Queue,
    ) -> None:
        """
        Performs saving data to storage with multiple processes.

        Starts predefined number of processes and uses 2 queues to make sure the results
        are complete:
        - local_results_queue - to send the actual results
        - count_queue - small queue to mark worker as completed

        Using just one queue disallowed proper exception handling.

        This method is meant to be run in a forked subprocess.
        Triggering GC during execution leads to CUDA errors
        (cleaning up tensors owned by the parent process).
        To prevent this, we disable the GC explicitly for this function with _disable_gc.

        Args:
            write_buckets (List[WriteBucket]): write plan
            global_results_queue (mp.Queue): mp.Queue to collect Dict[List[WriteResults]] (or an Exception)
                from parallel write processes to the main training process
        Returns: None
        """
        w_start = time()
        write_results_or_exc: Union[dict, Exception] = dict()
        ctx = mp.get_context('fork')
        local_results_queue = ctx.Queue()
        count_queue = ctx.JoinableQueue()
        p_list = []
        for i, write_bucket in enumerate(write_buckets):
            try:
                count_queue.put(i)
                p_list.append(
                    ctx.Process(
                        target=VirtualWriterAsync.write_preloaded_data,
                        args=(i, write_bucket, local_results_queue, count_queue, True),
                    )
                )
            except Exception as e:
                stack_trace = traceback.format_exc()
                err_msg = (
                    f"An error is caught while a proc {i} is created, error: {e}\n"
                    f"Stack trace:\n{stack_trace}"
                )
                logger.error(err_msg)
                write_results_or_exc = RuntimeError(err_msg)

        if not isinstance(write_results_or_exc, Exception):
            for p in p_list:
                p.start()

            logger.debug('FileSystemWriterAsync: collecting worker results...')

            # To make sure all nodes are completed
            count_queue.join()
            # At this point, all workers completed, so the queue should have exactly `len(write_buckets)` items
            for proc_idx in range(len(write_buckets)):
                try:
                    local_proc_idx, local_results_or_exc = local_results_queue.get()
                except queue.Empty:
                    write_results_or_exc = RuntimeError(
                        f'Unexpected empty `local_results_queue` (got only {proc_idx}/{len(write_buckets)} items)'
                    )
                    break
                else:
                    if isinstance(local_results_or_exc, Exception):
                        err_msg = f"Local process {local_proc_idx} encountered an error: {local_results_or_exc}"
                        logger.error(err_msg)
                        write_results_or_exc = local_results_or_exc
                        break
                    else:
                        assert isinstance(local_results_or_exc, list), type(local_results_or_exc)
                        write_results_or_exc[local_proc_idx] = local_results_or_exc
                        p_list[local_proc_idx].join()

            logger.debug('FileSystemWriterAsync: collected worker results successfully')

        global_results_queue.put(write_results_or_exc)

    @staticmethod
    @_disable_gc()
    def write_preloaded_data(
        local_proc_idx: int,
        write_bucket: WriteBucket,
        results_queue: mp.SimpleQueue,
        count_queue: mp.JoinableQueue,
        use_fsync: bool,
    ) -> None:
        """
        Performs actual data saving to storage.

        Args:
            local_proc_idx (int): index of a local process that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (mp.Queue): queue to return the write results to the proxy checkpoint process.
            count_queue (mp.JoinableQueue): queue to marks worker task as completed
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None, the write result are put into the `queue`
        """
        mem_before = _process_memory()

        local_results = []
        try:
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            with VirtualFileSystem().create_stream(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    local_results.append(_write_item(stream, data, write_item, storage_key))

                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu
                    local_results.append(_write_item(stream, tensor, write_item, storage_key))
                    
                # if use_fsync:
                #     os.fsync(stream.fileno())

            local_output = (local_proc_idx, local_results)
        except Exception as e:
            local_output = (local_proc_idx, e)

        results_queue.put(local_output)
        # Signal this process is done.
        count_queue.get()
        count_queue.task_done()

        mem_after = _process_memory()
        logger.debug(
            f"{local_proc_idx} consumed: {mem_after - mem_before}, before: {mem_before}, after: {mem_after}"
        )

    # TODO: results in excessive HeadObject requests when resuming
    # def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
    #     new_plans = [dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}/")) for i, plan in enumerate(plans)]
    #     return new_plans


    # @retry(
    #     wait=wait_random_exponential(multiplier=1, min=1, max=16),
    #     stop=stop_after_attempt(3),
    #     retry=retry_if_exception_type(Exception),
    #     before_sleep=before_sleep_log(logger, logging.WARNING),
    # )
    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:        
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        
        metadata.storage_meta = self.storage_meta()
        
        # tmp_path = cast(Path, self.fs.concat_path(self.path, f"{_metadata_fn}.tmp"))
        meta_path = self.path / ".metadata"
        with self.fs.create_stream(meta_path, "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            # if self.sync_files:
            #     try:
            #         os.fsync(metadata_file.fileno())
            #     except AttributeError:
            #         os.sync()

        # if self.fs.exists(self.metadata_path):
        #     self.fs.rm_file(self.metadata_path)

        # self.fs.rename(tmp_path, self.metadata_path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return VirtualFileSystem.validate_checkpoint_id(checkpoint_id)