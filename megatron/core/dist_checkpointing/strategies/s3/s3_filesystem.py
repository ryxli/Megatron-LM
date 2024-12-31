import contextlib
import errno
import io
import itertools
import logging
import os
import random
import tempfile
import time
import fcntl
import threading
from pathlib import Path
from pathlib_abc import PathBase
from contextlib import contextmanager
from typing import IO, ClassVar, Dict, Generator, List, Optional, Union, cast

import torch
from torch import Tensor
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
)
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
)
from torch.futures import Future

from megatron.core.utils_s3 import _get_s3_client

from tenacity import (
    retry,
    before_sleep_log,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)


logger = logging.getLogger(__name__)

SHARED_MEM_DIR = "/dev/shm"


def _cast(path: Union[PathBase, os.PathLike]) -> PathBase:
    if not isinstance(path, PathBase):
        path = Path(path)
    return path


def _is_local_rank0():
    local_world_size = None
    for var in [
        "LOCAL_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "SLURM_TASKS_PER_NODE",
    ]:
        if var in os.environ:
            local_world_size = int(os.environ[var])
            break
    if local_world_size is None:
        local_world_size = torch.cuda.device_count()
    return torch.distributed.get_rank() % local_world_size == 0


class VirtualFileSystem(FileSystemBase):
    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike], mode: str) -> Generator[io.IOBase, None, None]:  # type: ignore
        with _cast(path).open(mode) as stream:
            yield cast(io.IOBase, stream)

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> PathBase:
        return _cast(path) / suffix

    def init_path(self, path: Union[str, os.PathLike]) -> PathBase:
        return _cast(path)

    def rename(self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]) -> None:
        # s3 not supported
        _cast(path).rename(new_path)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        # s3 not supported
        _cast(path).mkdir(parents=True, exist_ok=True)

    def exists(self, path: Union[str, os.PathLike]):
        return _cast(path).exists()

    def isdir(self, path: Union[str, os.PathLike]):
        return _cast(path).is_dir()

    def ls(self, path: Union[str, os.PathLike]):
        return list(_cast(path).iterdir())

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        _cast(path).unlink()

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        if isinstance(checkpoint_id, PathBase):
            return True
        return False


@contextlib.contextmanager
def override_tempfile_names():
    """Override temp dir behavior to get consistent tmp dir for all ranks"""
    original_get_candidate_names = tempfile._get_candidate_names
    tempfile._get_candidate_names = lambda: itertools.repeat("")
    try:
        yield
    finally:
        tempfile._get_candidate_names = original_get_candidate_names


class VirtualReader(FileSystemReader):
    _tmp_dir: ClassVar[Optional[tempfile.TemporaryDirectory]] = None
    _tmp_dir_name: ClassVar[Optional[Path]] = None
    _tmp_dir_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _maybe_set_tmp_dir(cls):
        if cls._tmp_dir_name is None:
            with cls._tmp_dir_lock:
                if _is_local_rank0():
                    if cls._tmp_dir is None and os.path.exists(SHARED_MEM_DIR):
                        with override_tempfile_names():
                            cls._tmp_dir = tempfile.TemporaryDirectory(dir=SHARED_MEM_DIR, prefix="fs_reader")
            cls._tmp_dir_name = Path(SHARED_MEM_DIR) / "fs_reader"
            torch.distributed.barrier()

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return VirtualFileSystem.validate_checkpoint_id(checkpoint_id)

    @classmethod
    def cleanup(cls):
        if cls._tmp_dir is not None:
            cls._tmp_dir.cleanup()

    def __init__(self, path: Union[str, PathBase, os.PathLike]):
        super().__init__(path="")
        self.fs = VirtualFileSystem()
        self.path = self.fs.init_path(path)
        if os.path.exists(SHARED_MEM_DIR):
            VirtualReader._maybe_set_tmp_dir()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        if not os.path.exists(SHARED_MEM_DIR):
            return super().read_data(plan, planner)

        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path in per_file.keys():
            self._safe_cached_path(relative_path, skip_wait=True)

        for relative_path, reqs in per_file.items():
            new_path = self._safe_cached_path(relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(file_slice.read(item_md.length))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        tensor = cast(
                            Tensor,
                            torch.load(cast(IO[bytes], file_slice), map_location="cpu"),
                        )
                        tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                        target_tensor = planner.resolve_tensor(req).detach()
                        assert (
                            target_tensor.size() == tensor.size()
                        ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def _safe_cached_path(self, relative_path, skip_wait=False):
        """
        Atomically download a file from S3 to shared memory if it doesn't exist.
        Ensures only one thread/process downloads the file.
        """
        assert os.path.exists(SHARED_MEM_DIR), "Shared memory directory does not exist."
        rank = torch.distributed.get_rank()
        tmp_file = Path(VirtualReader._tmp_dir_name, relative_path)
        tmp_lock_file = Path(VirtualReader._tmp_dir_name, relative_path + ".lock")
        tmp_done_file = Path(VirtualReader._tmp_dir_name, relative_path + ".done")
        new_path = self.fs.concat_path(self.path, relative_path)

        if tmp_done_file.exists() and tmp_file.exists():
            logger.info(f"rank{rank} found {tmp_file} on disk, skipping.")
            return tmp_file

        try:
            os.open(tmp_lock_file, os.O_CREAT | os.O_EXCL)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        lock_file_fd = None
        try:
            lock_file_fd = os.open(tmp_lock_file, os.O_RDWR)
            while True:
                try:
                    fcntl.flock(lock_file_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    logger.info(f"rank{rank} acquired lock {tmp_lock_file}")
                    break
                except (IOError, OSError) as e:
                    if e.errno != errno.EWOULDBLOCK:
                        raise
                    if skip_wait:
                        logger.info(f"rank{rank} failed to acquire lock {tmp_lock_file}, skipping.")
                        return None
                    time.sleep(random.uniform(0.1, 0.5))

            if tmp_done_file.exists() and tmp_file.exists():
                logger.info(f"rank{rank} found {tmp_file} on disk, skipping.")
            else:
                tmp_file.parent.mkdir(exist_ok=True, parents=True)
                download_file(new_path, tmp_file, rank)

                # atomic mark completion of download
                tmp_done_file_tmp = tmp_done_file.with_suffix(".tmp")
                with tmp_done_file_tmp.open("w") as f:
                    f.write("")
                    f.flush()
                    os.fsync(f.fileno())
                os.rename(tmp_done_file_tmp, tmp_done_file)
                logger.info(f"rank{rank} successfully downloaded and marked {tmp_file} as done.")
        finally:
            if lock_file_fd is not None:
                fcntl.flock(lock_file_fd, fcntl.LOCK_UN)
                os.close(lock_file_fd)
        return tmp_file


@retry(
    wait=wait_random_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    # before=random_initial_jitter,
)
def download_file(new_path, tmp_file, rank):
    start = time.time()
    logger.info(f"rank{rank} start downloading {new_path}")
    _get_s3_client().download_file(new_path.bucket, new_path.key, tmp_file)
    logger.info(f"rank{rank} finish downloading {new_path} in {time.time() - start}")
