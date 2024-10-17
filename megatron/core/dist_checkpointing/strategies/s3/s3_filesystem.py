import contextlib
import io
import logging
import os
import pathlib
import pickle
import tempfile
import itertools
import time
from typing import IO, Dict, List, Union, cast, Generator
from pathlib_abc import PathBase

from contextlib import contextmanager

import torch
from torch import Tensor
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.futures import Future
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.storage import StorageReader, StorageWriter, WriteResult
from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
    FileSystemWriter,
    _StorageInfo,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, LoadItemType, ReadItem
from torch.distributed.checkpoint.metadata import Metadata
from s3torchconnectorclient._mountpoint_s3_client import S3Exception

logger = logging.getLogger(__name__)

SHARED_MEM_DIR = "/dev/shm"


def _cast(path: Union[PathBase, os.PathLike]):
    if not isinstance(path, PathBase):
        path = pathlib.Path(path)
    return path


def _is_local_rank0():
    local_world_size = None
    for var in ["LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_TASKS_PER_NODE"]:
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

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        if isinstance(checkpoint_id, PathBase):
            return True

        return False

    def exists(self, path: Union[str, os.PathLike]):
        return _cast(path).exists()

    def isdir(self, path: Union[str, os.PathLike]):
        return _cast(path).is_dir()

    def ls(self, path: Union[str, os.PathLike]):
        return list(_cast(path).iterdir())


class VirtualWriter(FileSystemWriter):
    def __init__(
        self,
        path: Union[str, PathBase, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = False,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
    ):
        StorageWriter.__init__(self)
        self.fs = VirtualFileSystem()
        self.path = self.fs.init_path(path)
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files
        self.thread_count = thread_count
        self.per_thread_copy_ahead = per_thread_copy_ahead

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return VirtualFileSystem.validate_checkpoint_id(checkpoint_id)

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        meta_path = self.path / ".metadata"

        # TODO: simple retry logic
        max_attempts = 8
        for i in range(max_attempts):
            try:
                with self.fs.create_stream(meta_path, "wb") as metadata_file:
                    pickle.dump(metadata, metadata_file)
                break
            except S3Exception as e:
                logger.debug(f"retry {i} encountered s3 exception when saving .metadata: {e}")
            except Exception as e:
                logger.debug(f"retry {i} when saving .metadata: {e}")


@contextlib.contextmanager
def override_tempfile_names():
    original_get_candidate_names = tempfile._get_candidate_names
    tempfile._get_candidate_names = lambda: itertools.repeat("")
    try:
        yield
    finally:
        tempfile._get_candidate_names = original_get_candidate_names


class VirtualReader(FileSystemReader):
    def __init__(self, path: Union[str, PathBase, os.PathLike]):
        StorageReader.__init__(self)
        self.fs = VirtualFileSystem()
        self.path = self.fs.init_path(path)
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()

        if os.path.exists(SHARED_MEM_DIR):
            with override_tempfile_names():
                self._tmp_dir = (
                    tempfile.TemporaryDirectory(dir=SHARED_MEM_DIR, prefix="fs_reader") if _is_local_rank0() else None
                )
                self._tmp_dir_name = pathlib.Path(SHARED_MEM_DIR) / "fs_reader"

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return VirtualFileSystem.validate_checkpoint_id(checkpoint_id)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        if not os.path.exists(SHARED_MEM_DIR):
            return super().read_data(plan, planner)

        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            self._safe_cached_path(path, skip_wait=True)
            per_file.setdefault(path, []).append(read_item)

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
        Atomic operation to download the file from s3 in the current rank to shm
        if it doesn't exist, wait if another rank is already downloading the file,
        or return the file directly if it already exists
        Only works with POSIX
        """

        # download to shm on local node
        assert os.path.exists(SHARED_MEM_DIR)
        rank = torch.distributed.get_rank()
        new_path = self.fs.concat_path(self.path, relative_path)
        tmp_file = pathlib.Path(self._tmp_dir_name, relative_path)
        tmp_lock_file = pathlib.Path(self._tmp_dir_name, relative_path + ".lock")
        # acquire lock on relative
        while True:
            try:
                fd = os.open(tmp_lock_file, os.O_CREAT | os.O_EXCL)
                break
            except FileExistsError:
                if skip_wait:
                    return
                time.sleep(1)
        try:
            # attempt to download ckpt to shm if it doesn't already exist
            if not tmp_file.exists():
                start = time.time()
                logger.info(f"rank{rank} start downloading {new_path}")
                with new_path.open("rb") as reader, tmp_file.open("wb") as writer:
                    while chunk := reader.read(1024 * 1024):
                        writer.write(chunk)
                logger.info(f"rank{rank} finish downloading {new_path} in {time.time() - start}")
            else:
                logger.debug(f"rank{rank} skipped downloading {new_path}")
        finally:
            # release lock
            os.close(fd)
            os.remove(tmp_lock_file)

        return tmp_file

    def cleanup(self):
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()

    def __del__(self):
        self.cleanup()
