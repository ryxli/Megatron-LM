# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass
from functools import cache
import os
import time
from typing import Any, Dict, Protocol, Tuple
from urllib.parse import urlparse

from git import Optional
import torch

try:
    import boto3
    from botocore.config import Config
    import botocore.exceptions as exceptions
except ModuleNotFoundError:
    pass

S3_PREFIX = "s3://"


class S3Client(Protocol):
    """The protocol which all s3 clients should abide by"""

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None: ...

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None: ...

    def head_object(self, Bucket: str, Key: str) -> Dict[str, Any]: ...

    def get_object(self, Bucket: str, Key: str, Range: str) -> Dict[str, Any]: ...

    def close(self) -> None: ...


@cache
def _get_s3_client() -> S3Client:
    session = boto3.Session()
    return session.client(
        "s3",
        config=Config(retries={"max_attempts": 10, "mode": "standard"}),
    )


@dataclass
class S3Config:
    """Config when the data (.bin) file and the index (.idx) file are in S3

    TODO: These parameters are few and can be consolidated with parameters specific to bin reader
    classes - @jkamalu

    Attributes:

        path_to_idx_cache (str): The local directory where we will store the index (.idx) file

        bin_chunk_nbytes (int): If the number of bytes is too small, then we send a request to S3 at each call of the `read` method in _S3BinReader, which is slow, because each request has a fixed cost independent of the size of the byte range requested. If the number of bytes is too large, then we only rarely have to send requests to S3, but it takes a lot of time to complete the request when we do, which can block training. We've found that 256 * 1024 * 1024 (i.e., 256 MiB) has worked well (though we have not put that much effort into tuning it), so we default to it.

        synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks behavior. Set to False when we enforce this behavior at higher level.
    """

    path_to_idx_cache: str
    """Path for caching indices for s3 dataloading."""

    block_size: Optional[int] = None

    bin_chunk_nbytes: int = 256 * 1024 * 1024

    synchronize_ranks: bool = True

    def __getstate__(self):
        """exclude unpicklable objects from state"""
        state = {k: v for k, v in self.__dict__.items() if k not in {"_lock", "s3_client"}}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def is_s3_path(path: str) -> bool:
    """Ascertain whether a path is in S3

    Args:
        path (str): The path

    Returns:
        bool: True if the path is in S3, False otherwise
    """
    return path.startswith(S3_PREFIX)


def parse_s3_path(path: str) -> Tuple[str, str]:
    """Parses the given S3 path returning correspsonding bucket and key.

    Args:
        path (str): The S3 path

    Returns:
        Tuple[str, str]: A (bucket, key) tuple
    """
    _, bucket, key, _, _, _ = urlparse(path)
    return bucket, key.lstrip("/")


def object_exists(client: S3Client, path: str) -> bool:
    """Ascertain whether the object at the given S3 path exists in S3

    Args:
        client (S3Client): The S3 client

        path (str): The S3 path

    Raises:
        botocore.exceptions.ClientError: The error code is 404

    Returns:
        bool: True if the object exists in S3, False otherwise
    """
    parsed_s3_path = parse_s3_path(path)
    try:
        response = client.head_object(bucket=parsed_s3_path[0], key=parsed_s3_path[1])
    except exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise e
    return True


def _download_file(client: S3Client, s3_path: str, local_path: str) -> None:
    """Download the object at the given S3 path to the given local file system path

    Args:
        client (S3Client): The S3 client

        s3_path (str): The S3 source path

        local_path (str): The local destination path
    """
    dirname = os.path.dirname(local_path)
    os.makedirs(dirname, exist_ok=True)
    parsed_s3_path = parse_s3_path(s3_path)
    client.download_file(parsed_s3_path[0], parsed_s3_path[1], local_path)


def maybe_download_file(s3_path: str, local_path: str, synchronize_ranks: bool = True) -> None:
    """Download the object at the given S3 path to the given local file system path

    In a distributed setting, downloading the S3 object proceeds in stages in order
    to try to have the minimum number of processes download the object in order for
    all the ranks to have access to the downloaded object.

    Args:
        s3_path (str): The S3 source path

        local_path (str): The local destination path

        synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks behavior. Set to False when we enforce this behavior at higher level.
    """

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        local_rank = rank % torch.cuda.device_count()
    else:
        rank = int(os.getenv("RANK", "0"))
        local_world = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        local_rank = rank % local_world

    s3_client = _get_s3_client()

    if (not os.path.exists(local_path)) and (rank == 0):
        _download_file(s3_client, s3_path, local_path)

    if synchronize_ranks and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # If the `local_path` is in a file system that is not
    # shared across all the ranks, then we assume it's in the
    # host file system and each host needs to download the file.
    if synchronize_ranks and ((not os.path.exists(local_path)) and (local_rank == 0)):
        _download_file(s3_client, s3_path, local_path)

    if synchronize_ranks and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # If the `local_path` still does not exist, then we assume
    # each rank is saving to a separate location.
    if not os.path.exists(local_path):
        _download_file(s3_client, s3_path, local_path)

    if synchronize_ranks and torch.distributed.is_initialized():
        torch.distributed.barrier()

    assert os.path.exists(
        local_path
    ), f"{local_path} does not exist {s3_path}, rank={local_rank}, synchronize_ranks={synchronize_ranks}"
