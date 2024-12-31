import os
import sys
from types import ModuleType
import pytest
from unittest.mock import patch

import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor


from megatron.core.datasets.utils import should_build_on_rank
import megatron.core.utils_s3 as utils_s3
from tests.unit_tests.test_utilities import Utils

try:
    import boto3
    import botocore.exceptions as exceptions
except ModuleNotFoundError:
    boto3 = ModuleType("boto3")
    sys.modules[boto3.__name__] = boto3
    exceptions = ModuleType("botocore.exceptions")
    sys.modules[exceptions.__name__] = exceptions


class _MockClient(utils_s3.S3Client):
    
    def __init__(self, **kwargs):
        pass

    def download_file(self, Bucket: str, Key: str, Filename: str, **kwargs) -> None:  # type: ignore
        with open(Filename, "w") as f:
            f.write("mock")

class _MockSession(boto3.Session):
    def client(self, *args, **kwargs):
        return _MockClient()


setattr(boto3, "client", _MockClient)
setattr(boto3, "Session", _MockSession)


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown_model_parallel():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


@pytest.fixture(scope="function")
def shared_tmpdir(tmp_path_factory):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            shared_dir = tmp_path_factory.mktemp("shared_dir")
        else:
            shared_dir = None
        shared_dir = [shared_dir]
        dist.broadcast_object_list(shared_dir, src=0)
        shared_dir = shared_dir[0]
        dist.barrier()
    else:
        shared_dir = tmp_path_factory.mktemp("shared_dir")

    yield shared_dir

    if dist.is_initialized():
        dist.barrier()


@pytest.fixture(scope="function")
def non_shared_tmpdir(tmp_path_factory):
    return tmp_path_factory.mktemp("test_dist")


@pytest.fixture
def s3_path():
    return "s3://bucket/file.txt"


@pytest.fixture
def mock_download_file():
    with patch("megatron.core.utils_s3._download_file", wraps=utils_s3._download_file) as mock:
        yield mock


def download_and_assert(s3_path, local_path, synchronize=True):
    utils_s3.maybe_download_file(s3_path, str(local_path), synchronize)
    assert os.path.exists(local_path)
    with open(local_path, "r") as f:
        assert f.read() == "mock"


def test_s3_client():
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        all_futures = [executor.submit(utils_s3._get_s3_client)]
        for future in all_futures:
            results.append(future.result())
    assert all(client is results[0] for client in results), (
        "S3 client references are not consistent within a distributed process"
    )

@pytest.mark.parametrize("tmpdir_fixture, is_shared", [
    ("shared_tmpdir", True),
    ("non_shared_tmpdir", False)
])
def test_maybe_download_file(mock_download_file, s3_path, request, tmpdir_fixture, is_shared):
    tmpdir = request.getfixturevalue(tmpdir_fixture)
    local_path = os.path.join(tmpdir, "file.txt")

    download_and_assert(s3_path, local_path)
    
    if is_shared:
        if dist.get_rank() == 0:
            mock_download_file.assert_called_once_with(utils_s3._get_s3_client(), s3_path, str(local_path))
        else:
            mock_download_file.assert_not_called()
    else:
        mock_download_file.assert_called_once_with(utils_s3._get_s3_client(), s3_path, str(local_path))


@pytest.mark.parametrize("tmpdir_fixture, is_shared_filesystem, should_build_func", [
    ("shared_tmpdir", True, should_build_on_rank),
    ("non_shared_tmpdir", False, lambda is_shared, rank: rank % 1 == 0)
])
def test_maybe_download_file_non_synchronize_ranks(mock_download_file, s3_path, request, tmpdir_fixture, is_shared_filesystem, should_build_func):
    tmpdir = request.getfixturevalue(tmpdir_fixture)
    local_path = os.path.join(tmpdir, "file.txt")

    if dist.is_initialized():
        rank = dist.get_rank()

        if should_build_func(is_shared_filesystem, rank):
            download_and_assert(s3_path, local_path, synchronize=False)

        dist.barrier()

        if rank != 0:
            download_and_assert(s3_path, local_path, synchronize=False)

        if should_build_func(is_shared_filesystem, rank):
            mock_download_file.assert_called_once_with(utils_s3._get_s3_client(), s3_path, str(local_path))
        else:
            mock_download_file.assert_not_called()
