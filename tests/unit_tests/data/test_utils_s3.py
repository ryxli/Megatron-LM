import sys
from types import ModuleType

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import megatron.core.datasets.utils_s3 as utils_s3
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
    def __init__(self, *args: Any) -> None:
        pass

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        pass

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        pass

    def head_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        pass

    def get_object(self, Bucket: str, Key: str, Range: str) -> Dict[str, Any]:
        pass

    def close(self) -> None:
        pass


class _MockSession(boto3.Session):
    def client(self, *args, **kwargs):
        return _MockClient()


setattr(boto3, "client", _MockClient)
setattr(boto3, "Session", _MockSession)


class MockDataset:
    def __init__(self, s3_config: utils_s3.S3Config):
        self.s3_config = s3_config
        self.s3_config.init_s3_client()


def test_s3_config():
    # Setup.
    Utils.initialize_model_parallel()
    s3_config = utils_s3.S3Config(path_to_idx_cache="path")

    def _build_mock_s3_datasets(size: int, s3_config: utils_s3.S3Config):
        return MockDataset(s3_config)

    # Execution using ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        all_futures = [executor.submit(_build_mock_s3_datasets, 2, s3_config)]
        for future in all_futures:
            results.append(future.result())

    current_client_reference = None
    for item in results:
        if current_client_reference is not None:
            assert (
                item.s3_config.s3_client is current_client_reference
            ), "S3 client references are not the same across datasets within a process"
        current_client_reference = item.s3_config.s3_client

    # Teardown.
    Utils.destroy_model_parallel()
