# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Module for managing distributed checkpoints metadata."""

import json
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Optional

from pathlib_abc import PathBase

from s3torchconnectorclient._mountpoint_s3_client import S3Exception

CONFIG_FNAME = "metadata.json"
logger = logging.getLogger(__name__)


class CheckpointingException(Exception):
    """Base checkpointing related exception"""

    pass


@dataclass
class CheckpointingConfig:
    """Documents backends used in the checkpoint.

    Checkpoint config keeps track of formats used for storing the sharded tensors
    (sharded_backend) and other objects (common_backend).

    Note that versioning is not for the checkpoint content (which is application specific),
    but for the checkpoint format itself.
    """

    sharded_backend: str
    sharded_backend_version: int = 1
    common_backend: str = "torch"
    common_backend_version: int = 1


def check_is_distributed_checkpoint(checkpoint_dir):
    """Checks if `metadata.json` exists in the checkpoint and is a valid config.

    Args:
        checkpoint_dir: checkpoint directory

    Returns:
        bool: True if `metadata.json` exists in the checkpoint and is a valid config.
    """
    return maybe_load_config(checkpoint_dir) is not None


def maybe_load_config(checkpoint_dir: str) -> Optional[CheckpointingConfig]:
    """Returns checkpoint config if `checkpoint_dir` is a distributed checkpoint and None otherwise

    Args:
        checkpoint_dir: checkpoint directory

    Returns:
        CheckpointingConfig (optional): None if checkpoint is not a valid distributed checkpoint
    """
    if not isinstance(checkpoint_dir, PathBase):
        config_path = Path(checkpoint_dir, CONFIG_FNAME)
    else:
        config_path = checkpoint_dir / CONFIG_FNAME
    if not config_path.exists():
        return None
    with config_path.open() as f:
        config_dict = json.load(f)
    return CheckpointingConfig(**config_dict)


def save_config(config: CheckpointingConfig, checkpoint_dir: str):
    """Save given config to checkpoint directory.

    Args:
        config: checkpoint config
        checkpoint_dir: checkpoint directory

    Returns:
        None
    """
    if not isinstance(checkpoint_dir, PathBase):
        config_path = Path(checkpoint_dir, CONFIG_FNAME)
    else:
        config_path = checkpoint_dir / CONFIG_FNAME

    # TODO: simple retry logic
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            with config_path.open("w") as f:
                json.dump(asdict(config), f)
            break
        except S3Exception as e:
            logger.warning(f"retry {attempt} encountered s3 exception when saving config: {e}")
            time.sleep(min(0.5 * 2**attempt, 3.0))
        except Exception as e:
            logger.warning(f"retry {attempt} when saving config: {e}")
