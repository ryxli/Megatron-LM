import logging
import os
from pathlib import Path
from pathlib_abc import PathBase
import torch

from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace, nested_values
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedStateDict, StateDict, is_main_replica
from megatron.core.dist_checkpointing.strategies.common import (
    COMMON_STATE_FNAME,
    TorchCommonLoadStrategy,
    TorchCommonSaveStrategy,
)


logger = logging.getLogger(__name__)


class S3TorchCommonSaveStrategy(TorchCommonSaveStrategy):
    def save_common(self, common_state_dict: StateDict, checkpoint_dir: PathBase | os.PathLike):
        if torch.distributed.get_rank() == 0:
            path = checkpoint_dir / COMMON_STATE_FNAME
            with path.open(mode="wb") as s3writer:
                torch.save(common_state_dict, s3writer)

    def save_sharded_objects(
        self, sharded_objects_state_dict: ShardedStateDict, checkpoint_dir: PathBase | os.PathLike
    ):
        for sh_obj in nested_values(sharded_objects_state_dict):
            if is_main_replica(sh_obj.replica_id):
                save_path = checkpoint_dir / f"{sh_obj.unique_key}.pt"
                with save_path.open(mode="wb") as s3writer:
                    torch.save(sh_obj.data, s3writer)


class S3TorchCommonLoadStrategy(TorchCommonLoadStrategy):
    def load_common(self, checkpoint_dir: PathBase | os.PathLike):
        """Load common (non-sharded) objects state dict from the checkpoint.

        Args:
            checkpoint_dir (Path): checkpoint directory

        Returns:
            StateDict: state dict with non-sharded objects from the checkpoint
        """
        load_path = checkpoint_dir / COMMON_STATE_FNAME
        try:
            with load_path.open(mode="rb") as s3reader:
                return torch.load(s3reader, map_location="cpu")
        except FileNotFoundError as e:
            err_msg = f"Common file {load_path} does not exist"
            ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
            logger.debug(f"{err_msg}. Checkpoint directory content: {ckpt_files}")
            raise CheckpointingException(err_msg) from e

    def load_sharded_objects(self, sharded_objects_state_dict: ShardedStateDict, checkpoint_dir: Path):
        """Replaces all ShardedObject from a given state dict with values loaded from the checkpoint.

        Args:
            sharded_objects_state_dict (ShardedStateDict): sharded state dict defining what objects should be loaded.
            checkpoint_dir (Path): checkpoint directory

        Returns:
            None: sharded state dict is modified in place
        """

        def load_sharded_object(sh_obj: ShardedObject):
            sh_obj.data = None
            load_path = checkpoint_dir / f"{sh_obj.unique_key}.pt"
            try:
                with load_path.open(mode="rb") as s3reader:
                    loaded_obj = torch.load(s3reader)
            except FileNotFoundError as e:
                # Backward compatible logic: previously the save format was incorrect
                old_load_path = (checkpoint_dir / sh_obj.unique_key).with_suffix(".pt")
                try:
                    with load_path.open(mode="rb") as s3reader:
                        loaded_obj = torch.load(s3reader)
                except FileNotFoundError:
                    err_msg = f"Object shard {load_path} not found"
                    obj_subdir = checkpoint_dir / sh_obj.key
                    if obj_subdir.exists():
                        obj_files = [f.name for f in obj_subdir.iterdir()]
                        logger.debug(f"{err_msg}. Object {sh_obj.key} directory content: {obj_files}")
                    else:
                        ckpt_files = [f.name for f in checkpoint_dir.iterdir()]
                        logger.debug(
                            f"{err_msg}. Object {sh_obj.key} directory does not exist. Checkpoint directory content: {ckpt_files}"
                        )
                    raise CheckpointingException(err_msg) from e
            return loaded_obj

        return dict_list_map_inplace(load_sharded_object, sharded_objects_state_dict)

    def load_sharded_metadata(self, checkpoint_dir: Path) -> ShardedStateDict:
        sharded_metadata = {}
        for subdir in checkpoint_dir.iterdir():
            if not subdir.is_dir():
                continue
            shard_files = list(subdir.glob("shard_*.pt"))
            if not shard_files:
                continue
            sh_objs = []
            for shard_file in shard_files:
                full_key = f"{subdir.name}/{shard_file.stem}"
                sh_objs.append(ShardedObject.empty_from_unique_key(full_key))

            # This is a backward-compatibility fix, where the last global shape is missing in the name
            if sh_objs[0].global_shape[-1] < 0:
                max_last_offset = max(map(lambda sh_obj: sh_obj.global_offset[-1], sh_objs))
                for sh_obj in sh_objs:
                    sh_obj.global_shape = (*sh_obj.global_shape[:-1], max_last_offset + 1)

            # Update the sharded state dict
            for sh_obj in sh_objs:
                sharded_metadata[sh_obj.unique_key] = sh_obj
        return sharded_metadata

    @property
    def can_handle_sharded_objects(self):
        return True

    def check_backend_compatibility(self, loaded_version):
        pass

    def check_version_compatibility(self, loaded_version):
        pass
