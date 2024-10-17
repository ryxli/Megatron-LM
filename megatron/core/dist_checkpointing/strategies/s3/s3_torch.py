import io
import logging
from pathlib import Path
from pathlib_abc import PathBase
from typing import Any, Dict, List, Union, cast

import torch
from torch.distributed import checkpoint
from torch.distributed.checkpoint import Metadata, TensorStorageMetadata, BytesStorageMetadata
from torch.distributed._sharded_tensor import ShardedTensor as TorchShardedTensor

from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor, StateDict, ShardedObject
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.resharding import (
    TensorReformulationMetadata,
    apply_nd_flattened_tensors_reformulation,
    is_nd_flattened_tensor,
    restore_nd_flattened_tensors_formulation,
)
from megatron.core.dist_checkpointing.strategies.state_dict_saver import (
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreLoadPlanner,
    MCoreSavePlanner,
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
    _replace_sharded_keys_with_state_dict_keys,
    _replace_state_dict_keys_with_sharded_keys,
    _restore_dict_types,
    _unwrap_pyt_sharded_tensor,
    mcore_to_pyt_state_dict,
)

from megatron.core.dist_checkpointing.strategies.s3.s3_async import S3WriterAsync
from megatron.core.dist_checkpointing.strategies.s3.s3_filesystem import VirtualReader

logger = logging.getLogger(__name__)


class S3TorchDistSaveShardedStrategy(TorchDistSaveShardedStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def async_save(self, sharded_state_dict: Dict[str, Any], checkpoint_dir: Path) -> AsyncRequest:
        (
            sharded_state_dict,
            flat_mapping,
            rename_mapping,
        ) = _replace_state_dict_keys_with_sharded_keys(sharded_state_dict, self.keep_only_main_replica)
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        # Use PyT saving mechanism
        writer = S3WriterAsync(checkpoint_dir, thread_count=self.thread_count)
        # This should be set differently if we run in a smaller process group than the default
        coordinator = 0
        # Try twice to validate the generated `central_plan` is the same across iterations
        # If so, reuse `cached_central_plan` and `cached_global_metadata`
        # From the 3rd iteration, `save_state_dict_async_plan` will not generate `global_metadata`
        # (return None) so `self.cached_global_metadata` is reused
        args_cached_plans = None
        if self.use_cached_ckpt_structure:
            args_cached_plans = (
                self.cached_central_plan,
                self.cached_local_plan,
                self.validated_cache_reuse,
            )

        (
            save_state_dict_ret,
            self.cached_central_plan,
            self.cached_local_plan,
            self.validated_cache_reuse,
        ) = save_state_dict_async_plan(
            pyt_state_dict,
            writer,
            None,
            coordinator,
            planner=MCoreSavePlanner(dedup_replicated_tensors=not self.keep_only_main_replica),
            cached_ckpt_structure=args_cached_plans,
        )
        rank = torch.distributed.get_rank()
        if self.use_cached_ckpt_structure:
            if self.validated_cache_reuse:
                logger.debug(f"rank: {rank}, cache validated")
                if save_state_dict_ret[1]:  # when global_metadata is not cached
                    self.cached_global_metadata = save_state_dict_ret[1]  # Cache Metadata
                # Only Coordinator rank holds cached global_metadata
                # (None is returned for global_metadata)
                elif coordinator == rank:
                    logger.debug(f"rank: {rank}, reuse metadata, {save_state_dict_ret[1]}")
                    save_state_dict_ret = list(save_state_dict_ret)
                    save_state_dict_ret[1] = self.cached_global_metadata

        return self._get_save_and_finalize_callbacks(writer, save_state_dict_ret)

    def _get_save_and_finalize_callbacks(self, writer, save_state_dict_ret) -> AsyncRequest:
        save_fn_args = writer.get_save_function_and_args()
        save_fn, save_args = save_fn_args

        def finalize_fn():
            save_state_dict_async_finalize(*save_state_dict_ret)
            torch.distributed.barrier()

        return AsyncRequest(save_fn, save_args, [finalize_fn])


def _get_reformulation_metadata(
    sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
) -> Dict[str, TensorReformulationMetadata]:
    ckpt_metadata = VirtualReader(checkpoint_dir).read_metadata()
    reformulation_metadata = {}
    for sh_ten in nested_values(sharded_state_dict):
        if not is_nd_flattened_tensor(sh_ten):
            continue
        try:
            ckpt_global_shape = ckpt_metadata.mcore_data[sh_ten.key]["nd_reformulated_orig_global_shape"]
        except KeyError as e:
            raise CheckpointingException(
                f"Cannot find global shape metadata for N-D flattened tensor {sh_ten} in checkpoint metadata: {ckpt_metadata.mcore_data}"
            ) from e

        reformulation_metadata[sh_ten.key] = TensorReformulationMetadata(
            ckpt_global_shape, ckpt_metadata.state_dict_metadata[sh_ten.key].size
        )
    return reformulation_metadata


class S3TorchDistLoadShardedStrategy(TorchDistLoadShardedStrategy):
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path | PathBase) -> StateDict:
        """Translates MCore ShardedTensors to PyT ShardedTensors and loads from PyT Distributed format.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict with mapping
                information to instruct loading
            checkpoint_dir (Path): checkpoint directory

        Returns: loaded state dict
        """
        # Apply N-D tensors resharding
        sharded_state_dict, formulation_restore_data = apply_nd_flattened_tensors_reformulation(
            sharded_state_dict, _get_reformulation_metadata(sharded_state_dict, checkpoint_dir)
        )

        flexible_shape_sharded_tensors = [
            sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and not sh_ten.allow_shape_mismatch
        ]

        orig_sharded_state_dict = sharded_state_dict
        # MCore state dict to PyT Distributed compatible
        (
            sharded_state_dict,
            flat_mapping,
            rename_mapping,
        ) = _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, True)
        storage_reader = VirtualReader(checkpoint_dir)
        # Load PyT Distributed format
        checkpoint.load_state_dict(
            state_dict=pyt_state_dict,
            storage_reader=storage_reader,
            planner=MCoreLoadPlanner(shapes_validation_sharded_tensors=flexible_shape_sharded_tensors),
        )
        pyt_state_dict = cast(Dict[str, Union[TorchShardedTensor, List[io.BytesIO]]], pyt_state_dict)
        # Unwrap ShardedTensors and return to original state dict
        mcore_state_dict = {
            k: v if not isinstance(v, TorchShardedTensor) else _unwrap_pyt_sharded_tensor(v)
            for k, v in pyt_state_dict.items()
        }
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(mcore_state_dict, flat_mapping, rename_mapping)
        _restore_dict_types(mcore_state_dict, orig_sharded_state_dict)
        # Apply N-D tensors resharding postprocessing
        mcore_state_dict = restore_nd_flattened_tensors_formulation(mcore_state_dict, formulation_restore_data)
        storage_reader.cleanup()
        return mcore_state_dict

    def load_tensors_metadata(self, checkpoint_dir: Path, metadata: Metadata = None):
        """Uses tensors metadata stored in the metadata file."""
        if metadata is None:
            fs_reader = VirtualReader(checkpoint_dir)
            metadata = fs_reader.read_metadata()

        mcore_data = getattr(metadata, "mcore_data", {})
        sharded_metadata = {}
        for k, tp in metadata.state_dict_metadata.items():
            if not isinstance(tp, TensorStorageMetadata):
                continue  # load only tensors

            nd_orig_global_shape = mcore_data.get(k, {}).get("nd_reformulated_orig_global_shape")
            if nd_orig_global_shape is None:
                # Regular tensor
                sharded_metadata[k] = ShardedTensor.from_rank_offsets(
                    k,
                    torch.empty(tp.size, **tp.properties.__dict__, device="meta"),
                ).without_data()
            else:
                # N-D flattened tensor
                unflat_ten = torch.empty(nd_orig_global_shape, **tp.properties.__dict__, device="meta")
                flat_ten = unflat_ten.flatten()
                sharded_metadata[k] = ShardedTensor.from_rank_offsets_flat(
                    k,
                    flat_ten,
                    unflat_ten.shape,
                    flattened_range=slice(0, unflat_ten.numel()),  # whole slice
                ).without_data()

        return sharded_metadata

    def load_sharded_metadata(self, checkpoint_dir: Path) -> ShardedStateDict:
        """Uses tensors and objects metadata stored in the metadata file."""
        fs_reader = VirtualReader(checkpoint_dir)
        metadata = fs_reader.read_metadata()

        sharded_metadata = {}
        for metadata_key, storage_metadata in metadata.state_dict_metadata.items():
            if not isinstance(storage_metadata, BytesStorageMetadata):
                continue
            sh_obj = ShardedObject.empty_from_unique_key(metadata_key)
            sharded_metadata[sh_obj.unique_key] = sh_obj

        sharded_metadata.update(self.load_tensors_metadata(checkpoint_dir, metadata))
        return sharded_metadata
