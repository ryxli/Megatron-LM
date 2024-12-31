##
# Compile megatron.core.datasets.helpers dependencies before BlendedDataset import
##

import random

import numpy
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset, _block_shuffle, _build_shuffle_index
from megatron.core.datasets.utils import compile_helpers
from megatron.training.tokenizer.tokenizer import _NullTokenizer
from tests.unit_tests.test_utilities import Utils

_MOCK_VOCAB_SIZE = 8192


def sample_N(dataset, N, randomize):
    if randomize:
        indices = [random.randint(0, len(dataset) - 1) for _ in range(N)]
    else:
        indices = list(range(N))
    samples = [dataset[index]["tokens"].numpy() for index in indices]
    return samples


def test_mock_gpt_dataset():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = _NullTokenizer(vocab_size=_MOCK_VOCAB_SIZE)

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,9,1",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        tokenizer=tokenizer,
    )

    datasets = BlendedMegatronDatasetBuilder(MockGPTDataset, [100, 100, 100], lambda: True, config).build()

    N = 10

    # Check iso-index variance by split
    subsets = [sample_N(dataset, N, randomize=False) for dataset in datasets]
    assert not numpy.allclose(subsets[0], subsets[1])
    assert not numpy.allclose(subsets[0], subsets[2])
    assert not numpy.allclose(subsets[1], subsets[2])

    # Check iso-split / iso-index identity
    subset_1A = sample_N(datasets[0], N, randomize=False)
    subset_1B = sample_N(datasets[0], N, randomize=False)
    assert numpy.allclose(subset_1A, subset_1B)

    # Check iso-split variance by index
    subset_1A = sample_N(datasets[0], N, randomize=True)
    subset_1B = sample_N(datasets[0], N, randomize=True)
    assert not numpy.allclose(subset_1A, subset_1B)

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,10,0",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        drop_last_partial_validation_sequence=False,
        add_extra_token_to_sequence=False,
        tokenizer=tokenizer,
    )

    datasets = BlendedMegatronDatasetBuilder(MockGPTDataset, [0, None, 0], lambda: True, config).build()

    sample = datasets[1][datasets[1].shuffle_index.argmax()]
    argmax = sample["labels"].shape[0] - torch.flip(sample["labels"], [0]).argmax() - 1

    # Test add_extra_token_to_sequence
    assert sample["tokens"][argmax] != tokenizer.eod
    assert sample["labels"][argmax] == tokenizer.eod

    # Test eod_mask_loss, drop_last_partial_validation_sequence
    assert argmax < sample["labels"].shape[0] - 1
    assert torch.all(sample["labels"][argmax + 1 :] == 0)
    assert not torch.any(
        sample["loss_mask"][torch.logical_and(sample["labels"] == tokenizer.eod, sample["labels"] == 0)]
    )

    sample = datasets[1][None]

    # Check handling of None index
    assert not torch.any(sample["loss_mask"])


def test_build_shuffle_index_equivalence_to_block_shuffle():
    shuffled = _build_shuffle_index(12, 12, numpy.random.RandomState(seed=0))
    block_one_shuffled = _build_shuffle_index(12, 12, numpy.random.RandomState(seed=0), shuffle_block_size=1)
    block_all_shuffled = _build_shuffle_index(12, 12, numpy.random.RandomState(seed=0), shuffle_block_size=12)
    assert numpy.array_equal(shuffled, block_one_shuffled)
    assert numpy.array_equal(shuffled, block_all_shuffled)


def test_block_shuffle():
    np_rng = numpy.random.RandomState(seed=0)

    # Regular case with shuffle_block_size > 1
    arr = numpy.arange(start=0, stop=12, step=1)
    shuffle_block_size = 4
    shuffled = _block_shuffle(arr.copy(), shuffle_block_size, np_rng)
    expected = numpy.array([10, 11, 8, 9, 4, 6, 5, 7, 3, 0, 2, 1])
    assert numpy.array_equal(shuffled, expected), f"block_size=4: {shuffled}"

    # shuffle_block_size = 1, equivalent to full array shuffle
    arr = numpy.array([0, 1, 2, 3, 4, 5])
    shuffle_block_size = 1
    shuffled = _block_shuffle(arr.copy(), shuffle_block_size, np_rng)
    expected = numpy.array([3, 1, 2, 4, 5, 0])
    assert numpy.array_equal(shuffled, expected), f"block_size=1: {shuffled}"

    # Empty array
    arr = numpy.array([])
    shuffle_block_size = 4
    shuffled = _block_shuffle(arr.copy(), shuffle_block_size, np_rng)
    expected = numpy.array([])
    assert numpy.array_equal(shuffled, expected), f"empty: {shuffled}"

    # Single element
    arr = numpy.array([42])
    shuffle_block_size = 1
    shuffled = _block_shuffle(arr.copy(), shuffle_block_size, np_rng)
    expected = numpy.array([42])
    assert numpy.array_equal(shuffled, expected), f"single element: {shuffled}"


if __name__ == "__main__":
    test_mock_gpt_dataset()
