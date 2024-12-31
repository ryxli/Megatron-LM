import megatron.core.datasets.utils as util
from tests.unit_tests.test_utilities import Utils


def test_should_build_on_rank():
    # Setup.
    Utils.initialize_model_parallel(2, 4)
    rank = Utils.rank
    world = Utils.world_size
    local_rank = rank % world

    if local_rank == 0:
        assert util.should_build_on_rank(rank, True)
    else:
        assert not util.should_build_on_rank(rank, True)

    # Teardown.
    Utils.destroy_model_parallel()
