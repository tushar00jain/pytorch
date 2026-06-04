# Owner(s): ["oncall: distributed"]

"""
Regression tests for: new_group() must not route a 'fake' backend through the
TorchComms split_group delegation.

When TorchComms is enabled, new_group() delegates to split_group(). A "fake"
backend has no real communicator to split, and split_group cannot honor
use_local_synchronization=True -- which is exactly how DeviceMesh creates
disabled/unflattened mesh dimensions (it wants hashed PG names). Before the fix
that delegation raised NotImplementedError. The fix skips the delegation for fake
backends, falling through to the normal path, which builds a FakeProcessGroup and
still produces a hashed name.

These tests run single-process on CPU: no GPU and no real TorchComms are needed.
"""

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.fake_pg import FakeStore


def _init_fake_world(world_size=2):
    dist.init_process_group(
        backend="fake", rank=0, world_size=world_size, store=FakeStore()
    )


def test_fake_new_group_under_torchcomms_builds_fake_pg():
    """The fix: with TorchComms 'enabled', new_group(backend='fake',
    use_local_synchronization=True) builds a FakeProcessGroup (no
    NotImplementedError) and keeps the hashed name."""
    _init_fake_world()
    try:
        # Simulate TorchComms being enabled for just the new_group call, so the
        # test needs neither the torchcomms package nor a GPU.
        orig = c10d._use_torchcomms_enabled
        c10d._use_torchcomms_enabled = lambda: True
        try:
            g = dist.new_group(
                ranks=[0, 1], backend="fake", use_local_synchronization=True
            )
        finally:
            c10d._use_torchcomms_enabled = orig

        assert g is not None
        assert dist.get_backend(g) == "fake"
        # Hashed-name path preserved: sha1 hexdigest (40 chars), not a counter int.
        name = g.group_name
        assert not name.isdigit(), f"expected hashed name, got integer {name!r}"
        assert len(name) == 40, f"expected 40-char sha1 hash, got {name!r}"
    finally:
        dist.destroy_process_group()


def test_torchcomms_real_flag_if_available():
    """Same as above but driving the real dist_config.use_torchcomms flag.
    Skipped when the torchcomms package is not installed."""
    import torch.distributed.config as dist_config

    try:
        import torchcomms  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("torchcomms not installed")

    prev = dist_config.use_torchcomms
    dist_config.use_torchcomms = True
    try:
        _init_fake_world()
        try:
            g = dist.new_group(
                ranks=[0, 1], backend="fake", use_local_synchronization=True
            )
            assert g is not None and dist.get_backend(g) == "fake"
            assert len(g.group_name) == 40
        finally:
            dist.destroy_process_group()
    finally:
        dist_config.use_torchcomms = prev


if __name__ == "__main__":
    run_tests()
