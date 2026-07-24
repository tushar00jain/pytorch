# Owner(s): ["oncall: distributed"]
#
# Tests specific to the in-tree torchcomms NCCL backends.

import time

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


class ProcessGroupNCCL2Test(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_split_group(self) -> None:
        # Splitting is a nccl2 feature; nccl-lazy builds subgroups members-only
        # via new_group instead. MultiProcContinuousTest turns skipTest into a
        # class-poisoning error, so return early (all ranks agree on the backend
        # so this stays collective-safe).
        if self.backend_str() != "nccl2":
            return
        # dist.split_group drives Backend::split (ncclCommSplit under the hood).
        # It requires the default pg to have a bound device.
        default_pg = dist.distributed_c10d._get_default_group()
        default_pg.bound_device_id = self.device
        # Initialize the parent communicator before splitting.
        dist.all_reduce(torch.ones(4, device=self.device))

        # First half are members; the rest are non-members and must get the
        # NON_GROUP_MEMBER sentinel (they still issue the no-color split so the
        # collective stays in lockstep across the parent group).
        members = list(range(self.world_size // 2))
        subgroup = dist.split_group(split_ranks=[members])

        if self.rank in members:
            self.assertIsNot(subgroup, dist.GroupMember.NON_GROUP_MEMBER)
            self.assertEqual(dist.get_process_group_ranks(subgroup), members)
            tensor = torch.full((4,), float(self.rank + 1), device=self.device)
            dist.all_reduce(tensor, group=subgroup)
            expected = float(sum(r + 1 for r in members))
            self.assertEqual(tensor, torch.full_like(tensor, expected))
        else:
            self.assertIs(subgroup, dist.GroupMember.NON_GROUP_MEMBER)

        dist.barrier()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_split_group_full_partition(self) -> None:
        # Partition the whole parent group into two subgroups; every rank is a
        # member of exactly one split and gets a live communicator.
        if self.backend_str() != "nccl2":
            return
        if self.world_size % 2 != 0:
            return
        default_pg = dist.distributed_c10d._get_default_group()
        default_pg.bound_device_id = self.device
        dist.all_reduce(torch.ones(4, device=self.device))

        half = self.world_size // 2
        first = list(range(half))
        second = list(range(half, self.world_size))
        subgroup = dist.split_group(split_ranks=[first, second])
        my_group = first if self.rank in first else second

        self.assertIsNot(subgroup, dist.GroupMember.NON_GROUP_MEMBER)
        self.assertEqual(dist.get_process_group_ranks(subgroup), my_group)
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        dist.all_reduce(tensor, group=subgroup)
        expected = float(sum(r + 1 for r in my_group))
        self.assertEqual(tensor, torch.full_like(tensor, expected))
        dist.barrier()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_watchdog_does_not_release_python_backed_tensor(self) -> None:
        class TensorSubclass(torch.Tensor):
            pass

        tensor = torch.ones(4, device=self.device).as_subclass(TensorSubclass)
        outputs = [torch.empty(4, device=self.device) for _ in range(self.world_size)]
        work = dist.all_gather(outputs, tensor, async_op=True)
        del tensor
        del work

        torch.cuda.synchronize()
        time.sleep(2)
        dist.barrier()


class ProcessGroupNCCL2ExpandableSegmentsTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
        super()._init_pg(rank, world_size, rdvz_file)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_large_in_place_all_gather(self) -> None:
        numel = 16 * 1024 * 1024
        output = torch.empty(
            self.world_size * numel, dtype=torch.bfloat16, device=self.device
        )
        input = output.narrow(0, self.rank * numel, numel)
        input.fill_(self.rank)
        self.assertTrue(
            any(segment["is_expandable"] for segment in torch.cuda.memory_snapshot())
        )

        dist.all_gather_single(output, input)

        for rank, chunk in enumerate(output.chunk(self.world_size)):
            self.assertEqual(chunk, torch.full_like(chunk, rank))


class ProcessGroupNCCLLazyTest(ProcessGroupNCCL2Test):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl-lazy"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_lazy_pair_channels(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        before_collective = backend._num_active_channels()
        t = torch.full((4,), 1.0, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self.assertEqual(backend._num_active_channels(), before_collective)

        send_t = torch.full((4,), float(self.rank), device=self.device)
        recv_t = torch.empty((4,), device=self.device)
        nxt = (self.rank + 1) % self.world_size
        prev = (self.rank - 1) % self.world_size
        if self.rank % 2 == 0:
            dist.send(send_t, nxt)
            dist.recv(recv_t, prev)
        else:
            dist.recv(recv_t, prev)
            dist.send(send_t, nxt)
        torch.cuda.synchronize()
        self.assertEqual(recv_t, torch.full((4,), float(prev), device=self.device))

        expected = 1 if nxt == prev else 2
        self.assertGreaterEqual(backend._num_active_channels(), expected)


if __name__ == "__main__":
    if TEST_CUDA:
        run_tests()
