# Owner(s): ["oncall: distributed"]
#
# Tests specific to the in-tree torchcomms NCCL backends.

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


class ProcessGroupNCCLLazyTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl-lazy"

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
    def test_p2p_send_recv_event_ordering(self) -> None:
        # Regression test for the single-op P2P group-wrap fix (TorchComms
        # D109625550). sendImpl/recvImpl now wrap ncclSend/ncclRecv in
        # ncclGroupStart/End so the kernel is enqueued before the end CUDA
        # event is recorded. Loop with distinct per-iteration values and read
        # back immediately after wait(): an early event completion (the bug)
        # would surface as a value mismatch.
        #
        # Exchange within disjoint rank pairs (r ^ 1). send and recv share the
        # backend's internal stream, so the two ops run in issue order on it;
        # even ranks issue send-then-recv and odd ranks recv-then-send so the
        # matching kernels rendezvous instead of deadlocking. A leftover odd
        # rank (odd world_size) has no partner and sits out.
        paired = (self.world_size // 2) * 2
        if self.rank >= paired:
            return
        partner = self.rank ^ 1
        numel = 1024 * 1024
        for i in range(50):
            send_val = float(i + 1) + self.rank
            recv_val = float(i + 1) + partner
            send_t = torch.full((numel,), send_val, device=self.device)
            recv_t = torch.empty((numel,), device=self.device)
            if self.rank % 2 == 0:
                send_work = dist.isend(send_t, partner)
                recv_work = dist.irecv(recv_t, partner)
            else:
                recv_work = dist.irecv(recv_t, partner)
                send_work = dist.isend(send_t, partner)
            send_work.wait()
            recv_work.wait()
            self.assertEqual(recv_t, torch.full((numel,), recv_val, device=self.device))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_sync_barrier_blocks_host_on_stream(self) -> None:
        # Regression test for the synchronous-barrier host block (TorchComms
        # D109866902 "add synchronize to sync barrier").
        # nccl2's Work.wait() is stream-order-only, so before the fix a
        # synchronous barrier did NOT host-block the CPU thread. Stock
        # ProcessGroupNCCL's barrier host-blocks, and downstream code (the
        # flashinfer trtllm one-shot Lamport all_reduce) relies on that to
        # flush an async buffer clear on the stream before the first
        # all_reduce; a stream-order-only barrier lets the all_reduce race the
        # clear and both ranks spin forever.
        #
        # Enqueue a long-running kernel on the current stream, confirm the
        # stream is still busy, then run a synchronous barrier + wait(): it
        # must host-block until the stream drains, so stream.query() is True.
        # This passes on stock nccl and (pre-fix) fails on nccl2.
        stream = torch.cuda.current_stream()
        torch.cuda._sleep(1_000_000_000)
        self.assertFalse(
            stream.query(), "precondition: enqueued work should leave stream busy"
        )
        work = dist.barrier(async_op=True)
        work.wait()
        self.assertTrue(
            stream.query(),
            "synchronous barrier must host-block until prior stream work completes",
        )


if __name__ == "__main__":
    if TEST_CUDA:
        run_tests()
