# Owner(s): ["oncall: distributed"]

"""
Reproducer for the pipeline-parallel send/recv deadlock with the torchcomms
communication backend (``torch.distributed.config.use_torchcomms``).

Root cause
----------
Every PP schedule in ``torch.distributed.pipelining`` fuses a send and a recv
into a single *mixed* P2P batch (e.g. 1F1B fires ``fwd_sends + bwd_recvs`` via
``_batch_p2p`` -> ``torch.distributed.batch_isend_irecv``). ``batch_isend_irecv``
only coalesces the ops into one NCCL group when the backend reports
``supports_coalescing == True``. A torchcomms ``_BackendWrapper`` without
coalescing issues each op sequentially on the single PP communicator: send first,
recv second. Two neighbours that both enqueue send-before-recv on the same
in-order comm deadlock.

Cross-batch ordering hazard (coalescing is NOT sufficient)
----------------------------------------------------------
Coalescing only makes a *single* mixed batch safe. A single PP communicator is a
FIFO and a schedule issues many batches; when ranks reach their per-neighbour
exchanges in different relative orders (pipeline skew, looped / V schedules, skip
connections) the FIFO forms a dependency cycle and deadlocks even though every
batch is coalesced. This is backend-independent -- it deadlocks on nccl and
torchcomms alike (see the ring tests). The fix is per-direction communicators
(upstream via ``torch.distributed.config.pipeline_per_direction_p2p``; here
exercised at the primitive level by ``_run_ring_per_direction``).

Spawns its own worker processes. Pair cases need >= 2 GPUs; ring cases >= 3.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    import torchcomms  # noqa: F401

    HAS_TORCHCOMMS = True
except ImportError:
    HAS_TORCHCOMMS = False


def requires_gpus(n: int):
    return unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= n,
        f"needs >= {n} GPUs",
    )


def _init_pg(
    backend_mode: str, rank: int, world_size: int, master_port: int = 29593
) -> torch.device:
    """Init a process group, optionally routing P2P through torchcomms."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    device_id = None
    if backend_mode == "torchcomms":
        import torchcomms  # noqa: F401

        import torch.distributed.config as dist_config

        dist_config.use_torchcomms = True
        # torchcomms requires a concrete device_id at init time.
        device_id = device

    import torch.distributed as dist

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device_id,
    )
    return device


def _run_mixed_batch_p2p(rank: int, world_size: int, *, native_batch: bool) -> None:
    """Mimic the schedule's mixed send+recv batch between neighbor ranks.

    Each rank pairs with its neighbor and builds a P2POp list with the *send
    first* and the recv second -- the same ordering ``_batch_p2p`` produces for
    ``fwd_sends + bwd_recvs``. With proper coalescing this is deadlock-free;
    without it the symmetric send-first ordering deadlocks. With ``native_batch``,
    the exchange is issued through torchcomms' ``BatchSendRecv`` primitive (the
    one the coalescing path routes to).
    """
    import torch.distributed as dist

    # Pair (0,1), (2,3), ... so every rank both sends and recvs.
    peer = rank ^ 1
    if peer >= world_size:
        return

    send_t = torch.full((1024,), float(rank), device="cuda")
    recv_t = torch.empty((1024,), device="cuda")

    if native_batch:
        comm = dist.group.WORLD._get_backend(send_t.device).get_comm()
        batch = comm.batch_op_create()
        batch.send(send_t, peer)
        batch.recv(recv_t, peer)
        batch.issue(async_op=True).wait()
    else:
        ops = [
            dist.P2POp(dist.isend, send_t, peer),  # send first ...
            dist.P2POp(dist.irecv, recv_t, peer),  # ... recv second (deadlock hazard)
        ]
        works = dist.batch_isend_irecv(ops)
        for w in works:
            w.wait()
    torch.cuda.synchronize()

    expected = float(peer)
    assert torch.allclose(recv_t, torch.full_like(recv_t, expected)), (  # noqa: S101
        f"rank {rank} got {recv_t[0].item()} expected {expected}"
    )


def _worker(
    rank: int, world_size: int, backend_mode: str, native_batch: bool = False
) -> None:
    import torch.distributed as dist

    _init_pg(backend_mode, rank, world_size)
    _run_mixed_batch_p2p(rank, world_size, native_batch=native_batch)
    dist.barrier()
    dist.destroy_process_group()


# Cross-batch ordering hazard: an N-rank ring where each rank does a coalesced
# exchange with its next neighbour then its prev neighbour, each as a *separate*
# blocking batch_isend_irecv. For N>=3 the firsts form a cycle even though each
# batch is coalesced -- backend-independent.


def _run_ring_single_comm(rank: int, world_size: int) -> None:
    """Per-neighbour coalesced exchanges, uniform [next, prev] order, on ONE comm.
    Each exchange is coalesced (safe), but the cross-batch FIFO cycle deadlocks
    for world_size >= 3."""
    import torch.distributed as dist

    grp = dist.group.WORLD
    nxt = (rank + 1) % world_size
    prv = (rank - 1) % world_size
    for peer in (nxt, prv):
        send_t = torch.full((256,), float(rank), device="cuda")
        recv_t = torch.empty((256,), device="cuda")
        ops = [
            dist.P2POp(dist.isend, send_t, peer, group=grp),
            dist.P2POp(dist.irecv, recv_t, peer, group=grp),
        ]
        for w in dist.batch_isend_irecv(ops):
            w.wait()  # block before the next exchange -> models a schedule action
    torch.cuda.synchronize()


def _run_ring_per_direction(rank: int, world_size: int) -> None:
    """The fix (mechanism-level): split the same transfers across two
    per-direction communicators -- ``down`` carries r->r+1 traffic, ``up`` carries
    r->r-1 traffic -- so forward and backward P2P live on separate comms/streams
    and cannot block each other in a FIFO.

    This mirrors what PipelineStage does upstream when
    config.pipeline_per_direction_p2p is set; here we build the two comms with
    plain ``dist`` primitives (new_group over all ranks, equivalent to split_group
    for this pure-PP ring) so the test stays self-contained.
    """
    import torch.distributed as dist

    nxt = (rank + 1) % world_size
    prv = (rank - 1) % world_size
    all_ranks = list(range(world_size))
    down = dist.new_group(ranks=all_ranks)  # r -> r+1 traffic
    up = dist.new_group(ranks=all_ranks)  # r -> r-1 traffic

    send_next = torch.full((256,), float(rank), device="cuda")  # activation -> next
    send_prev = torch.full((256,), float(rank) + 0.5, device="cuda")  # grad -> prev
    recv_prev = torch.empty((256,), device="cuda")  # activation <- prev
    recv_next = torch.empty((256,), device="cuda")  # grad <- next

    # Downstream batch on `down`, upstream batch on `up`. Issue both before
    # waiting: different comms/streams, so they overlap and cannot deadlock.
    down_works = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send_next, nxt, group=down),
            dist.P2POp(dist.irecv, recv_prev, prv, group=down),
        ]
    )
    up_works = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send_prev, prv, group=up),
            dist.P2POp(dist.irecv, recv_next, nxt, group=up),
        ]
    )
    for w in [*down_works, *up_works]:
        w.wait()
    torch.cuda.synchronize()

    assert torch.allclose(  # noqa: S101
        recv_prev, torch.full_like(recv_prev, float(prv))
    ), f"rank {rank} recv_prev={recv_prev[0].item()} expected {float(prv)}"
    assert torch.allclose(  # noqa: S101
        recv_next, torch.full_like(recv_next, float(nxt) + 0.5)
    ), f"rank {rank} recv_next={recv_next[0].item()} expected {float(nxt) + 0.5}"


def _ring_worker(
    rank: int, world_size: int, backend_mode: str, per_direction: bool
) -> None:
    import torch.distributed as dist

    _init_pg(backend_mode, rank, world_size, master_port=29603)
    if per_direction:
        _run_ring_per_direction(rank, world_size)
    else:
        _run_ring_single_comm(rank, world_size)
    dist.barrier()
    dist.destroy_process_group()


def _watch(procs, timeout_s: float) -> str:
    """Join `procs` with a wall-clock timeout; classify the outcome."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(not p.is_alive() for p in procs):
            break
        time.sleep(0.5)

    alive = [p for p in procs if p.is_alive()]
    if alive:
        result = "deadlock"
    elif all(p.exitcode == 0 for p in procs):
        result = "completed"
    else:
        result = "crashed"
    for p in alive:
        p.terminate()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    return result


def _spawn_and_watch(
    backend_mode: str,
    world_size: int,
    timeout_s: float,
    native_batch: bool = False,
) -> str:
    """Spawn `world_size` pair-exchange workers; return the outcome."""
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(r, world_size, backend_mode, native_batch))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    return _watch(procs, timeout_s)


def _spawn_and_watch_ring(
    backend_mode: str,
    world_size: int,
    timeout_s: float,
    per_direction: bool,
) -> str:
    """Spawn `world_size` ring workers (cross-batch ordering); return outcome."""
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_ring_worker, args=(r, world_size, backend_mode, per_direction)
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    return _watch(procs, timeout_s)


class PPTorchCommsP2PDeadlockTest(TestCase):
    @requires_gpus(2)
    def test_nccl_mixed_p2p_completes(self):
        result = _spawn_and_watch("nccl", world_size=2, timeout_s=60)
        self.assertEqual(result, "completed")

    @requires_gpus(2)
    @unittest.skipUnless(HAS_TORCHCOMMS, "torchcomms not installed")
    def test_torchcomms_mixed_p2p_completes(self):
        # Regression guard: torchcomms BackendWrapper coalescing fuses the mixed
        # send+recv batch so it does not deadlock (needs the coalescing fix in
        # the installed torchcomms build).
        result = _spawn_and_watch("torchcomms", world_size=2, timeout_s=45)
        self.assertEqual(result, "completed")

    @requires_gpus(2)
    @unittest.skipUnless(HAS_TORCHCOMMS, "torchcomms not installed")
    def test_torchcomms_native_batch_completes(self):
        # The primitive the coalescing path routes to: BatchSendRecv is
        # deadlock-free for the same mixed exchange.
        result = _spawn_and_watch(
            "torchcomms", world_size=2, timeout_s=45, native_batch=True
        )
        self.assertEqual(result, "completed")

    @requires_gpus(3)
    def test_ring_single_comm_deadlocks_nccl(self):
        # The cross-batch FIFO cycle is backend-independent: nccl coalesces each
        # exchange but still deadlocks, because the hazard is the shared comm.
        result = _spawn_and_watch_ring("nccl", 3, timeout_s=40, per_direction=False)
        self.assertEqual(result, "deadlock")

    @requires_gpus(3)
    @unittest.skipUnless(HAS_TORCHCOMMS, "torchcomms not installed")
    def test_ring_single_comm_deadlocks_torchcomms(self):
        result = _spawn_and_watch_ring(
            "torchcomms", 3, timeout_s=40, per_direction=False
        )
        self.assertEqual(result, "deadlock")

    @requires_gpus(3)
    def test_ring_per_direction_comms_completes_nccl(self):
        result = _spawn_and_watch_ring("nccl", 3, timeout_s=40, per_direction=True)
        self.assertEqual(result, "completed")

    @requires_gpus(3)
    @unittest.skipUnless(HAS_TORCHCOMMS, "torchcomms not installed")
    def test_ring_per_direction_comms_completes_torchcomms(self):
        result = _spawn_and_watch_ring(
            "torchcomms", 3, timeout_s=40, per_direction=True
        )
        self.assertEqual(result, "completed")


if __name__ == "__main__":
    run_tests()
