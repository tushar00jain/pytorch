# Owner(s): ["oncall: distributed"]

"""
End-to-end validation for per-direction PP communicators in
``torch.distributed.pipelining.PipelineStage``, enabled via
``torch.distributed.config.pipeline_per_direction_p2p``.

It runs a real 4-stage ``Schedule1F1B`` pipeline through the actual
``PipelineStage`` + schedule code -- once with the single shared communicator
(default) and once with per-direction communicators -- and checks:

* both runs complete (no deadlock), and
* the per-stage loss and gradient norm are bitwise identical between the two
  runs. Per-direction P2P only changes which communicator carries the bytes, not
  the math, so identical numerics is the correctness bar.

Needs >= 4 GPUs (some cases >= 2). Spawns its own worker processes.
"""

from __future__ import annotations

import os
import time
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    import torchcomms  # noqa: F401

    HAS_TORCHCOMMS = True
except ImportError:
    HAS_TORCHCOMMS = False


DIM = 16
MICRO_BATCH = 4
N_MICROBATCHES = 4
GLOBAL_BATCH = MICRO_BATCH * N_MICROBATCHES
MASTER_PORT = 29613


def requires_gpus(n: int):
    return unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= n,
        f"needs >= {n} GPUs",
    )


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(output, target)


def _worker(rank: int, world_size: int, backend_mode: str, per_direction: bool, q):
    import torch.distributed as dist
    import torch.distributed.config as dist_config
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import Schedule1F1B

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(MASTER_PORT)

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    if backend_mode == "torchcomms":
        import torchcomms  # noqa: F401

        dist_config.use_torchcomms = True

    # Per-direction P2P is controlled by the global config (no constructor arg).
    dist_config.pipeline_per_direction_p2p = per_direction

    # device_id binds the default PG so split_group can duplicate the PP comm.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )

    # Deterministic, reproducible init/data across both runs.
    torch.manual_seed(1234 + rank)
    torch.use_deterministic_algorithms(True, warn_only=False)

    module = _Layer().to(device)
    example = torch.randn(MICRO_BATCH, DIM, device=device)
    stage = PipelineStage(
        module,
        rank,
        world_size,
        device,
        input_args=(example,),
        group=dist.group.WORLD,
    )
    schedule = Schedule1F1B(stage, n_microbatches=N_MICROBATCHES, loss_fn=_loss_fn)

    # Fixed input on the first stage, fixed target on the last stage.
    gen = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(GLOBAL_BATCH, DIM, device=device, generator=gen)
    target = torch.randn(GLOBAL_BATCH, DIM, device=device, generator=gen)

    losses: list[torch.Tensor] = []
    if rank == 0:
        schedule.step(x)
    elif rank == world_size - 1:
        schedule.step(target=target, losses=losses)
    else:
        schedule.step()
    torch.cuda.synchronize()

    # Report this stage's loss (last stage only) and gradient norm.
    loss_val = (
        float(torch.stack([loss_.detach() for loss_ in losses]).sum().item())
        if losses
        else float("nan")
    )
    grad_norm = float(
        torch.norm(
            torch.stack(
                [
                    p.grad.detach().norm()
                    for p in module.parameters()
                    if p.grad is not None
                ]
            )
        ).item()
        if any(p.grad is not None for p in module.parameters())
        else 0.0
    )
    q.put((rank, loss_val, grad_norm))

    dist.barrier()
    dist.destroy_process_group()


def _run(backend_mode: str, world_size: int, per_direction: bool, timeout_s: float):
    """Spawn a pipeline run; return ('completed'|'deadlock'|'crashed', results)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker, args=(r, world_size, backend_mode, per_direction, q)
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()

    results: dict[int, tuple[float, float]] = {}
    deadline = time.time() + timeout_s
    while time.time() < deadline and len(results) < world_size:
        try:
            rank, loss_val, grad_norm = q.get(timeout=1.0)
            results[rank] = (loss_val, grad_norm)
        except Exception:
            if all(not p.is_alive() for p in procs) and q.empty():
                break

    # Give procs a moment to reach their final barrier / exit.
    end = time.time() + 10
    while time.time() < end and any(p.is_alive() for p in procs):
        time.sleep(0.5)

    alive = [p for p in procs if p.is_alive()]
    if alive:
        status = "deadlock"
    elif all(p.exitcode == 0 for p in procs):
        status = "completed"
    else:
        status = "crashed"
    for p in alive:
        p.terminate()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    return status, results


def _compare(backend_mode: str, world_size: int = 4, timeout_s: float = 90.0):
    base_status, base = _run(backend_mode, world_size, False, timeout_s)
    fix_status, fix = _run(backend_mode, world_size, True, timeout_s)
    return base_status, base, fix_status, fix


def _config_activation_worker(rank: int, world_size: int, q):
    """Enable per-direction via the global config (no explicit constructor arg)
    and report whether the stage built two distinct direction communicators."""
    import torch.distributed as dist
    import torch.distributed.config as dist_config
    from torch.distributed.pipelining import PipelineStage

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(MASTER_PORT + 1)

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )

    # Turn the feature on globally via config; PipelineStage takes no arg for it.
    dist_config.pipeline_per_direction_p2p = True

    module = _Layer().to(device)
    example = torch.randn(MICRO_BATCH, DIM, device=device)
    stage = PipelineStage(
        module, rank, world_size, device, input_args=(example,), group=dist.group.WORLD
    )
    activated = (
        stage.p2p_per_direction
        and stage._fwd_group is not dist.group.WORLD
        and stage._bwd_group is not dist.group.WORLD
        and stage._fwd_group is not stage._bwd_group
    )
    q.put((rank, bool(activated)))
    dist.barrier()
    dist.destroy_process_group()


def _run_config_activation(world_size: int = 2, timeout_s: float = 60.0):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_config_activation_worker, args=(r, world_size, q))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    deadline = time.time() + timeout_s
    while time.time() < deadline and len(results) < world_size:
        try:
            rank, activated = q.get(timeout=1.0)
            results[rank] = activated
        except Exception:
            if all(not p.is_alive() for p in procs) and q.empty():
                break
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    return results


class PerDirectionPipelineStageTest(TestCase):
    def _assert_match(self, base, fix):
        self.assertEqual(set(base), set(fix))
        for rank in base:
            b_loss, b_gn = base[rank]
            f_loss, f_gn = fix[rank]
            # Bitwise identical: per-direction P2P is a non-computation change.
            if b_loss == b_loss:  # not NaN (only the last stage reports a loss)
                self.assertEqual(b_loss, f_loss)
            self.assertEqual(b_gn, f_gn)

    @requires_gpus(4)
    def test_per_direction_matches_single_comm_nccl(self):
        base_status, base, fix_status, fix = _compare("nccl")
        self.assertEqual(base_status, "completed")
        self.assertEqual(fix_status, "completed")
        self._assert_match(base, fix)

    @requires_gpus(4)
    @unittest.skipUnless(HAS_TORCHCOMMS, "torchcomms not installed")
    def test_per_direction_matches_single_comm_torchcomms(self):
        base_status, base, fix_status, fix = _compare("torchcomms")
        self.assertEqual(base_status, "completed")
        self.assertEqual(fix_status, "completed")
        self._assert_match(base, fix)

    @requires_gpus(2)
    def test_config_flag_activates_without_explicit_arg(self):
        # torch.distributed.config.pipeline_per_direction_p2p turns the feature
        # on with no constructor arg -- frameworks need no code change.
        results = _run_config_activation(world_size=2)
        self.assertTrue(results and all(results.values()), results)


if __name__ == "__main__":
    run_tests()
