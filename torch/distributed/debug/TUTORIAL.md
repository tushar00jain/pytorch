# PyTorch Distributed Debug Server Tutorial

The **Debug Server** is a lightweight HTTP-based diagnostic tool built into
`torch.distributed`. It lets you inspect live distributed training jobs across
all ranks from a single browser tab — collecting stack traces, flight-recorder
events, NCCL traces, profiler captures, wait-counter metrics, and TCPStore
contents without restarting or redeploying your job.

> **⚠ Security:** The debug server is intended for **trusted network
> environments only**. It is not designed to be secure and must not be exposed
> to the public internet.

> **⚠ Experimental:** This feature is experimental and may change at any time.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Configuration Reference](#configuration-reference)
4. [Frontend Endpoints (Browser UI)](#frontend-endpoints-browser-ui)
   - [Home `/`](#home-)
   - [Python Stack Traces `/stacks`](#python-stack-traces-stacks)
   - [py-spy Stack Traces `/pyspy_dump`](#py-spy-stack-traces-pyspy_dump)
   - [FlightRecorder CPU `/fr_trace`](#flightrecorder-cpu-fr_trace)
   - [FlightRecorder CPU JSON `/fr_trace_json`](#flightrecorder-cpu-json-fr_trace_json)
   - [FlightRecorder NCCL `/fr_trace_nccl`](#flightrecorder-nccl-fr_trace_nccl)
   - [FlightRecorder NCCL JSON `/fr_trace_nccl_json`](#flightrecorder-nccl-json-fr_trace_nccl_json)
   - [TorchComms FlightRecorder `/torchcomms_fr_trace`](#torchcomms-flightrecorder-torchcomms_fr_trace)
   - [TorchComms FlightRecorder JSON `/torchcomms_fr_trace_json`](#torchcomms-flightrecorder-json-torchcomms_fr_trace_json)
   - [torch.profiler `/profile`](#torchprofiler-profile)
   - [Wait Counters `/wait_counters`](#wait-counters-wait_counters)
   - [TCPStore Keys `/tcpstore`](#tcpstore-keys-tcpstore)
5. [Worker-Level Endpoints (Per-Rank HTTP API)](#worker-level-endpoints-per-rank-http-api)
   - [`ping`](#ping)
   - [`dump_traceback`](#dump_traceback)
   - [`pyspy_dump`](#pyspy_dump)
   - [`fr_trace_json`](#fr_trace_json)
   - [`dump_nccl_trace_json`](#dump_nccl_trace_json)
   - [`dump_nccl_trace_pickle`](#dump_nccl_trace_pickle)
   - [`torchcomms_fr_trace_json`](#torchcomms_fr_trace_json)
   - [`torch_profile`](#torch_profile)
   - [`wait_counter_values`](#wait_counter_values)
6. [Periodic Dumping](#periodic-dumping)
7. [Registering Custom Handlers](#registering-custom-handlers)
   - [Python Custom Handler](#python-custom-handler)
   - [C++ Custom Handler](#c-custom-handler)
   - [Custom Frontend Handler (DebugHandler)](#custom-frontend-handler-debughandler)
8. [Using with TorchElastic](#using-with-torchelastic)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The debug server has a **two-tier** architecture:

```
┌───────────────────────────────────┐
│     Frontend Server (rank 0)      │
│  HTTP server on a fixed port      │
│  Renders HTML dashboards          │
│  Fans out requests to all ranks   │
└──────────┬────────────────────────┘
           │  HTTP POST /handler/<endpoint>?<args>
           ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ WorkerServer (0) │  │ WorkerServer (1) │  │ WorkerServer (N) │
│ Per-rank C++     │  │ Per-rank C++     │  │ Per-rank C++     │
│ HTTP server      │  │ HTTP server      │  │ HTTP server      │
│ Exposes handlers │  │ Exposes handlers │  │ Exposes handlers │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

- **WorkerServer** (`_WorkerServer`): A lightweight C++ HTTP server started on
  every rank. It serves *handler* endpoints registered via `_register_handler()`
  (Python) or `RegisterHandler` (C++). Each handler receives a `_Request` and
  writes to a `_Response`.

- **FrontendServer**: A Python HTTP server that starts only on **rank 0**. It
  aggregates data from all worker servers and renders HTML dashboards using
  Jinja2 templates.

- **TCPStore**: Workers publish their addresses (hostname + port) to the
  existing `TCPStore` so the frontend knows where to find each rank.

---

## Quick Start

### Prerequisites

Install required dependencies (not bundled with PyTorch by default):

```bash
pip install jinja2 aiohttp tabulate
```

`aiohttp` is optional — the server falls back to `requests` + thread pool if
`aiohttp` is unavailable.

### Basic Usage

Call `start_debug_server()` **after** `dist.init_process_group()` on every rank:

```python
import torch
import torch.distributed as dist
from torch.distributed.debug import start_debug_server, stop_debug_server

dist.init_process_group("nccl")

# Start the debug server on all ranks.
# The frontend (browser UI) is served on rank 0 at port 25999.
start_debug_server(port=25999)

# ... your training loop ...

stop_debug_server()
dist.destroy_process_group()
```

Then open `http://<rank0-hostname>:25999` in a browser.

### Minimal `torchrun` Example

```python
# train.py
import torch
import torch.distributed as dist
from torch.distributed.debug import start_debug_server

def main():
    dist.init_process_group("nccl")
    start_debug_server()

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    model = torch.nn.Linear(10, 10).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    for _ in range(1000):
        x = torch.randn(32, 10, device=device)
        loss = model(x).sum()
        loss.backward()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

```bash
torchrun --nproc-per-node=2 train.py
# Open http://localhost:25999
```

---

## Configuration Reference

`start_debug_server()` accepts the following parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `port` | `int` | `25999` | Port for the frontend HTTP server (rank 0 only). |
| `worker_port` | `int` | `0` | Port for the per-rank worker server. `0` = ephemeral. |
| `start_method` | `str \| None` | `None` | Multiprocessing start method for the frontend process (`"fork"`, `"spawn"`, or `"forkserver"`). `"spawn"` is recommended with CUDA. |
| `dump_dir` | `str \| None` | `None` | Directory for periodic debug dumps. `None` disables dumping. |
| `dump_interval` | `float` | `60.0` | Seconds between periodic dumps. |
| `enabled_dumps` | `set[str] \| None` | `None` | Which handlers to dump (e.g. `{"stacks", "fr_trace"}`). `None` = default set. |
| `handlers` | `list[DebugHandler] \| None` | `None` | Custom handler list. `None` = all default handlers. |
| `fetch_timeout` | `float` | `60.0` | Timeout (seconds) when the frontend fetches data from workers. |

### Environment Variables

The debug server reads these standard environment variables:

| Variable | Description |
|---|---|
| `RANK` | Current process rank. |
| `WORLD_SIZE` | Total number of ranks. |
| `MASTER_ADDR` | Address of the TCPStore master. |
| `MASTER_PORT` | Port of the TCPStore master. |

These are set automatically by `torchrun` / TorchElastic.

---

## Frontend Endpoints (Browser UI)

The frontend server (rank 0) exposes the following endpoints. Each one
aggregates data from **all** worker ranks and renders an HTML page.

### Home `/`

The landing page with navigation links to all other endpoints.

### Python Stack Traces `/stacks`

Calls `dump_traceback` on every worker to collect Python stack traces using
`faulthandler.dump_traceback()`. Useful for diagnosing hangs or deadlocks
without attaching a debugger.

**What you see:** A `<pre>` block per rank showing the Python call stack of
every thread.

### py-spy Stack Traces `/pyspy_dump`

Calls `pyspy_dump` on every worker. This uses [py-spy](https://github.com/benfred/py-spy)
to dump both Python and optionally native (C/C++) stack traces.

**Query parameters:**

| Parameter | Description |
|---|---|
| `native=1` | Include native (C/C++) frames in the stack trace. |
| `subprocesses=1` | Include subprocesses. |

> **Note:** `py-spy` must be installed and the process must have `SYS_PTRACE`
> capability. The `nonblocking=1` flag is added automatically.

### FlightRecorder CPU `/fr_trace`

Fetches CPU-side flight recorder data from `fr_trace_json` on all workers,
parses it into structured tables (Groups, Memberships, Collectives, NCCL
Calls), and renders them as HTML tables.

### FlightRecorder CPU JSON `/fr_trace_json`

Same data as `/fr_trace` but rendered as raw formatted JSON per rank.

### FlightRecorder NCCL `/fr_trace_nccl`

Fetches NCCL-side flight recorder data from `dump_nccl_trace_json` (with
`onlyactive=true`) on all workers and renders structured HTML tables.

### FlightRecorder NCCL JSON `/fr_trace_nccl_json`

Same data as `/fr_trace_nccl` but rendered as raw formatted JSON per rank.

### TorchComms FlightRecorder `/torchcomms_fr_trace`

Fetches TorchComms flight recorder data from `torchcomms_fr_trace_json` (with
`onlyactive=true`) on all workers. Renders the same structured tables as
the FlightRecorder views (Groups, Memberships, Collectives, NCCL Calls).

### TorchComms FlightRecorder JSON `/torchcomms_fr_trace_json`

Same data as `/torchcomms_fr_trace` but rendered as raw formatted JSON.

### torch.profiler `/profile`

Triggers `torch.profiler.profile()` on every worker for a configurable
duration, then returns the Chrome trace JSON. The frontend page provides a
**"View" button** per rank that opens the trace directly in
[Perfetto UI](https://ui.perfetto.dev/) (no download required).

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `duration` | `1` | Profiling duration in seconds (1–60). |

### Wait Counters `/wait_counters`

Fetches `wait_counter_values` from all workers and renders the JSON data.
Wait counters track time spent waiting in collective operations, useful for
identifying stragglers and load imbalance.

### TCPStore Keys `/tcpstore`

Connects to the TCPStore and lists all keys with their values (truncated to
100 characters). Useful for inspecting the state of the distributed key-value
store.

---

## Worker-Level Endpoints (Per-Rank HTTP API)

Each `WorkerServer` instance exposes handler endpoints at:

```
POST http://<worker-host>:<worker-port>/handler/<handler_name>?<params>
```

These are the built-in handlers:

### `ping`

**Response:** `"pong"` (text/plain, 200)

Simple health check. Useful for verifying that a worker's HTTP server is
responsive.

```bash
curl -X POST http://worker-host:port/handler/ping
# pong
```

### `dump_traceback`

**Response:** Python stack traces of all threads (text/plain)

Uses `faulthandler.dump_traceback()` to capture every thread's Python stack.
Requires the GIL.

```bash
curl -X POST http://worker-host:port/handler/dump_traceback
```

### `pyspy_dump`

**Response:** py-spy stack dump output (text/plain)

Runs `py-spy dump --pid <pid>` to capture stack traces without stopping the
process.

**Parameters:**

| Parameter | Description |
|---|---|
| `native` | Include native C/C++ frames. |
| `subprocesses` | Include subprocess stacks. |
| `nonblocking` | Use non-blocking mode (recommended). |

```bash
curl -X POST "http://worker-host:port/handler/pyspy_dump?nonblocking=1&native=1"
```

### `fr_trace_json`

**Response:** CPU flight-recorder trace (application/json)

Returns the flight-recorder ring buffer contents as JSON. This includes all
recorded collective operations, their metadata, and timing information.

```bash
curl -X POST http://worker-host:port/handler/fr_trace_json
```

### `dump_nccl_trace_json`

**Response:** NCCL flight-recorder trace (application/json)

Returns NCCL-level trace data. Only available when built with NCCL support.

**Parameters:**

| Parameter | Values | Default | Description |
|---|---|---|---|
| `includecollectives` | `true`/`false` | `true` | Include collective details. |
| `onlyactive` | `true`/`false` | `false` | Only include active (in-flight) operations. |

```bash
curl -X POST "http://worker-host:port/handler/dump_nccl_trace_json?onlyactive=true"
```

### `dump_nccl_trace_pickle`

**Response:** NCCL trace in pickle format (application/octet-stream)

Returns the NCCL trace as a Python pickle. Useful for programmatic analysis.

**Parameters:**

| Parameter | Values | Default | Description |
|---|---|---|---|
| `includecollectives` | `true`/`false` | `true` | Include collective details. |
| `includestacktraces` | `true`/`false` | `true` | Include stack traces. |
| `onlyactive` | `true`/`false` | `false` | Only include active operations. |

```bash
curl -X POST "http://worker-host:port/handler/dump_nccl_trace_pickle?onlyactive=true" \
     --output trace.pkl
```

### `torchcomms_fr_trace_json`

**Response:** TorchComms flight-recorder trace (application/json)

Returns TorchComms-level trace data as JSON. Similar to `dump_nccl_trace_json`
but for the TorchComms communication layer.

**Parameters:**

| Parameter | Values | Default | Description |
|---|---|---|---|
| `onlyactive` | `true`/`false` | `false` | Only include active operations. |

```bash
curl -X POST "http://worker-host:port/handler/torchcomms_fr_trace_json?onlyactive=true"
```

### `torch_profile`

**Response:** Chrome trace JSON (application/json)

Runs `torch.profiler.profile()` for the specified duration and returns the
Chrome trace format JSON. This captures all PyTorch operations (CPU ops,
CUDA kernels, memory allocations, etc.).

**Parameters:**

| Parameter | Description |
|---|---|
| `duration` | Profiling duration in seconds (required). |

```bash
curl -X POST "http://worker-host:port/handler/torch_profile?duration=5" \
     --output trace.json
# Open trace.json in chrome://tracing or https://ui.perfetto.dev
```

### `wait_counter_values`

**Response:** Wait counter values (application/json)

Returns a JSON object with wait-counter metrics that track time spent waiting
in distributed collective operations.

```bash
curl -X POST http://worker-host:port/handler/wait_counter_values
```

---

## Periodic Dumping

Enable periodic dumping to automatically save debug data to disk at regular
intervals. This is useful for post-mortem analysis when a job hangs or crashes.

```python
start_debug_server(
    dump_dir="/shared/nfs/debug_dumps",
    dump_interval=120.0,  # dump every 2 minutes
    enabled_dumps={"stacks", "fr_trace", "pyspy_dump", "wait_counters", "tcpstore"},
)
```

Dump files are saved as `<handler_name>_<timestamp>.txt` (e.g.,
`stacks_20250330_192000.txt`).

**Handlers that support dumping:**

| Handler | Dump filename | Content |
|---|---|---|
| `StacksHandler` | `stacks` | Python stack traces for all ranks. |
| `PySpyHandler` | `pyspy_dump` | py-spy stack dumps (nonblocking) for all ranks. |
| `FlightRecorderHandler` | `fr_trace` | CPU + NCCL flight-recorder tables. |
| `TorchCommsFlightRecorderHandler` | `torchcomms_fr_trace` | TorchComms flight-recorder tables. |
| `WaitCountersHandler` | `wait_counters` | Wait counter JSON for all ranks. |
| `TCPStoreHandler` | `tcpstore` | All TCPStore key-value pairs. |

By default (when `enabled_dumps=None`), only `"stacks"` and `"fr_trace"` are
enabled.

---

## Registering Custom Handlers

You can extend the debug server with custom handlers at both the worker level
(per-rank data collection) and the frontend level (aggregation + UI).

### Python Custom Handler

Register a handler on every worker that will be callable via the worker's HTTP
API:

```python
from torch._C._distributed_c10d import _register_handler, _Request, _Response
import json

def my_custom_handler(req: _Request, resp: _Response) -> None:
    # req.get_param("key") retrieves query parameters
    # req.body() returns the POST body as bytes
    data = {
        "gpu_memory_allocated": torch.cuda.memory_allocated(),
        "gpu_memory_reserved": torch.cuda.memory_reserved(),
    }
    resp.set_content(json.dumps(data), "application/json")
    resp.set_status(200)

_register_handler("gpu_memory", my_custom_handler)
```

The handler is now accessible at:
```bash
curl -X POST http://worker-host:port/handler/gpu_memory
```

### C++ Custom Handler

Register a handler from C++ for minimal overhead:

```cpp
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

namespace c10d::control_plane {
namespace {
RegisterHandler myHandler{
    "my_metric",
    [](const Request& req, Response& res) {
        // req.getParam("key") retrieves query parameters
        // req.body() returns the POST body
        res.setContent("{\"status\": \"ok\"}", "application/json");
        res.setStatus(200);
    }};
} // namespace
} // namespace c10d::control_plane
```

### Custom Frontend Handler (DebugHandler)

Create a custom frontend page that aggregates data from all workers:

```python
from torch.distributed.debug._frontend import (
    DebugHandler,
    fetch_all,
    NavLink,
    Route,
)

CUSTOM_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}GPU Memory{% endblock %}</h1>
{% endblock %}
{% block content %}
    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ format_json(resp.text) }}</pre>
        {% endif %}
    {% endfor %}
{% endblock %}
"""


class GPUMemoryHandler(DebugHandler):
    def routes(self):
        return [Route("/gpu_memory", self._handle)]

    def nav_links(self):
        return [NavLink("/gpu_memory", "GPU Memory")]

    def templates(self):
        return {"gpu_memory.html": CUSTOM_TEMPLATE}

    def _handle(self, req):
        addrs, resps = fetch_all("gpu_memory", timeout=self.fetch_timeout)
        return req.frontend.render_template(
            "gpu_memory.html", addrs=addrs, resps=resps
        )

    def dump(self):
        addrs, resps = fetch_all("gpu_memory", timeout=self.fetch_timeout)
        parts = []
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            parts.append(resp.text if resp.status_code == 200 else f"Error: {resp.status_code}")
        return "\n".join(parts)

    def dump_filename(self):
        return "gpu_memory"
```

Then pass it to `start_debug_server`:

```python
from torch.distributed.debug._debug_handlers import default_handlers

handlers = default_handlers() + [GPUMemoryHandler()]
start_debug_server(handlers=handlers)
```

---

## Using with TorchElastic

The debug server integrates with TorchElastic's `worker_main` context manager
for the control-plane-based worker server (unix socket mode):

```python
from torch.distributed.elastic.control_plane import worker_main

@worker_main()
def main():
    # WorkerServer is automatically started using the
    # TORCH_WORKER_SERVER_SOCKET environment variable.
    # All registered handlers are accessible via the unix socket.
    pass

if __name__ == "__main__":
    main()
```

This is orthogonal to `start_debug_server()`. The TorchElastic control plane
exposes the same per-rank handler endpoints via a unix socket, while
`start_debug_server()` provides the HTTP-based frontend aggregation.

---

## Troubleshooting

### `AssertionError: debug server already started`

`start_debug_server()` was called twice. Ensure you only call it once per
process.

### Workers fail to respond (408/503 errors)

- Check that all ranks have started and registered their addresses in TCPStore.
- Increase `fetch_timeout` for large clusters or slow networks.
- Verify network connectivity between rank 0 and all worker hosts.

### py-spy returns errors

- Ensure `py-spy` is installed: `pip install py-spy`
- The process may need `SYS_PTRACE` capability. In Docker:
  ```bash
  docker run --cap-add SYS_PTRACE ...
  ```
- On Linux, you may need to set:
  ```bash
  echo 0 > /proc/sys/kernel/yama/ptrace_scope
  ```

### Frontend server not reachable

- Ensure port `25999` (or your custom port) is not blocked by a firewall.
- The frontend only starts on **rank 0**. Connect to the rank 0 host.
- Check the logs for `Frontend server started on port <port>`.

### `spawn` start method required with CUDA

If you see CUDA re-initialization errors, use:
```python
start_debug_server(start_method="spawn")
```

This ensures the frontend server process does not inherit CUDA state from the
parent process.
