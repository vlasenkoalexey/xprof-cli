# 🛠️ XProf CLI

CLI-first TPU/GPU profile analysis for AI agents and humans — analyze
JAX / PyTorch-XLA / TensorFlow profiles via the open-source
[xprof](https://github.com/openxla/xprof) profiler. **Also ships an MCP
server** for assistants that prefer structured tools (Claude Code, Gemini,
Cursor, etc.).

> **Renamed from `xprof-mcp`** (2026-07). Same repo, full history preserved;
> GitHub redirects all old URLs, so existing clones, submodules, and MCP
> client configs keep working unchanged. The MCP server is retained as a
> fully supported frontend — but the **CLI is the recommended interface**:
> it needs no running server, no MCP wiring, picks up new profiles on every
> invocation, and works identically from any agent framework that can run
> a shell command.

Tools read `.xplane.pb` traces and XLA/LLO dump directories directly from
disk (in-process conversion); a locally running `xprof` HTTP server is only
needed for the interactive trace-viewer UI, not for analysis.

**See also:** [TPU Performance Optimization Guide](docs/TPU_OPTIMIZATION.md) — practical guide covering the roofline model, common gotchas (dimension alignment, dtype, fusion failures, KV cache, rematerialization), training and inference optimization strategies, and decision trees for diagnosing bottlenecks.

---

## What's here

One shared tool core (`tool_registry.py`, 38 tools) with two frontends:

| Frontend | Entry point | Use it when |
|---|---|---|
| **CLI (recommended)** | `xprof-cli <tool> --logdir=... [--run=...]` | Any agent or human with a shell. No server, no wiring; every invocation sees the latest captures; per-tool `--help` from the same docstrings that drive the MCP schemas. |
| **MCP server (preserved)** | `xprof-mcp` / `python -m xprof_mcp.server.xprof_mcp_server` | Assistants that prefer structured tools (Claude Code agents, Gemini). Identical tool surface. |

Functionality highlights (see the [full tool table](#available-tools) below):

- **Whole-model analyses**: overview, top HLO ops, op profile, memory
  profile, device info, consolidated `get_kpi_metrics`.
- **Roofline** (`get_roofline_model`): per-op compute- vs memory-bound
  classification with ridge points — **caveat-annotated in the output**
  (FLOPs/bytes are XLA cost-model estimates; custom calls are invisible
  to it; there is no communication-bound class; no HW counters on
  TPU v5p/v6e). Read the `caveats` field before acting on the numbers.
- **Communication**: `get_pod_viewer` (ICI/step breakdown),
  `get_megascale_stats` (DCN/multi-slice) — the comm attribution the
  roofline model structurally lacks.
- **Memory attribution**: `get_memory_viewer` — per-buffer HBM map
  (which tensor holds the peak), beyond `get_memory_profile`'s scalar.
- **Host/input pipeline**: `get_input_pipeline`, plus
  `get_framework_op_stats` (device time by framework op name) and
  `get_smart_suggestions` (xprof's automated triage).
- **HLO**: module listing/content, instruction-neighborhood BFS, XLA dump
  reading + stage diffing (`--xla_dump_to` dirs, no trace needed).
- **Pallas / Mosaic kernel internals** (unique to this repo): LLO
  per-functional-unit utilization, Mosaic pipeline stage breakdown with
  DMA `wait_ratio`, VLIW bundle inspection, lowered-MLIR audit, and a
  capture-flag sanity check (`check_kernel_profiling`).
- **In-process by default**: tools read `.xplane.pb` and dump dirs
  directly and run the OSS xprof converters in this process
  (`XPROF_MODE=local`). The old server-backed path is kept as
  `XPROF_MODE=http`.
- **CLI result cache**: per-user SQLite cache (1h TTL) keyed on args
  **and the profile directory's mtime** — re-captured runs are never
  served stale; errors are never cached; `--bypass_cache=True` forces
  recompute.

## Repo history

This repo was **`vlasenkoalexey/xprof-mcp`** until 2026-07: an MCP-only
wrapper. It was renamed (not forked — full commit history, issues, and
stars carried over) when the CLI became the primary interface; GitHub
permanently redirects all old URLs, so existing clones, submodules, and
MCP client configs continue to work without changes. The Python package
directory is still `xprof_mcp/` and the MCP entry point is unchanged —
**all MCP functionality is preserved**; the CLI is simply the recommended
way to consume the same tools.

---

## Quick Start (CLI — recommended)

### 1. Generate a profile

With JAX (see [JAX profiling guide](https://jax.readthedocs.io/en/latest/profiling.html)):

```python
import jax
import jax.numpy as jnp

# Collect a profile into /tmp/profiles/
with jax.profiler.trace("/tmp/profiles/", create_perfetto_link=False):
    y = jnp.dot(jnp.ones((1024, 1024)), jnp.ones((1024, 1024))).block_until_ready()
```

With PyTorch/XLA (see [scaling-book profiling guide](https://jax-ml.github.io/scaling-book/profiling/)):

```python
import torch_xla.debug.profiler as xp
server = xp.start_server(9012)
xp.trace('localhost:9012', '/tmp/profiles/', duration_ms=2000)
```

### 2. Install and run

```bash
cd /path/to/xprof-cli
pip install -e '.[full]'    # xprof converters + tensorflow-cpu

xprof-cli list_runs --logdir=/tmp/profiles
xprof-cli get_overview --logdir=/tmp/profiles --run=<run>
xprof-cli get_roofline_model --logdir=/tmp/profiles --run=<run>
xprof-cli get_llo_utilization --logdir=/tmp/profiles --run=<run> --kernel=<name>

xprof-cli                       # list all 38 commands
xprof-cli get_overview -- --help   # per-tool help
```

No server required. Set `XPROF_LOGDIR` instead of passing `--logdir`
each time. Results print to stdout as JSON; failures exit non-zero with
a JSON error on stderr — script/agent friendly.

### 3. (Optional) the interactive UI

The xprof web UI is the one thing that still wants the server:

```bash
pip install xprof
xprof --logdir=/tmp/profiles --port=8791   # browse http://localhost:8791
```

---

## MCP server (preserved frontend)

The same 38 tools over the Model Context Protocol. Two modes; **HTTP mode
is recommended** for active development — you can restart the MCP server
without restarting your assistant.

#### Mode A: HTTP (recommended — restart-friendly)

The MCP server runs as a standalone HTTP process. Your AI assistant connects to
it via URL. Restart or edit the server anytime without touching your assistant.

**Start the server** (run once; re-run after any code change):

```bash
PYTHONPATH=/path/to/xprof_mcp/.. \
XPROF_URL=http://localhost:8791 \
XPROF_LOGDIR=/tmp/profiles \
XLA_HLO_DUMP_DIR=/tmp/hlo_dumps \  # optional: enables HLO dump tools
MCP_PORT=8792 \
python -m xprof_mcp.server.xprof_mcp_server --transport http \
  > /tmp/xprof_mcp.log 2>&1 &
```

To restart after a code change:
```bash
kill $(pgrep -f "xprof_mcp_server --transport http") 2>/dev/null
# then re-run the start command above
```

**Claude Code** — run once:

```bash
claude mcp add --transport http --scope user xprof http://localhost:8792/mcp
```

No restart needed — Claude Code connects immediately.

**Gemini CLI** — edit `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "xprof": {
      "url": "http://localhost:8792/sse"
    }
  }
}
```

Restart Gemini CLI once to pick up the config change. After that, MCP server
restarts are transparent — no assistant restart needed.

---

#### Mode B: stdio (simpler setup, assistant manages the process)

The assistant spawns the MCP server as a subprocess. Requires an assistant
restart whenever you update the MCP server code.

**Claude Code** — edit `~/.claude/.mcp.json`:

```json
{
  "xprof": {
    "command": "python",
    "args": ["-m", "xprof_mcp.server.xprof_mcp_server"],
    "env": {
      "PYTHONPATH": "/path/to/xprof_mcp/..",
      "XPROF_URL": "http://localhost:8791",
      "XPROF_LOGDIR": "/tmp/profiles",
      "XLA_HLO_DUMP_DIR": "/tmp/hlo_dumps"
    }
  }
}
```

**Gemini CLI** — edit `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "xprof": {
      "command": "python",
      "args": ["-m", "xprof_mcp.server.xprof_mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/xprof_mcp/..",
        "XPROF_URL": "http://localhost:8791",
        "XPROF_LOGDIR": "/tmp/profiles",
        "XLA_HLO_DUMP_DIR": "/tmp/hlo_dumps"
      }
    }
  }
}
```

> **Tip:** If `python` resolves to the wrong interpreter, use the full path
> (e.g. `/usr/bin/python3` or the path to your venv's Python).

---

## Directory Structure

```
xprof_mcp/                 # package dir keeps its pre-rename name on purpose
├── tool_registry.py       # SINGLE source of truth: the 38-tool surface
├── cli/
│   ├── main.py            # xprof-cli entry point (fire over the registry)
│   └── cache.py           # per-user SQLite result cache (mtime-salted TTL)
├── internal/
│   ├── xprof_client.py    # HTTP client (XPROF_MODE=http) + disk access + mode switch
│   ├── local_client.py    # in-process converters (XPROF_MODE=local, default)
│   ├── xprof_data.py      # get_profile_summary, get_op_profile, get_hosts, ...
│   ├── hlo_tools.py       # list_hlo_modules, get_hlo_module_content,
│   │                      #   get_hlo_neighborhood
│   ├── xplane_tools.py    # list_xplane_events, aggregate_xplane_events,
│   │                      #   get_xspace_proto  (require tensorflow)
│   ├── kernel_profiling_tools.py  # check_kernel_profiling, list_kernel_invocations,
│   │                      #   get_llo_utilization, get_kernel_stage_breakdown
│   ├── llo_dump_tools.py  # list_llo_programs, get_llo_schedule_analysis,
│   │                      #   get_llo_static_utilization, get_llo_bundles
│   └── mosaic_tools.py    # get_custom_call_mlir
├── tools/
│   ├── analysis_tools.py  # roofline / pod / megascale / memory_viewer /
│   │                      #   input_pipeline / framework_op_stats /
│   │                      #   smart_suggestions / kpi_metrics
│   ├── list_runs_tool.py           # list_runs
│   ├── get_overview_tool.py        # get_overview
│   ├── get_memory_profile_tool.py  # get_memory_profile
│   └── get_top_hlo_ops_tool.py     # get_top_hlo_ops
├── server/
│   └── xprof_mcp_server.py  # FastMCP entry point (iterates the registry)
└── tests/                 # pytest suite + real v6e trace/dump fixtures
```

---

<a id="available-tools"></a>
## Available Tools

Every tool is simultaneously a CLI subcommand and an MCP tool — one
registry (`tool_registry.py`), two frontends.

| Tool | Description | Needs TF? |
|------|-------------|-----------|
| `list_runs` | List profiling sessions in the logdir | No |
| `get_hosts` | List hosts in a run | No |
| `get_overview` | Step time, device utilization, run environment | No |
| `get_memory_profile` | Peak HBM usage, heap/stack breakdown | No |
| `get_top_hlo_ops` | Top ops by time, FLOPs, bytes accessed | No |
| `get_profile_summary` | Text summary of top ops | No |
| `get_device_information` | Accelerator specs from Roofline Model | No |
| `get_kpi_metrics` | Consolidated headline KPIs (step time, duty cycle, MXU, peak HBM) | No |
| `get_roofline_model` | Per-op compute/memory-bound classification + ridge points — **caveat-annotated** (cost-model FLOPs; custom-call-blind; no comm class) | No |
| `get_pod_viewer` | Pod-level step breakdown + ICI collective stats | No |
| `get_megascale_stats` | Multi-slice DCN collective stats (per-rendezvous) | No |
| `get_memory_viewer` | Per-buffer HBM attribution for one HLO module (which tensor holds peak) | No |
| `get_input_pipeline` | Host-vs-device input-pipeline stall decomposition | No |
| `get_framework_op_stats` | Device time by framework-level op name (JAX/PyTorch/TF) | No |
| `get_utilization_viewer` | Sampled utilization timeline (achieved vs peak over time) | No |
| `get_perf_counters` | Measured HW counters (TPU v7+/Ironwood; empty on v5p/v6e — noted in output) | No |
| `get_smart_suggestions` | xprof's automated bottleneck triage | No |
| `detect_unfused_reshapes` | Automated audit: standalone reshape/copy/transpose ops forcing HBM intermediates | No |
| `list_hlo_modules` | List compiled HLO programs in a run | No |
| `get_hlo_module_content` | Full HLO text for a module | No |
| `get_hlo_neighborhood` | BFS neighborhood of an HLO instruction | No |
| `list_xplane_events` | Filter timeline events by regex | Yes |
| `aggregate_xplane_events` | Stats (count/avg/stddev) per event type | Yes |
| `get_xspace_proto` | Raw XSpace proto bytes or text | Yes |
| `list_hlo_dump_modules` | List modules and stages in an XLA dump dir | No |
| `get_hlo_dump` | Read HLO text at a specific compilation stage | No |
| `diff_hlo_stages` | Unified diff between two compilation stages | No |
| `get_hlo_dump_neighborhood` | BFS neighborhood from a dump file | No |
| `check_kernel_profiling` | Were the kernel-profiling (LLO) flags active at capture? | Yes |
| `list_kernel_invocations` | Pallas/Mosaic custom-call executions + duration stats | Yes |
| `get_llo_utilization` | Per-functional-unit LLO `% util` (MXU/ALU/loads/...) — kernel bottleneck verdict | Yes |
| `get_kernel_stage_breakdown` | Mosaic pipeline stage times (`ep_*`) + DMA wait_ratio | Yes |
| `list_llo_programs` | Programs & pass checkpoints in an `--xla_jf_dump_to` LLO dump dir | No |
| `get_llo_schedule_analysis` | Bundle counts per HLO op / opcode (static attribution) | No |
| `get_llo_static_utilization` | Per-bundle slot occupancy vs capacity + hot ranges | No |
| `get_llo_bundles` | Windowed VLIW bundle listing (by address range / grep) | No |
| `get_llo_fit_summary` | ~30-line kernel fit digest: VMEM vs limit, MXU width, spills, timeline classes, ranked levers + verdict class (`--diff_dump_dir` for deltas) | No |
| `get_device_wall_report` | Device-busy vs wall-clock dual report with labeled ratios + physical-floor audit | Yes |
| `get_custom_call_mlir` | Lowered Mosaic MLIR for a Pallas kernel (noop audit) | No |

"Needs TF?" = requires `tensorflow-cpu` and `XPROF_LOGDIR` to be set.

### Kernel profiling / LLO workflow

Capture with the XProf Kernel Profiling flags to light up the LLO-level tools:

```bash
# Trace-side (LLO utilization tracks, bundle markers, Mosaic ep_* stages):
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true \
                  --xla_xprof_register_llo_debug_info=true" \
python your_pallas_workload.py   # with jax.profiler.trace(logdir)

# Dump-side (per-pass LLO IR: VLIW bundles, per-bundle slot utilization):
LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/jf_dump --xla_jf_dump_llo_text=true \
                  --xla_mosaic_dump_to=/tmp/mosaic_dump" \
python your_pallas_workload.py
export XLA_JF_DUMP_DIR=/tmp/jf_dump
```

Note: the LLO dumper uses its own `--xla_jf_dump_to` flag — XLA's
`--xla_dump_to` does **not** receive LLO dumps. Analysis flow:
`check_kernel_profiling` → `list_kernel_invocations` → `get_llo_utilization`
/ `get_kernel_stage_breakdown` → (drilldown) `list_llo_programs` →
`get_llo_static_utilization` → `get_llo_bundles`, with `get_custom_call_mlir`
as the structural did-it-lower-as-planned audit. `get_llo_fit_summary`
composes the dump-side tools into one ~30-line digest (VMEM allocation vs
limit, MXU matmul width, spill/fill rate, bundle-timeline classification,
ranked levers with ceiling estimates); `get_device_wall_report` keeps the
device-time and wall-time framings side by side (kernel "wins" measured
device-side routinely evaporate 1.6-27x at wall clock when dispatch
dominates). The `Tensor Core` trace
markers carry bundle addresses that plug directly into `get_llo_bundles`'
`address_range`. Trace `% util` values are LLO static slot occupancy over
measured time windows; raw runtime counter sampling requires TPU v7+.
Full guide (capture recipes, failure signatures, flag discovery):
[docs/KERNEL_PROFILING.md](docs/KERNEL_PROFILING.md).

---

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `XPROF_MODE` | *(auto)* | `local` = in-process converters, no server (CLI default); `http` = talk to a running xprof server at `XPROF_URL` (legacy mode); unset = local when converters + logdir are available, else http. |
| `XPROF_LOGDIR` | *(auto-detected)* | Root of the profile captures (the path you'd pass to `xprof --logdir=...`). Required for local mode and disk-based tools; the CLI's `--logdir` flag sets it per-call. Auto-detected from a localhost xprof server process when one is running. GCS paths work with tensorflow installed. |
| `XPROF_URL` | `http://localhost:8791` | URL of the running xprof server — **only used in http mode**. |
| `XLA_JF_DUMP_DIR` | *(unset)* | `--xla_jf_dump_to` directory for the LLO dump tools (`list_llo_programs`, `get_llo_bundles`, ...). |
| `XLA_HLO_DUMP_DIR` | *(empty)* | Directory the MCP dump tools read from (set to the same path as `--xla_dump_to`). Optional — can also be passed per-call as `dump_dir`. |

---

## Profile File Structure

```
<logdir>/
└── plugins/
    └── profile/
        └── <run_name>/
            ├── host0.xplane.pb
            ├── host1.xplane.pb
            └── <module_name>.hlo_proto.pb  (if exported)
```

The `run_name` is what you pass as `run` to all tools.

---

## XLA HLO Dump Workflow

HLO dumps let you inspect the XLA compiler's work **without running the
xprof server** and at every compilation stage, including per-pass diffs.

### Enable dumps

```bash
# Step 1 — tell XLA to write dumps when running your program:
export XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps \
                  --xla_dump_hlo_as_text \
                  --xla_dump_hlo_pass_re=.*"
python your_script.py

# Step 2 — point the tools at those files (CLI):
xprof-cli list_hlo_dump_modules --dump_dir=/tmp/hlo_dumps
# or set XLA_HLO_DUMP_DIR=/tmp/hlo_dumps (used by both CLI and MCP server)
```

**Recommended format: `--xla_dump_hlo_as_text`** (default above). This produces human-readable
HLO text files that all tools here support. Binary proto (`--xla_dump_hlo_as_proto`) requires a
protobuf parser and is not needed for text-based analysis.

Two file naming formats are supported automatically:

**JAX / TensorFlow** (classic `.hlo` files):
```
/tmp/hlo_dumps/
├── module_0001.jit_my_fn.before_optimizations.hlo  ← raw JAX/TF output
├── module_0001.jit_my_fn.after_optimizations.hlo   ← final compiled HLO
├── module_0001.jit_my_fn.after_pass_HloCSE.hlo     ← per-pass (with --xla_dump_hlo_pass_re=.*)
└── ...
```

**PyTorch/XLA** (`torch.compile` with openxla backend — `.txt` files):
```
/tmp/hlo_dumps/
├── module_0006.jit_my_fn.0000.<pipeline>.after_<X>.before_<Y>.txt  ← first stage
├── module_0006.jit_my_fn.0001.<pipeline>.after_<X>.before_<Y>.txt  ← intermediate
├── module_0006.jit_my_fn.0005.<pipeline>.after_<X>.before_<Y>.txt  ← last stage
└── ...
```
Stage aliases: `"before_optimizations"` = first file, `"after_optimizations"` = last file.
Intermediate stages accessible as `"pass_NNNN"` (e.g. `"pass_0003"`).

### Analysis workflow

```
list_hlo_dump_modules()                        # discover modules + stages (both formats)
get_hlo_dump("my_fn", "before_optimizations")  # first/pre-opt stage
get_hlo_dump("my_fn", "after_optimizations")   # last/final stage
diff_hlo_stages("my_fn",                       # what did the optimizer change?
    "before_optimizations", "after_optimizations")
diff_hlo_stages("my_fn",                       # what did one pass do? (JAX format)
    "after_pass_HloCSE", "after_pass_AlgebraicSimplifier")
diff_hlo_stages("my_fn",                       # two consecutive passes (PyTorch/XLA)
    "pass_0003", "pass_0004")
get_hlo_dump_neighborhood("fusion.3", "my_fn") # root-cause a specific op
```

For PyTorch/XLA with many instances of the same module name (e.g. multiple `jit_splash_fn`
compilations), include the module number to target a specific one:
```
get_hlo_dump("module_0006.jit_my_fn", "after_optimizations")
```

### When to use dumps vs. xprof

| Situation | Use |
|-----------|-----|
| No profiling yet, just want to see compiled HLO | HLO dumps |
| Want to see pre-optimization HLO (what JAX emitted) | HLO dumps |
| Debugging a compiler regression (which pass changed something) | HLO dumps |
| Want timing data (which op is slow) | xprof server |
| Want memory profile, step time breakdown | xprof server |
| Want timeline events (kernel durations, gaps) | xprof server + tensorflow |

---

## Recommended Analysis Workflow

Same tool names in both frontends; CLI shown (`xprof-cli <tool>
--logdir=... --run=...`).

**1. Orient** — `list_runs` → `get_kpi_metrics` (headline numbers) →
`get_overview` (bottleneck category, run environment).

**2. Attribute** — `get_top_hlo_ops` (time / FLOPs / bytes leaderboards)
and `get_roofline_model` (per-op compute- vs memory-bound + ridge
points; **read its `caveats` field** — cost-model FLOPs, custom-call
blind, no comm class).

**3. Branch on the bottleneck:**

| Signal | Next tools |
|---|---|
| Device idle / host-bound | `get_input_pipeline` (host stall decomposition), `get_utilization_viewer` (utilization over time) |
| Collective-bound (multi-chip) | `get_pod_viewer` (ICI/step breakdown); multi-slice: `get_megascale_stats` (DCN per-rendezvous) |
| Memory-bound / OOM hunting | `get_memory_profile` (peak + timeline) → `get_memory_viewer` (which buffer holds the peak, per module) |
| Large copy/reshape/transpose in top ops | `detect_unfused_reshapes` (automated HBM-materialization audit) |
| Compute-bound but low efficiency | `get_framework_op_stats` (map to framework op names), then the HLO lane below |
| A Pallas/Mosaic kernel dominates | kernel lane below — roofline is blind here |

**4. HLO root-cause** — `list_hlo_modules` → `get_hlo_module_content` →
`get_hlo_neighborhood(instruction)`; compiler-stage questions:
`list_hlo_dump_modules` / `get_hlo_dump` / `diff_hlo_stages` (from
`--xla_dump_to`, no trace needed).

**5. Kernel lane (Pallas/Mosaic)** — `check_kernel_profiling` (were the
capture flags on? ALWAYS first) → `list_kernel_invocations` →
`get_llo_utilization` (per-unit bottleneck verdict) /
`get_kernel_stage_breakdown` (DMA `wait_ratio`) → drilldown
`list_llo_programs` → `get_llo_static_utilization` → `get_llo_bundles`,
with `get_custom_call_mlir` as the lowering audit. See
[docs/KERNEL_PROFILING.md](docs/KERNEL_PROFILING.md).

**6. Timeline deep-dive (anything else)** — `list_xplane_events` /
`aggregate_xplane_events` with event regexes.

`get_smart_suggestions` is a free second opinion at any point.


---

