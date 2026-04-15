# XProf MCP Server (OSS)

An MCP server that lets AI assistants (Gemini, JetSki, etc.) analyze
JAX / PyTorch-XLA / TensorFlow profiles on TPUs and GPUs via the open-source
[xprof](https://github.com/openxla/xprof) profiler.

This is the OSS counterpart of the internal `xprof_mcp` server. Instead of
talking to Google-internal infrastructure, it connects to a locally running
`xprof` HTTP server and reads `.xplane.pb` files directly from disk.

---

## Quick Start

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

### 2. Start the xprof server

```bash
pip install xprof
xprof --logdir=/tmp/profiles --port=8791
```

Open `http://localhost:8791` to verify the UI loads and your profiles appear.

### 3. Install the MCP server

```bash
cd /path/to/xprof_mcp
pip install -r requirements.txt
# Optional: for XPlane timeline tools
pip install tensorflow-cpu
```

### 4. Connect to your AI assistant

There are two modes. **SSE mode is recommended** for active development — it lets
you restart or update the MCP server without restarting your AI assistant.

---

#### Mode A: SSE (recommended — restart-friendly)

The MCP server runs as a standalone HTTP process. Your AI assistant connects to
it via URL. Restart or edit the server anytime without touching your assistant.

**Start the server** (run once; re-run after any code change):

```bash
PYTHONPATH=/path/to/xprof_mcp/.. \
XPROF_URL=http://localhost:8791 \
XPROF_LOGDIR=/tmp/profiles \
MCP_PORT=8792 \
python -m xprof_mcp.server.xprof_mcp_server --transport sse \
  > /tmp/xprof_mcp.log 2>&1 &
```

To restart after a code change:
```bash
kill $(pgrep -f "xprof_mcp_server --transport sse") 2>/dev/null
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
      "XPROF_LOGDIR": "/tmp/profiles"
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
        "XPROF_LOGDIR": "/tmp/profiles"
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
xprof_mcp/
├── internal/
│   ├── xprof_client.py    # HTTP client for the xprof server + disk access
│   ├── xprof_data.py      # get_profile_summary, get_hlo_op_profile, get_hosts
│   ├── hlo_tools.py       # list_hlo_modules, get_hlo_module_content,
│   │                      #   get_hlo_neighborhood
│   └── xplane_tools.py    # list_xplane_events, aggregate_xplane_events,
│                          #   get_xspace_proto  (require tensorflow)
├── tools/
│   ├── list_runs_tool.py           # list_runs
│   ├── get_overview_tool.py        # get_overview
│   ├── get_memory_profile_tool.py  # get_memory_profile
│   └── get_top_hlo_ops_tool.py     # get_top_hlo_ops
└── server/
    └── xprof_mcp_server.py  # FastMCP entry point
```

---

## Available MCP Tools

| Tool | Description | Needs TF? |
|------|-------------|-----------|
| `list_runs` | List profiling sessions on the server | No |
| `get_hosts` | List hosts in a run | No |
| `get_overview` | Step time, device utilization, run environment | No |
| `get_memory_profile` | Peak HBM usage, heap/stack breakdown | No |
| `get_top_hlo_ops` | Top ops by time, FLOPs, bytes accessed | No |
| `get_profile_summary` | Text summary of top ops | No |
| `get_device_information` | Accelerator specs from Roofline Model | No |
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

"Needs TF?" = requires `tensorflow-cpu` and `XPROF_LOGDIR` to be set.

---

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `XPROF_URL` | `http://localhost:8791` | URL of the running xprof server |
| `XPROF_LOGDIR` | *(empty)* | Path passed to `xprof --logdir=...` |
| `XLA_HLO_DUMP_DIR` | *(empty)* | Path passed to `--xla_dump_to=...` |

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
# Before running your JAX/PyTorch-XLA program:
export XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps \
                  --xla_dump_hlo_as_text \
                  --xla_dump_hlo_pass_re=.*"
export XLA_HLO_DUMP_DIR=/tmp/hlo_dumps
python your_script.py
```

Files produced:
```
/tmp/hlo_dumps/
├── module_0001.jit_my_fn.before_optimizations.hlo  ← raw JAX/TF output
├── module_0001.jit_my_fn.after_optimizations.hlo   ← final compiled HLO
├── module_0001.jit_my_fn.after_pass_HloCSE.hlo
├── module_0001.jit_my_fn.after_pass_AlgebraicSimplifier.hlo
├── module_0001.jit_my_fn.hlo.pb                    ← binary proto
└── ...
```

### Analysis workflow

```
list_hlo_dump_modules()                        # discover modules + stages
get_hlo_dump("my_fn", "before_optimizations")  # see what JAX produced
get_hlo_dump("my_fn", "after_optimizations")   # see what XLA compiled
diff_hlo_stages("my_fn",                       # what did the optimizer change?
    "before_optimizations", "after_optimizations")
diff_hlo_stages("my_fn",                       # what did one pass do?
    "after_pass_HloCSE", "after_pass_AlgebraicSimplifier")
get_hlo_dump_neighborhood("fusion.3", "my_fn") # root-cause a specific op
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

1. **`list_runs()`** — find available sessions
2. **`get_overview(run)`** — step time, utilization, bottleneck category
3. **`get_top_hlo_ops(run)`** — which ops use the most time / compute / memory
4. **`list_hlo_modules(run)`** → **`get_hlo_module_content(run, module)`** — inspect compiled HLO
5. **`get_hlo_neighborhood(run, instruction_name)`** — root-cause a slow op
6. **`get_memory_profile(run)`** — memory pressure analysis
7. **`list_xplane_events(run)`** / **`aggregate_xplane_events(run)`** — timeline deep-dive

## Differences from Internal xprof_mcp

| Feature | Internal | OSS (this) |
|---------|----------|------------|
| Backend | `xprof_analysis_client` RPC | HTTP to local xprof server |
| Session IDs | Opaque IDs from xprof service | Run directory names |
| `find_session` | XManager / Borg / F1 query | `list_runs` (HTTP) |
| op_profile | Binary proto via RPC | `hlo_stats` JSON endpoint |
| HLO content | `xla_client.HloModule` | `graph_viewer` HTTP endpoint |

---

