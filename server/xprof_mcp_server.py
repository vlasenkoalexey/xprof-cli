"""OSS XProf MCP Server entry point.

Exposes XProf profiling data through the Model Context Protocol (MCP) so that
AI assistants (Claude, Gemini, etc.) can analyze JAX/PyTorch/TensorFlow
profiles on TPUs and GPUs.

Prerequisites:
  1. Install the xprof server:
       pip install xprof
  2. Start the xprof server pointing at your profile data:
       xprof --logdir=<path_to_profiles> --port=8791
  3. (Optional) For XPlane/raw timeline tools, also install tensorflow:
       pip install tensorflow-cpu

Configuration (environment variables):
  XPROF_URL    URL of the running xprof server (default: http://localhost:8791)
  XPROF_LOGDIR Path to the logdir used with `xprof --logdir=...`
               Required only for raw XPlane tools (list_xplane_events,
               aggregate_xplane_events, get_xspace_proto).

Usage:
  python -m xprof_mcp.server.xprof_mcp_server

Usage (stdio, default — Claude Code manages the process):
  python -m xprof_mcp.server.xprof_mcp_server

Usage (HTTP — recommended standalone mode, restart without restarting Claude Code):
  python -m xprof_mcp.server.xprof_mcp_server --transport http --port 8792

Add to Claude Code MCP config (~/.claude/settings.json):

  Stdio mode (Claude Code manages the process):
  {
    "mcpServers": {
      "xprof": {
        "command": "python",
        "args": ["-m", "xprof_mcp.server.xprof_mcp_server"],
        "env": {
          "XPROF_URL": "http://localhost:8791",
          "XPROF_LOGDIR": "/path/to/your/logdir"
        }
      }
    }
  }

  HTTP mode (recommended — server runs independently, stateless, restart-friendly):
    claude mcp add --transport http --scope user xprof http://localhost:8792/mcp
"""

import os
import sys

from mcp import types
from mcp.server import fastmcp

from xprof_mcp.internal import hlo_dump_tools
from xprof_mcp.internal import hlo_tools
from xprof_mcp.internal import xplane_tools
from xprof_mcp.internal import xprof_data
from xprof_mcp.tools import get_memory_profile_tool
from xprof_mcp.tools import get_overview_tool
from xprof_mcp.tools import get_top_hlo_ops_tool
from xprof_mcp.tools import list_runs_tool

mcp = fastmcp.FastMCP(
    "XProf",
    port=int(os.environ.get("MCP_PORT", "8792")),
    stateless_http=True,
)

# ---------------------------------------------------------------------------
# Discovery / listing tools
# ---------------------------------------------------------------------------
mcp.add_tool(list_runs_tool.list_runs)
mcp.add_tool(xprof_data.get_hosts)

# ---------------------------------------------------------------------------
# Performance summary tools
# ---------------------------------------------------------------------------
mcp.add_tool(get_overview_tool.get_overview)
mcp.add_tool(get_memory_profile_tool.get_memory_profile)
mcp.add_tool(get_top_hlo_ops_tool.get_top_hlo_ops)
mcp.add_tool(xprof_data.get_op_profile)
mcp.add_tool(xprof_data.get_profile_summary)
mcp.add_tool(xprof_data.get_device_information)

# ---------------------------------------------------------------------------
# HLO analysis tools
# ---------------------------------------------------------------------------
mcp.add_tool(hlo_tools.list_hlo_modules)
mcp.add_tool(hlo_tools.get_hlo_module_content)
mcp.add_tool(hlo_tools.get_hlo_neighborhood)

# ---------------------------------------------------------------------------
# XPlane / timeline tools (require tensorflow + XPROF_LOGDIR)
# ---------------------------------------------------------------------------
mcp.add_tool(xplane_tools.list_xplane_events)
mcp.add_tool(xplane_tools.aggregate_xplane_events)
mcp.add_tool(xplane_tools.get_xspace_proto)

# ---------------------------------------------------------------------------
# XLA HLO dump tools (read from --xla_dump_to directory, no server needed)
# ---------------------------------------------------------------------------
mcp.add_tool(hlo_dump_tools.list_hlo_dump_modules)
mcp.add_tool(hlo_dump_tools.get_hlo_dump)
mcp.add_tool(hlo_dump_tools.diff_hlo_stages)
mcp.add_tool(hlo_dump_tools.get_hlo_dump_neighborhood)


# ---------------------------------------------------------------------------
# Discovery prompt
# ---------------------------------------------------------------------------
@mcp.prompt()
def discovery_flow() -> list[types.PromptMessage]:
    """Recommended analysis workflow for XProf OSS profiles."""
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=(
                    "## XProf MCP Analysis Guide\n\n"
                    "### Investigation Workflow\n\n"
                    "1. **Discover runs**: `list_runs()` — find available sessions.\n"
                    "   Start xprof first: `xprof --logdir=<path> --port=8791`\n\n"
                    "2. **Overview**: `get_overview(run)` — step time, device utilization,\n"
                    "   bottleneck category (compute / memory / host / network).\n\n"
                    "3. **Top ops**: `get_top_hlo_ops(run)` — most expensive HLO ops by\n"
                    "   time, FLOPs, and memory. Falls back to `op_profile` for inference\n"
                    "   runs where `hlo_stats` is empty.\n\n"
                    "4. **Program breakdown**: `get_op_profile(run)` — per-program view\n"
                    "   with idle time and compute breakdown. Useful for inference.\n\n"
                    "5. **HLO deep dive**: `list_hlo_modules(run)` then\n"
                    "   `get_hlo_module_content(run, module_name)` to inspect the\n"
                    "   compiled instruction graph.\n\n"
                    "6. **Root cause**: `get_hlo_neighborhood(run, instruction_name)` —\n"
                    "   BFS of producers/consumers of a slow op (radius=2 default).\n\n"
                    "7. **Memory**: `get_memory_profile(run)` — HBM usage and peak.\n\n"
                    "8. **Timeline** (requires tensorflow + XPROF_LOGDIR):\n"
                    "   `aggregate_xplane_events(run, plane_regex='/device:TPU:0')` —\n"
                    "   kernel statistics. `list_xplane_events(run, event_regex='...')` —\n"
                    "   find specific kernel instances.\n\n"
                    "### XLA HLO Dump Workflow (no xprof server needed)\n\n"
                    "Enable dumps: `XLA_FLAGS='--xla_dump_to=/tmp/hlo "
                    "--xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*'`\n\n"
                    "- `list_hlo_dump_modules(dump_dir)` — stages available\n"
                    "- `get_hlo_dump(module_pattern, stage)` — read HLO at a stage\n"
                    "- `diff_hlo_stages(module_pattern, stage_before, stage_after)` — diff\n"
                    "- `get_hlo_dump_neighborhood(instruction, module_pattern, stage)` — BFS\n\n"
                    "### TPU Performance Gotchas (apply when reading profiles)\n\n"
                    "**Roofline rule (v5e):** a bf16 matmul is compute-bound when per-chip\n"
                    "batch B > 240. Below that threshold the chip waits on HBM bandwidth.\n"
                    "On v6e the threshold is ~570. Multi-chip TP crossover is D > ~8755\n"
                    "(sharded hidden dim, not batch).\n\n"
                    "**Dimension alignment:** MXU tile is 128×128 (v5e) or 256×256 (v6e).\n"
                    "Per-chip dimensions must be multiples of 128 (256 preferred). A\n"
                    "hidden dim of 8512 sharded across 16 chips = 532 per chip → NOT a\n"
                    "multiple → poor MXU utilization (46%). Reshard to 8 chips = 1064\n"
                    "per chip → 62% utilization (+16% throughput).\n\n"
                    "**Wrong dtype:** using fp32 weights for inference is the most common\n"
                    "unforced error. Expected gain from switching to bf16: ~17% device\n"
                    "speedup. Diagnose: HLO graph viewer → matmul weights → look for f32.\n\n"
                    "**Materialized broadcasts:** some broadcast patterns fail to fuse,\n"
                    "landing as full-size HBM temporaries. Diagnosed via large bitcast/\n"
                    "copy ops at the top of HLO Op Stats. Fix: restructure so the\n"
                    "consumer appears inside the same fusion as the broadcast.\n\n"
                    "**Dynamic shapes:** each unique input shape triggers a separate XLA\n"
                    "compilation. For inference, use StaticCache (fixed KV cache shape).\n"
                    "Measured: DynamicCache eager 130.9s → StaticCache + jit 14.8s (8.8×\n"
                    "speedup) on Llama-2-7B / 50 tokens / TPU v6e.\n\n"
                    "**High rematerialization:** check HLO Op Stats for remat time. If HBM\n"
                    "headroom is large (>20%), tune the remat policy to do less recompute.\n"
                    "Selective AC = +2.7% compute, -70% activation memory (preferred over\n"
                    "full AC which costs +30-40% compute).\n\n"
                    "**Tensor parallelism limit:** TP > 8 across ICI island boundary\n"
                    "(DCN) causes ~43% degradation. Keep TP ≤ 8 within a slice.\n\n"
                    "**KV cache in decode:** dynamic-slice / update-slice ops dominating\n"
                    "the timeline means decode is memory-bandwidth-bound (normal for\n"
                    "autoregressive generation with small batch). Fix: increase decode\n"
                    "batch, use splash_attention (keeps attention in VMEM, not HBM).\n\n"
                    "**Deferred execution traps (PyTorch/XLA):** `.item()`, `.cpu()`, or\n"
                    "`print(tensor)` inside a training loop trigger device sync and may\n"
                    "cause recompilation. Debug with `TPU_DEFER_NEVER=1`.\n\n"
                    "### Decision tree\n\n"
                    "High idle (>30%): host-bound (data pipeline) or collective-bound\n"
                    "  (all-reduce). Check get_overview bottleneck_category.\n"
                    "Low FLOP utilization (<40%): dimension misalignment or wrong dtype.\n"
                    "Large bitcast/copy ops: layout mismatch or broadcast materialization.\n"
                    "High remat time: tune rematerialization policy.\n"
                    "dynamic-slice dominates: KV cache / decode bottleneck.\n"
                ),
            ),
        )
    ]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="XProf MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport mode: 'stdio' (managed by Claude Code), 'sse' (SSE HTTP server), or 'http' (streamable-http, recommended for standalone use).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8792,
        help="Port for SSE transport (default: 8792). Ignored in stdio mode.",
    )
    args = parser.parse_args()

    transport = "streamable-http" if args.transport == "http" else args.transport
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
