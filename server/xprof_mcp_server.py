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

Add to Claude Code MCP config (~/.claude/settings.json):
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
"""

import sys

from mcp import types
from mcp.server import fastmcp

from xprof_mcp.internal import hlo_tools
from xprof_mcp.internal import xplane_tools
from xprof_mcp.internal import xprof_data
from xprof_mcp.tools import get_memory_profile_tool
from xprof_mcp.tools import get_overview_tool
from xprof_mcp.tools import get_top_hlo_ops_tool
from xprof_mcp.tools import list_runs_tool

mcp = fastmcp.FastMCP("XProf")

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
                    "To analyze an XProf profile efficiently, follow this workflow:\n\n"
                    "1. **Discover runs**: Call `list_runs()` to see available sessions.\n"
                    "   - Start the xprof server first: `xprof --logdir=<path> --port=8791`\n\n"
                    "2. **Overview**: Call `get_overview(run)` for step time, device\n"
                    "   utilization, and run environment.\n\n"
                    "3. **Top ops**: Call `get_top_hlo_ops(run)` to see the most\n"
                    "   expensive HLO operations by time, FLOPs, and memory.\n\n"
                    "4. **HLO deep dive**: Call `list_hlo_modules(run)` to find compiled\n"
                    "   programs, then `get_hlo_module_content(run, module_name)` to\n"
                    "   inspect the full instruction graph.\n\n"
                    "5. **Root cause**: Call `get_hlo_neighborhood(run, instruction_name)`\n"
                    "   to inspect producers/consumers of a slow op (radius=2 default).\n\n"
                    "6. **Memory**: Call `get_memory_profile(run)` for HBM usage.\n\n"
                    "7. **Timeline** (requires tensorflow + XPROF_LOGDIR):\n"
                    "   - `list_xplane_events(run)` — find specific kernel instances.\n"
                    "   - `aggregate_xplane_events(run)` — statistical breakdown.\n\n"
                    "Profile files are in:\n"
                    "  <logdir>/plugins/profile/<run_name>/<host>.xplane.pb\n"
                ),
            ),
        )
    ]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
