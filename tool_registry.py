"""Single source of truth for the analysis tool surface.

Both frontends consume this registry:
  - the MCP server (server/xprof_mcp_server.py) registers every function
    as an MCP tool;
  - the CLI (cli/main.py) exposes every function as a subcommand.

Adding a tool here makes it available in both. Function docstrings are
load-bearing: they become the MCP tool descriptions and the CLI help text.
"""

from collections import OrderedDict
from typing import Callable

from xprof_mcp.internal import hlo_dump_tools
from xprof_mcp.internal import hlo_tools
from xprof_mcp.internal import kernel_profiling_tools
from xprof_mcp.internal import llo_dump_tools
from xprof_mcp.internal import mosaic_tools
from xprof_mcp.internal import xplane_tools
from xprof_mcp.internal import xprof_data
from xprof_mcp.tools import analysis_tools
from xprof_mcp.tools import detect_tools
from xprof_mcp.tools import get_memory_profile_tool
from xprof_mcp.tools import get_overview_tool
from xprof_mcp.tools import get_top_hlo_ops_tool
from xprof_mcp.tools import list_runs_tool

# Ordered: discovery -> summaries -> HLO -> timeline -> dumps -> kernel/LLO.
ALL_TOOLS: "OrderedDict[str, Callable]" = OrderedDict(
    (fn.__name__, fn)
    for fn in [
        # Discovery / listing
        list_runs_tool.list_runs,
        xprof_data.get_hosts,
        # Performance summaries
        get_overview_tool.get_overview,
        get_memory_profile_tool.get_memory_profile,
        get_top_hlo_ops_tool.get_top_hlo_ops,
        xprof_data.get_op_profile,
        xprof_data.get_profile_summary,
        xprof_data.get_device_information,
        analysis_tools.get_kpi_metrics,
        # Whole-model analyses (roofline, comm, memory, host pipeline)
        analysis_tools.get_roofline_model,
        analysis_tools.get_pod_viewer,
        analysis_tools.get_megascale_stats,
        analysis_tools.get_memory_viewer,
        analysis_tools.get_input_pipeline,
        analysis_tools.get_framework_op_stats,
        analysis_tools.get_utilization_viewer,
        analysis_tools.get_smart_suggestions,
        detect_tools.detect_unfused_reshapes,
        # HLO analysis
        hlo_tools.list_hlo_modules,
        hlo_tools.get_hlo_module_content,
        hlo_tools.get_hlo_neighborhood,
        # XPlane / timeline
        xplane_tools.list_xplane_events,
        xplane_tools.aggregate_xplane_events,
        xplane_tools.get_xspace_proto,
        # XLA HLO dumps (--xla_dump_to)
        hlo_dump_tools.list_hlo_dump_modules,
        hlo_dump_tools.get_hlo_dump,
        hlo_dump_tools.diff_hlo_stages,
        hlo_dump_tools.get_hlo_dump_neighborhood,
        # Kernel profiling / LLO (Pallas-level)
        kernel_profiling_tools.check_kernel_profiling,
        kernel_profiling_tools.list_kernel_invocations,
        kernel_profiling_tools.get_llo_utilization,
        kernel_profiling_tools.get_kernel_stage_breakdown,
        llo_dump_tools.list_llo_programs,
        llo_dump_tools.get_llo_schedule_analysis,
        llo_dump_tools.get_llo_static_utilization,
        llo_dump_tools.get_llo_bundles,
        mosaic_tools.get_custom_call_mlir,
    ]
)

# Tools that operate on dump directories / raw traces rather than a
# server-visible run; the CLI does not cache these by default because their
# inputs are open-ended directory trees.
UNCACHED_TOOLS = frozenset(
    {
        "get_xspace_proto",  # returns a path / huge payloads
    }
)
