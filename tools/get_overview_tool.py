"""Tool to fetch overview page data from the OSS XProf server."""

import json
import logging
import re
import traceback

from xprof_mcp.internal import xprof_client

# Keys from the overview_page 'p' dict that describe performance.
PERFORMANCE_SUMMARY_KEYS = frozenset({
    "steptime_ms_average",
    "steptime_ms_standard_deviation",
    "tc_idle_ms_average",
    "tc_infeed_ms_average",
    "tc_outfeed_ms_average",
    "host_transfer_ms_average",
    "sc_step_time_ms_average",
    "sc_idle_ms_average",
    "sc_infeed_ms_average",
    "sc_outfeed_ms_average",
    "mxu_utilization_percent",
    "flop_rate_utilization_relative_to_roofline",
    "device_duty_cycle_percent",
    "memory_bw_utilization_relative_to_hw_limit",
    "program_goodput_percent",
    "host_tf_op_percent",
    "device_tf_op_percent",
    "host_op_time_eager_percent",
    "device_op_time_eager_percent",
    "device_idle_time_percent",
    "hbm_bw_utilization_percent",
    "host_idle_time_percent",
})

RUN_ENVIRONMENT_KEYS = frozenset({
    "is_training",
    "profile_start_time",
    "profile_duration_ms",
    "host_count",
    "task_count",
    "device_type",
    "device_core_count",
    "change_list",
    "build_time",
    "build_target",
})


def get_overview(run: str, include_command: bool = False) -> str:
    """Gets a comprehensive overview of an XProf profiling run.

    Returns a unified view of performance summary and run environment,
    extracted from the `overview_page` tool data.

    Args:
        run:             The run name (use `list_runs` to discover available runs).
        include_command: Whether to include the full command line (can be noisy).

    Returns:
        A JSON-formatted string containing 'performance_summary' and
        'run_environment' dicts.
    """
    run = str(run)
    client = xprof_client.get_client()
    try:
        data = client.fetch("overview_page", run, host="ALL_HOSTS")
        if not data:
            return json.dumps(
                {"error": "No overview data returned", "run": run}, indent=2
            )

        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")

        overview_data = json.loads(data)
        if not isinstance(overview_data, list) or not overview_data:
            return json.dumps(
                {"error": "Unexpected overview_page format", "run": run}, indent=2
            )

        # Merge all 'p' dicts from all sections
        all_p: dict = {}
        for section in overview_data:
            all_p.update(section.get("p", {}))

        # Extract hostname / bns / command from the run environment table
        hostname, bns, command = None, None, None
        for section in overview_data:
            if "cols" in section and "rows" in section:
                cols = section.get("cols", [])
                rows = section.get("rows", [])
                col_map = {col.get("id"): i for i, col in enumerate(cols)}
                if "host_id" in col_map and rows and rows[0].get("c"):
                    vals = [c.get("v") for c in rows[0]["c"]]
                    if (idx := col_map.get("host_id")) is not None:
                        hostname = vals[idx]
                    if (idx := col_map.get("bns_address")) is not None:
                        bns = vals[idx]
                    if (idx := col_map.get("command_line")) is not None:
                        command = vals[idx]
                    break

        # Performance summary
        perf: dict = {}
        for key, val in all_p.items():
            if key.startswith("stat_") or key.startswith("sc_") or key in PERFORMANCE_SUMMARY_KEYS:
                perf[key] = val

        # Run environment
        env: dict = {}
        for key, val in all_p.items():
            if key.startswith("run_") or key in RUN_ENVIRONMENT_KEYS:
                if key == "run_command" and not include_command:
                    continue
                env[key] = val

        if hostname:
            env["hostname"] = hostname
        if bns:
            env["bns"] = bns
        if "device_type" in all_p:
            env["device_type"] = all_p["device_type"]
        if include_command and command and "run_command" not in env:
            env["run_command"] = command

        return json.dumps(
            {"run": run, "performance_summary": perf, "run_environment": env},
            indent=2,
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching overview for run %s", run)
        return json.dumps(
            {
                "error": f"Error fetching overview: {e}",
                "run": run,
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )
