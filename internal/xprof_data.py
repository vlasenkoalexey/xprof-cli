"""High-level data fetching tools for OSS XProf MCP.

These tools use the xprof HTTP server's JSON endpoints only —
no protobuf libraries required.
"""

import json
import logging
from typing import Any

from xprof_mcp.internal import xprof_client


# ---------------------------------------------------------------------------
# Helpers for Google DataTable JSON format
# ---------------------------------------------------------------------------

def _parse_datatable(data: bytes | str) -> list[dict[str, Any]]:
    """Parses a Google DataTable JSON response into a list of row dicts.

    The DataTable format is a list of table objects:
      [{"cols": [{"id": "col1", ...}], "rows": [{"c": [{"v": val}, ...]}, ...]}]

    Returns a flat list of row dicts keyed by column id.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    parsed = json.loads(data)
    if not isinstance(parsed, list):
        return []

    rows_out = []
    for table in parsed:
        cols = table.get("cols", [])
        col_ids = [c.get("id") or c.get("label", f"col{i}") for i, c in enumerate(cols)]
        for row in table.get("rows", []):
            cells = row.get("c", [])
            row_dict = {}
            for col_id, cell in zip(col_ids, cells):
                row_dict[col_id] = cell.get("v") if cell else None
            rows_out.append(row_dict)
    return rows_out


def _extract_p_dict(data: bytes | str) -> dict[str, Any]:
    """Extracts the merged 'p' (properties) dict from all DataTable sections."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    parsed = json.loads(data)
    if not isinstance(parsed, list):
        return {}
    result: dict[str, Any] = {}
    for section in parsed:
        result.update(section.get("p", {}))
    return result


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_profile_summary(run: str) -> str:
    """Provides a high-level performance summary of a profiling run.

    **START HERE.** Identifies the top HLO ops by self-time, step time, and
    device utilization. Use the output to decide which ops need a deeper dive.

    Args:
        run: The run name (session directory name, e.g. 'my_experiment_run_1').
             List available runs with `list_runs`.

    Returns:
        A text summary of the profile's performance landscape.
    """
    client = xprof_client.get_client()
    try:
        # Use hlo_stats which returns JSON DataTable — no proto deps needed.
        data = client.fetch("hlo_stats", run, host="ALL_HOSTS")
        rows = _parse_datatable(data)

        if not rows:
            # Fallback: try overview_page for a minimal summary
            ov_data = client.fetch("overview_page", run, host="ALL_HOSTS")
            p = _extract_p_dict(ov_data)
            lines = [f"Profile Summary for: {run}"]
            for key in ("steptime_ms_average", "device_duty_cycle_percent",
                        "mxu_utilization_percent", "device_type"):
                if key in p:
                    lines.append(f"  {key}: {p[key]}")
            return "\n".join(lines) if len(lines) > 1 else f"No data found for run: {run}"

        # Sort by total_self_time_in_us descending
        time_col = next(
            (k for k in rows[0] if "self_time" in k or "total_time" in k), None
        )
        if time_col:
            rows.sort(key=lambda r: float(r.get(time_col) or 0), reverse=True)

        lines = [f"Profile Summary for: {run}", "", "Top HLO ops by self-time:"]
        lines.append(f"{'Op':<60} {'Self Time':>14} {'Occurrences':>12}")
        lines.append("-" * 90)

        total_time = sum(float(r.get(time_col) or 0) for r in rows) if time_col else 0

        for r in rows[:15]:
            name = str(r.get("hlo_expression") or r.get("name") or "?")
            name = name[:58]
            t = float(r.get(time_col) or 0)
            occ = r.get("occurrences") or ""
            frac = f"({t / total_time:.1%})" if total_time else ""
            t_str = f"{t/1000:.2f} ms {frac}" if t > 0 else "?"
            lines.append(f"{name:<60} {t_str:>14} {str(occ):>12}")

        return "\n".join(lines)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching profile summary for run %s", run)
        return f"Error fetching profile summary for run '{run}': {e}"


def get_hlo_op_profile(run: str, top_n: int = 15) -> str:
    """Summarizes the most expensive HLO operations in the run.

    Uses the `hlo_stats` JSON endpoint (no protobuf dependency).

    Args:
        run:   The run name.
        top_n: Number of top operations to return (default 15).

    Returns:
        A JSON-formatted list of the top operations with timing metrics.
    """
    client = xprof_client.get_client()
    try:
        data = client.fetch("hlo_stats", run, host="ALL_HOSTS")
        rows = _parse_datatable(data)
        if not rows:
            return json.dumps({"error": "No HLO stats data found", "run": run}, indent=2)

        time_col = next(
            (k for k in rows[0] if "self_time" in k or "total_time" in k), None
        )
        if time_col:
            rows.sort(key=lambda r: float(r.get(time_col) or 0), reverse=True)

        top = rows[:top_n]
        # Normalize column names for output
        result = []
        for r in top:
            entry = dict(r)
            if time_col and r.get(time_col):
                entry["total_self_time_ms"] = float(r[time_col]) / 1000.0
            result.append(entry)

        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching HLO op profile for run %s", run)
        return f"Error fetching HLO op profile: {e}"


def get_hosts(run: str) -> str:
    """Returns the list of hosts profiled in the run.

    Args:
        run: The run name.

    Returns:
        A JSON-formatted dict with the list of hosts.
    """
    client = xprof_client.get_client()
    try:
        hosts = client.get_hosts(run)
        return json.dumps({"run": run, "hosts": hosts}, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching hosts for run %s", run)
        return json.dumps({"error": f"Error fetching hosts: {e}"}, indent=2)


def get_device_information(run: str) -> str:
    """Returns hardware device information from the Roofline Model analysis.

    Retrieves device specs: accelerator type, peak FLOP rate, peak memory
    bandwidths, and ridge points.

    Args:
        run: The run name.

    Returns:
        A JSON-formatted dict of device information.
    """
    client = xprof_client.get_client()
    try:
        data = client.fetch("roofline_model", run, host="ALL_HOSTS")
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        parsed = json.loads(data)
        if not isinstance(parsed, list) or not parsed:
            return json.dumps({"error": "Unexpected roofline model format"}, indent=2)

        table_props = parsed[0].get("p", {})
        device_info: dict = {}
        for key, value in table_props.items():
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
            device_info[key] = value

        return json.dumps(device_info, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching device information for run %s", run)
        return json.dumps({"error": f"Error fetching device information: {e}"}, indent=2)
