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


def _parse_op_profile_tree(
    node: dict,
    total_time_ps: float,
    top_n: int,
) -> dict:
    """Recursively converts an op_profile node into a summary dict."""
    m = node.get("metrics", {})
    raw_time = float(m.get("rawTime") or 0)
    raw_flops = float(m.get("rawFlops") or 0)
    raw_bytes = float((m.get("rawBytesAccessedArray") or [0])[0])
    occurrences = int(m.get("occurrences") or 0)
    bw_utils = m.get("bandwidthUtils") or []
    hbm_bw_util = float(bw_utils[0]) if bw_utils else 0.0

    result: dict[str, Any] = {
        "name": node.get("name", "?"),
        "time_ms": round(raw_time / 1e9, 3),
        "time_percent": round(raw_time / total_time_ps * 100, 1) if total_time_ps else 0.0,
        "flops_tf": round(raw_flops / 1e12, 3),
        "bytes_gib": round(raw_bytes / (1024 ** 3), 3),
        "hbm_bw_utilization": round(hbm_bw_util, 4),
    }
    if occurrences:
        result["occurrences"] = occurrences

    children = node.get("children", [])
    if children:
        parsed = [_parse_op_profile_tree(c, total_time_ps, top_n) for c in children]
        parsed.sort(key=lambda x: x["time_ms"], reverse=True)
        result["top_ops"] = parsed[:top_n]

    return result


def get_op_profile(run: str, host: str = "", top_n: int = 10) -> str:
    """Fetches a breakdown of device time by program and operation type.

    This is the primary tool for understanding where device time is spent.
    It uses the `op_profile` endpoint which provides a hierarchical view:
      - Total time vs idle time
      - Top N programs by time (each compiled XLA program)
      - Top ops within each program (convolutions, all-reduce, attention, etc.)

    Use this when `get_top_hlo_ops` returns no data (common for inference
    runs where `hlo_stats` is unavailable).

    Args:
        run:   The run name (use `list_runs` to discover available runs).
        host:  Host to query. Defaults to the first available host.
        top_n: Number of top programs to show by time (default 10).
               Remaining programs are aggregated into 'other_programs'.
               Also controls max ops shown per program.

    Returns:
        A JSON-formatted dict with total time, idle time, top programs,
        and an aggregate of the remaining programs.
    """
    client = xprof_client.get_client()
    try:
        if not host:
            hosts = client.get_hosts(run)
            host = hosts[0]["hostname"] if hosts else "ALL_HOSTS"

        data = client.fetch("op_profile", run, host=host)
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        profile = json.loads(data)

        by_program = profile.get("byProgram", {})
        m = by_program.get("metrics", {})
        total_time_ps = float(m.get("rawTime") or 0)
        total_time_ms = total_time_ps / 1e9

        all_programs = []
        idle_ms = 0.0
        ops_per_program = max(3, top_n // 2)
        for child in by_program.get("children", []):
            parsed = _parse_op_profile_tree(child, total_time_ps, ops_per_program)
            if child.get("name") == "IDLE":
                idle_ms = parsed["time_ms"]
            else:
                all_programs.append(parsed)

        all_programs.sort(key=lambda x: x["time_ms"], reverse=True)

        shown = all_programs[:top_n]
        rest = all_programs[top_n:]
        other: dict[str, Any] = {}
        if rest:
            other = {
                "program_count": len(rest),
                "total_time_ms": round(sum(p["time_ms"] for p in rest), 3),
                "time_percent": round(
                    sum(p["time_ms"] for p in rest) / total_time_ms * 100, 1
                ) if total_time_ms else 0.0,
            }

        result: dict[str, Any] = {
            "run": run,
            "host": host,
            "total_time_ms": round(total_time_ms, 3),
            "idle_time_ms": round(idle_ms, 3),
            "idle_percent": round(idle_ms / total_time_ms * 100, 1) if total_time_ms else 0.0,
            "compute_time_ms": round(total_time_ms - idle_ms, 3),
            "total_program_count": len(all_programs),
            "top_programs": shown,
        }
        if other:
            result["other_programs"] = other

        return json.dumps(result, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching op profile for run %s", run)
        return json.dumps({"error": str(e), "run": run}, indent=2)


def _aggregate_ops_from_profile(profile: dict) -> list[dict[str, Any]]:
    """Flattens op_profile tree, aggregating ops by name across all programs."""
    by_program = profile.get("byProgram", {})
    totals: dict[str, dict[str, Any]] = {}

    for program in by_program.get("children", []):
        if program.get("name") == "IDLE":
            continue
        for op in program.get("children", []):
            name = op.get("name", "?")
            m = op.get("metrics", {})
            t = float(m.get("rawTime") or 0)
            f = float(m.get("rawFlops") or 0)
            b = float((m.get("rawBytesAccessedArray") or [0])[0])
            occ = int(m.get("occurrences") or 0)
            if name not in totals:
                totals[name] = {"name": name, "time_ps": 0.0, "flops": 0.0, "bytes": 0.0, "occurrences": 0}
            totals[name]["time_ps"] += t
            totals[name]["flops"] += f
            totals[name]["bytes"] += b
            totals[name]["occurrences"] += occ

    total_time_ps = float((by_program.get("metrics") or {}).get("rawTime") or 0)
    result = []
    for entry in totals.values():
        t_ms = entry["time_ps"] / 1e9
        result.append({
            "name": entry["name"],
            "total_time_ms": round(t_ms, 3),
            "time_percent": round(t_ms / total_time_ps * 1e9 * 100, 1) if total_time_ps else 0.0,
            "flops_tf": round(entry["flops"] / 1e12, 3),
            "bytes_gib": round(entry["bytes"] / (1024 ** 3), 3),
            "occurrences": entry["occurrences"],
        })
    return result


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
