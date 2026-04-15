"""Tool to fetch top HLO operations from the OSS XProf server.

Uses the `hlo_stats` JSON endpoint — no protobuf library required.
The `hlo_stats` tool returns per-op timing, flops, and memory access
statistics in a Google DataTable JSON format.
"""

import heapq
import json
import logging
import traceback
from typing import Any

from xprof_mcp.internal import xprof_client
from xprof_mcp.internal.xprof_data import _parse_datatable, _aggregate_ops_from_profile


def get_top_hlo_ops(run: str, *, limit: int = 10) -> str:
    """Fetches the top HLO operations sorted by time, FLOPs, and bytes accessed.

    Uses the `hlo_stats` JSON endpoint for OSS compatibility. Returns three
    ranked lists so you can immediately see where time, compute, and memory
    bandwidth are being spent.

    Args:
        run:   The run name (use `list_runs` to discover available runs).
        limit: Number of top operations per list (default 10).

    Returns:
        A JSON-formatted string with three lists:
          - top_by_time:           ops sorted by total self-time (ms)
          - top_by_flops:          ops sorted by FLOPs
          - top_by_bytes_accessed: ops sorted by memory bytes accessed
    """
    client = xprof_client.get_client()
    try:
        data = client.fetch("hlo_stats", run, host="ALL_HOSTS")
        rows = _parse_datatable(data)

        if not rows:
            # hlo_stats is empty (common for inference runs) — fall back to op_profile
            logging.info("hlo_stats empty for run %s, falling back to op_profile", run)
            hosts = client.get_hosts(run)
            host = hosts[0]["hostname"] if hosts else "ALL_HOSTS"
            raw = client.fetch("op_profile", run, host=host)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            profile = json.loads(raw)
            ops = _aggregate_ops_from_profile(profile)
            top_by_time = heapq.nlargest(limit, ops, key=lambda x: x["total_time_ms"])
            top_by_flops = heapq.nlargest(limit, ops, key=lambda x: x["flops_tf"])
            top_by_bytes = heapq.nlargest(limit, ops, key=lambda x: x["bytes_gib"])
            return json.dumps(
                {
                    "run": run,
                    "source": "op_profile (hlo_stats unavailable)",
                    "top_by_time": top_by_time,
                    "top_by_flops": top_by_flops,
                    "top_by_bytes_accessed": top_by_bytes,
                },
                indent=2,
            )

        # Discover column names dynamically
        sample = rows[0]
        time_col = next(
            (k for k in sample if "self_time" in k.lower() or ("total_time" in k.lower() and "us" in k.lower())),
            next((k for k in sample if "time" in k.lower()), None),
        )
        flops_col = next(
            (k for k in sample if "flop" in k.lower()), None
        )
        bytes_col = next(
            (k for k in sample if "byte" in k.lower() or "memory" in k.lower()), None
        )
        name_col = next(
            (k for k in sample if k in ("hlo_expression", "name", "op_name", "expression")),
            next(iter(sample), None),
        )
        module_col = next(
            (k for k in sample if "module" in k.lower()), None
        )
        occ_col = next(
            (k for k in sample if "occurrence" in k.lower() or "count" in k.lower()), None
        )

        def safe_float(row: dict, col: str | None) -> float:
            if col is None:
                return 0.0
            try:
                return float(row.get(col) or 0)
            except (ValueError, TypeError):
                return 0.0

        def make_entry(row: dict) -> dict[str, Any]:
            entry: dict[str, Any] = {
                "name": str(row.get(name_col) or "?"),
            }
            if module_col:
                entry["module"] = row.get(module_col)
            if occ_col:
                entry["occurrences"] = row.get(occ_col)
            if time_col:
                t_us = safe_float(row, time_col)
                entry["total_self_time_ms"] = round(t_us / 1000.0, 4)
                entry["_sort_time"] = t_us
            if flops_col:
                entry["flops"] = safe_float(row, flops_col)
            if bytes_col:
                entry["bytes_accessed"] = safe_float(row, bytes_col)
            return entry

        all_entries = [make_entry(r) for r in rows]

        top_by_time = heapq.nlargest(limit, all_entries, key=lambda x: x.get("_sort_time", 0))
        top_by_flops = heapq.nlargest(limit, all_entries, key=lambda x: x.get("flops", 0))
        top_by_bytes = heapq.nlargest(limit, all_entries, key=lambda x: x.get("bytes_accessed", 0))

        # Remove internal sort key
        for lst in (top_by_time, top_by_flops, top_by_bytes):
            for e in lst:
                e.pop("_sort_time", None)

        return json.dumps(
            {
                "run": run,
                "top_by_time": top_by_time,
                "top_by_flops": top_by_flops,
                "top_by_bytes_accessed": top_by_bytes,
            },
            indent=2,
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching top HLO ops for run %s", run)
        return json.dumps(
            {
                "error": f"Error fetching top HLO ops: {e}",
                "run": run,
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )
