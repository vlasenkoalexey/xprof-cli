"""Tool to fetch memory profile data from the OSS XProf server."""

import json
import logging
import traceback

from xprof_mcp.internal import xprof_client


def get_memory_profile(run: str, host: str = "") -> str:
    """Fetches a detailed memory profile analysis from an XProf run.

    Returns peak HBM usage, heap allocations, stack reservations, and overall
    device memory capacity. Use this to understand memory pressure and
    identify allocation hotspots.

    Args:
        run:  The run name (use `list_runs` to discover available runs).
        host: Specific host to query. If empty, queries the first available host.

    Returns:
        A JSON-formatted string with memory profile details.
    """
    run = str(run)
    client = xprof_client.get_client()

    # If host is not specified, pick the first available host
    if not host:
        try:
            hosts = client.get_hosts(run, tool="memory_profile")
            if hosts:
                host = hosts[0].get("hostname", "")
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    try:
        data = client.fetch("memory_profile", run, host=host)
        if not data:
            return json.dumps(
                {"error": "No memory profile data returned", "run": run}, indent=2
            )

        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            return json.dumps(
                {"error": f"Failed to parse memory profile JSON: {e}", "run": run},
                indent=2,
            )

        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list) or not parsed:
            return json.dumps(
                {"error": "Unexpected memory profile format", "run": run}, indent=2
            )

        peak_hbm_bytes = 0
        memory_capacity = 0
        stack_bytes = 0
        heap_bytes = 0
        free_bytes = 0
        fragmentation_raw = 0.0

        # Format 1: [{"memoryProfileSummary": ...}]
        if "memoryProfileSummary" in parsed[0]:
            for device_data in parsed:
                if not device_data or "memoryProfileSummary" not in device_data:
                    continue
                summary = device_data["memoryProfileSummary"]
                mem_cap = int(summary.get("memoryCapacity", 0))
                if mem_cap > memory_capacity:
                    memory_capacity = mem_cap
                if "peakStats" not in summary:
                    continue
                try:
                    current_peak = int(summary["peakStats"].get("peakBytesUsageHbm", 0))
                    if current_peak > peak_hbm_bytes:
                        peak_hbm_bytes = current_peak
                        stack_bytes = int(summary["peakStats"].get("stackReservedBytes", 0))
                        heap_bytes = int(summary["peakStats"].get("heapAllocatedBytes", 0))
                        free_bytes = int(summary["peakStats"].get("freeMemoryBytes", 0))
                        fragmentation_raw = float(summary["peakStats"].get("fragmentation", 0.0))
                except (ValueError, TypeError):
                    continue

        # Format 2: [{"peakMemoryUsageMiB": ...}]
        elif "peakMemoryUsageMiB" in parsed[0]:
            for item in parsed:
                try:
                    current_peak = float(item.get("peakMemoryUsageMiB", 0)) * (1024 ** 2)
                    if current_peak > peak_hbm_bytes:
                        peak_hbm_bytes = int(current_peak)
                except (ValueError, TypeError):
                    continue

        # Format 3: [{"memoryProfilePerAllocator": ...}]
        elif "memoryProfilePerAllocator" in parsed[0]:
            for _, stats in parsed[0]["memoryProfilePerAllocator"].items():
                profile_summary = stats.get("profileSummary", {})
                mem_cap = int(profile_summary.get("memoryCapacity", 0))
                if mem_cap > memory_capacity:
                    memory_capacity = mem_cap
                peak_stats = profile_summary.get("peakStats", {})
                current_peak = int(peak_stats.get("peakBytesInUse", 0))
                if current_peak > peak_hbm_bytes:
                    peak_hbm_bytes = current_peak
                    stack_bytes = int(peak_stats.get("stackReservedBytes", 0))
                    heap_bytes = int(peak_stats.get("heapAllocatedBytes", 0))
                    free_bytes = int(peak_stats.get("freeMemoryBytes", 0))
                    fragmentation_raw = float(peak_stats.get("fragmentation", 0.0))

        else:
            # Return raw data for unrecognised formats
            return json.dumps(
                {"raw_data": parsed[0] if len(parsed) == 1 else parsed, "run": run},
                indent=2,
            )

        capacity_gib = round(memory_capacity / (1024 ** 3), 2) if memory_capacity else 0.0
        peak_gib = round(peak_hbm_bytes / (1024 ** 3), 2) if peak_hbm_bytes else 0.0
        utilization = round((peak_gib / capacity_gib) * 100, 2) if capacity_gib else 0.0

        return json.dumps(
            {
                "run": run,
                "host": host or "(first available)",
                "memory_capacity_gib": capacity_gib,
                "peak_memory_usage_gib": peak_gib,
                "peak_usage_details": {
                    "stack_reservation_gib": round(stack_bytes / (1024 ** 3), 4),
                    "heap_allocation_gib": round(heap_bytes / (1024 ** 3), 4),
                    "free_memory_gib": round(free_bytes / (1024 ** 3), 4),
                    "fragmentation_percent": round(fragmentation_raw * 100, 2),
                    "utilization_percent": utilization,
                },
            },
            indent=2,
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching memory profile for run %s", run)
        return json.dumps(
            {
                "error": f"Error fetching memory profile: {e}",
                "run": run,
                "traceback": traceback.format_exc(),
            },
            indent=2,
        )
