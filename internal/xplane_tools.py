"""XPlane/XSpace tools for OSS XProf MCP.

These tools read raw `.xplane.pb` files directly from the xprof logdir
and require the `tensorflow` (or `tensorflow-cpu`) package to parse the
XSpace/XPlane protobuf format.

Set the `XPROF_LOGDIR` environment variable to the directory passed to
`xprof --logdir=...` before starting the MCP server.

Typical logdir structure:
  <logdir>/
    plugins/
      profile/
        <run_name>/
          <host>.xplane.pb
          <host>.xplane.pb
          ...
"""

import collections
import json
import logging
import re
import statistics
from typing import Optional

from xprof_mcp.internal import xprof_client

# ---------------------------------------------------------------------------
# Optional import of XPlane proto
# ---------------------------------------------------------------------------
try:
    # TF 2.x path (TF < 2.21)
    from tensorflow.python.profiler.trace import xplane_pb2  # type: ignore
    _HAS_XPLANE_PROTO = True
except ImportError:
    try:
        # TF 2.21+ path (moved to tsl)
        from tensorflow.tsl.profiler.protobuf import xplane_pb2  # type: ignore
        _HAS_XPLANE_PROTO = True
    except ImportError:
        try:
            from tensorboard_plugin_profile.protobuf import xplane_pb2  # type: ignore
            _HAS_XPLANE_PROTO = True
        except ImportError:
            _HAS_XPLANE_PROTO = False
            xplane_pb2 = None  # type: ignore

_XPLANE_IMPORT_ERROR = (
    "XPlane tools require the `tensorflow` or `tensorflow-cpu` package. "
    "Install it with: pip install tensorflow-cpu\n"
    "Alternatively, use `get_overview`, `get_top_hlo_ops`, or "
    "`get_hlo_module_content` which do not require tensorflow."
)


def _require_xplane_proto() -> None:
    if not _HAS_XPLANE_PROTO:
        raise ImportError(_XPLANE_IMPORT_ERROR)


def _fetch_xspace(run: str, host: str = "") -> "xplane_pb2.XSpace":
    """Reads and parses an XSpace proto from disk."""
    _require_xplane_proto()
    client = xprof_client.get_client()

    if not host:
        # Pick first available host
        hosts = client.list_xplane_hosts(run)
        if not hosts:
            raise FileNotFoundError(
                f"No .xplane.pb files found for run '{run}'. "
                "Check that XPROF_LOGDIR is set correctly."
            )
        host = hosts[0]

    raw = client.read_xplane_bytes(run, host)
    xspace = xplane_pb2.XSpace()
    xspace.ParseFromString(raw)
    return xspace


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def get_xspace_proto(
    run: str,
    host: str = "",
    as_text: bool = False,
    output_path: Optional[str] = None,
) -> str | bytes:
    """Returns the XSpace proto for a host in the run.

    Reads the raw `.xplane.pb` file from disk (requires tensorflow and
    XPROF_LOGDIR to be set).

    Args:
        run:         The run name.
        host:        Host name (e.g. 'host-0'). Defaults to the first host.
        as_text:     If True, returns the proto as a human-readable text string.
                     If False, returns serialized bytes.
        output_path: If provided, writes content to this path and returns the path.

    Returns:
        Proto content (str or bytes) or the output_path if provided.
    """
    try:
        _require_xplane_proto()
        client = xprof_client.get_client()
        if not host:
            hosts = client.list_xplane_hosts(run)
            if not hosts:
                return f"No .xplane.pb files found for run '{run}'."
            host = hosts[0]

        raw = client.read_xplane_bytes(run, host)

        if as_text:
            from google.protobuf import text_format  # type: ignore
            xspace = xplane_pb2.XSpace()
            xspace.ParseFromString(raw)
            content: str | bytes = text_format.MessageToString(xspace)
            mode = "w"
        else:
            content = raw
            mode = "wb"

        if output_path:
            with open(output_path, mode) as f:
                f.write(content)
            return output_path

        return content

    except ImportError:
        return _XPLANE_IMPORT_ERROR
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching XSpace proto for run %s host %s", run, host)
        return f"Error fetching XSpace proto: {e}"


def list_xplane_events(
    run: str,
    host: str = "",
    plane_regex: str = ".*",
    event_regex: str = ".*",
    start_time_ps: Optional[int] = None,
    end_time_ps: Optional[int] = None,
    max_events: int = 100,
    offset: int = 0,
) -> str:
    """Searches and filters timeline events across XPlanes.

    **Use this** to find specific instances of slow kernels or timeline gaps.
    Supports regex filtering on both the plane name (host/device) and the
    event name.

    Requires `tensorflow` and `XPROF_LOGDIR` to be set.

    Examples:
      - `plane_regex='Device.*'`, `event_regex='Fusion.*'`: device-side fusions.
      - `plane_regex='host.*'`, `event_regex='.*Wait.*'`: host sync waits.

    Args:
        run:           The run name.
        host:          Host name. Defaults to the first host found.
        plane_regex:   Regex to filter XPlanes (e.g. 'Device.*').
        event_regex:   Regex to filter event names (e.g. 'Fusion.*').
        start_time_ps: Filter by start time in picoseconds.
        end_time_ps:   Filter by end time in picoseconds.
        max_events:    Maximum number of events to return (default 100).
        offset:        Skip this many matching events before returning.

    Returns:
        A JSON-formatted list of matching timeline events.
    """
    try:
        _require_xplane_proto()
        xspace = _fetch_xspace(run, host)

        events = []
        skipped = 0
        p_re = re.compile(plane_regex)
        e_re = re.compile(event_regex)

        for plane in xspace.planes:
            if not p_re.search(plane.name):
                continue

            metadata_map: dict[int, str] = {}
            for k, v in plane.event_metadata.items():
                metadata_map[k] = v.display_name or v.name or f"ID:{v.id}"

            stat_metadata_map: dict[int, str] = {}
            for k, v in plane.stat_metadata.items():
                stat_metadata_map[k] = v.name

            for line in plane.lines:
                for event in line.events:
                    if start_time_ps is not None and event.offset_ps < start_time_ps:
                        continue
                    if end_time_ps is not None:
                        if event.offset_ps + event.duration_ps > end_time_ps:
                            continue

                    event_name = metadata_map.get(event.metadata_id, "Unknown")
                    if event_name.isdigit():
                        for stat in event.stats:
                            sname = stat_metadata_map.get(stat.metadata_id, "")
                            if sname in ("msg", "message", "annotation", "label"):
                                val = stat.str_value or str(stat.double_value) or str(stat.int64_value)
                                if val:
                                    event_name = val
                                    break

                    if not e_re.search(event_name):
                        continue
                    if skipped < offset:
                        skipped += 1
                        continue

                    events.append({
                        "plane": plane.name,
                        "line_id": line.id,
                        "event": event_name,
                        "offset_ps": event.offset_ps,
                        "duration_ps": event.duration_ps,
                    })
                    if len(events) >= max_events:
                        break
                if len(events) >= max_events:
                    break
            if len(events) >= max_events:
                break

        return json.dumps(events, indent=2)

    except ImportError:
        return _XPLANE_IMPORT_ERROR
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error listing XPlane events for run %s", run)
        return f"Error listing XPlane events: {e}"


def aggregate_xplane_events(
    run: str,
    host: str = "",
    plane_regex: str = ".*",
    event_regex: str = ".*",
    top_n: int = 50,
) -> str:
    """Calculates statistical aggregates for matching timeline events.

    **Systemic analysis.** Use this to determine if a kernel type is
    consistently slow or has high variance. Returns count, total, average,
    min, max, and standard deviation of durations (in picoseconds).

    Requires `tensorflow` and `XPROF_LOGDIR` to be set.

    Args:
        run:         The run name.
        host:        Host name. Defaults to the first host found.
        plane_regex: Regex to filter XPlanes.
        event_regex: Regex to filter event names.
        top_n:       Return only the top N event types by total duration
                     (default 50). Use a specific event_regex to narrow results.

    Returns:
        A JSON string with statistical aggregates grouped by event name.
    """
    try:
        _require_xplane_proto()
        xspace = _fetch_xspace(run, host)

        p_re = re.compile(plane_regex)
        e_re = re.compile(event_regex)

        stats_data: dict[str, list[int]] = collections.defaultdict(list)
        total_scanned = 0
        MAX_SCAN = 500_000

        for plane in xspace.planes:
            if not p_re.search(plane.name):
                continue

            metadata_map: dict[int, str] = {}
            for k, v in plane.event_metadata.items():
                metadata_map[k] = v.display_name or v.name or f"ID:{v.id}"

            stat_metadata_map: dict[int, str] = {}
            for k, v in plane.stat_metadata.items():
                stat_metadata_map[k] = v.name

            for line in plane.lines:
                for event in line.events:
                    name = metadata_map.get(event.metadata_id, "Unknown")
                    if name.isdigit():
                        for stat in event.stats:
                            sname = stat_metadata_map.get(stat.metadata_id, "")
                            if sname in ("msg", "message", "annotation", "label"):
                                val = stat.str_value or str(stat.double_value) or str(stat.int64_value)
                                if val:
                                    name = val
                                    break

                    if e_re.search(name):
                        stats_data[name].append(event.duration_ps)

                    total_scanned += 1
                    if total_scanned >= MAX_SCAN:
                        break
                if total_scanned >= MAX_SCAN:
                    break
            if total_scanned >= MAX_SCAN:
                break

        results = []
        for name, durations in stats_data.items():
            count = len(durations)
            total = sum(durations)
            results.append({
                "event": name,
                "count": count,
                "total_duration_ps": total,
                "avg_duration_ps": total / count,
                "min_duration_ps": min(durations),
                "max_duration_ps": max(durations),
                "std_dev_ps": statistics.stdev(durations) if count > 1 else 0.0,
            })

        results.sort(key=lambda x: x["total_duration_ps"], reverse=True)
        truncated = len(results) > top_n
        out = {
            "total_event_types": len(results),
            "shown": min(top_n, len(results)),
            "events": results[:top_n],
        }
        if truncated:
            out["note"] = f"Truncated to top {top_n} by total duration. Use event_regex to filter or increase top_n."
        return json.dumps(out, indent=2)

    except ImportError:
        return _XPLANE_IMPORT_ERROR
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error aggregating XPlane events for run %s", run)
        return f"Error aggregating XPlane events: {e}"
