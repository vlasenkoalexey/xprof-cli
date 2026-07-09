"""Kernel-profiling (LLO-level) trace tools for OSS XProf MCP.

These tools read the kernel-profiling lines that appear on the device
XPlanes when a workload is captured with the XProf Kernel Profiling flags:

    LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true \\
                      --xla_xprof_register_llo_debug_info=true"

With the flags active, each `/device:TPU:N` plane gains three line families
(validated on v6e, 2026-07-09):

  - `_counters_`    LLO slot-utilization time-series: events named `MXU`,
                    `Scalar ALU`, `Vector ALU`, `Vector Load`, `Vector Store`,
                    `XLU`, `Vector EUP` carry a `% util` stat per ~1us window;
                    `Vector Fills` / `Vector Spills` carry `fills` / `spills`
                    counts.
  - `Tensor Core`   `bundle.<instrumentation_id>` markers whose `long_name`
                    stat carries the LLO bundle address (e.g. `0x1ee`) — the
                    join key into `--xla_jf_dump_to` LLO dumps.
  - `XLA TraceMe`   Mosaic pipeline stage scopes (`ep_wait_in`, `ep_copy_in`,
                    `ep_run_kernel`, user `jax.named_scope`s, `ep_copy_out`,
                    `ep_wait_out`, ...).

NOTE on semantics: the `% util` values are LLO *static slot occupancy* laid
over measured time windows (static content, measured alignment) — not raw
hardware counters. Raw runtime counter sampling is TPU v7 (Ironwood)+ only
and silently produces no tracks on earlier generations.

Requires `tensorflow` (or `tensorflow-cpu`) and `XPROF_LOGDIR`, same as the
other xplane tools.
"""

import collections
import json
import logging
import re
import statistics
from typing import Optional

from xprof_mcp.internal import xplane_tools

# Line names produced by the kernel-profiling flags.
_COUNTERS_LINE = "_counters_"
_TENSOR_CORE_LINE = "Tensor Core"
_TRACEME_LINE = "XLA TraceMe"
_XLA_OPS_LINE = "XLA Ops"

_DEVICE_PLANE_RE = re.compile(r"/device:(TPU|GPU):\d+")

# Mosaic emit_pipeline stage scopes (fixed vocabulary; anything else on the
# TraceMe line inside a kernel span is treated as a user named_scope).
_EP_STAGES = {
    "ep_initialize_0", "ep_initialize_1", "ep_finalize",
    "ep_wait_in", "ep_copy_in", "ep_run_kernel",
    "ep_wait_out", "ep_copy_out",
}

_CAPTURE_HINT = (
    "Capture with LIBTPU_INIT_ARGS=\"--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true\" and jax.profiler.trace(...)."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _stat_map(plane) -> dict[int, str]:
    return {k: v.name for k, v in plane.stat_metadata.items()}


def _event_name_map(plane) -> dict[int, str]:
    return {
        k: (v.display_name or v.name or f"ID:{v.id}")
        for k, v in plane.event_metadata.items()
    }


def _stat_value(stat):
    """Extracts the populated value from an XStat."""
    for field in ("double_value", "int64_value", "uint64_value", "str_value"):
        v = getattr(stat, field, None)
        if v:
            return v
    return 0


def _event_stats(event, stat_names: dict[int, str]) -> dict:
    return {
        stat_names.get(s.metadata_id, str(s.metadata_id)): _stat_value(s)
        for s in event.stats
    }


def _abs_span_ps(line, event) -> tuple[int, int]:
    """Absolute (start_ps, end_ps) for an event (line timestamps are ns)."""
    start = line.timestamp_ns * 1000 + event.offset_ps
    return start, start + event.duration_ps


def _device_planes(xspace, device_regex: str = ""):
    d_re = re.compile(device_regex) if device_regex else _DEVICE_PLANE_RE
    return [p for p in xspace.planes if d_re.search(p.name)]


def _find_line(plane, name: str):
    for line in plane.lines:
        if line.name == name:
            return line
    return None


def _instrumented_windows(plane) -> list[tuple[int, int]]:
    """Spans of LLO instrumentation events (Tensor Core markers + TraceMe)."""
    windows: list[tuple[int, int]] = []
    for lname in (_TENSOR_CORE_LINE, _TRACEME_LINE):
        line = _find_line(plane, lname)
        if line is None:
            continue
        for event in line.events:
            windows.append(_abs_span_ps(line, event))
    return windows


def _kernel_invocation_spans(
    plane, kernel_regex: str = ".*"
) -> dict[str, list[tuple[int, int]]]:
    """Kernel (custom-call) invocation spans from the `XLA Ops` line.

    The raw XPlane op name carries no `custom-call` text, so an XLA Ops
    event counts as a kernel invocation iff LLO instrumentation events
    (Tensor Core bundle markers / XLA TraceMe stage scopes — only emitted
    inside custom-call regions when the kernel-profiling flags are on)
    overlap its span. Keyed by the HLO op name (e.g. `matmul_optimized.1`).
    """
    line = _find_line(plane, _XLA_OPS_LINE)
    if line is None:
        return {}
    instrumented = _instrumented_windows(plane)
    if not instrumented:
        return {}
    names = _event_name_map(plane)
    k_re = re.compile(kernel_regex)

    spans: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for event in line.events:
        op_name = names.get(event.metadata_id, "")
        if not k_re.search(op_name):
            continue
        span = _abs_span_ps(line, event)
        if _overlaps(span, instrumented):
            spans[op_name].append(span)
    return spans


def _overlaps(span: tuple[int, int], windows: list[tuple[int, int]]) -> bool:
    s, e = span
    return any(s < we and e > ws for ws, we in windows)


def _duration_stats(durations_ps: list[int]) -> dict:
    if not durations_ps:
        return {}
    return {
        "count": len(durations_ps),
        "total_us": round(sum(durations_ps) / 1e6, 3),
        "mean_us": round(statistics.mean(durations_ps) / 1e6, 3),
        "p50_us": round(statistics.median(durations_ps) / 1e6, 3),
        "min_us": round(min(durations_ps) / 1e6, 3),
        "max_us": round(max(durations_ps) / 1e6, 3),
    }


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def check_kernel_profiling(run: str, host: str = "") -> str:
    """Checks whether a trace was captured with the kernel-profiling flags.

    **Call this first** before any LLO-level analysis. Reports, per device
    plane, which kernel-profiling lines are present (`_counters_`,
    `Tensor Core`, `XLA TraceMe`) and therefore whether
    `--xla_enable_custom_call_region_trace` / `--xla_xprof_register_llo_debug_info`
    were active at capture time. Analyzing a flagless trace at the LLO level
    is the canonical silent-noop mistake.

    Args:
        run:  The run name.
        host: Host name. Defaults to the first host found.

    Returns:
        JSON: per-device line presence, counter units seen, TraceMe stages
        seen, kernel (custom-call) count, and a capture hint when inactive.
    """
    try:
        xspace = xplane_tools._fetch_xspace(run, host)  # pylint: disable=protected-access
        result: dict = {"devices": {}, "kernel_profiling_active": False}
        for plane in _device_planes(xspace):
            names = _event_name_map(plane)
            info: dict = {"lines": {}}
            for lname in (_COUNTERS_LINE, _TENSOR_CORE_LINE, _TRACEME_LINE):
                line = _find_line(plane, lname)
                info["lines"][lname] = len(line.events) if line else 0
            cline = _find_line(plane, _COUNTERS_LINE)
            if cline:
                info["counter_units"] = sorted(
                    {names.get(e.metadata_id, "?") for e in cline.events})
            tline = _find_line(plane, _TRACEME_LINE)
            if tline:
                info["traceme_stages"] = sorted(
                    {names.get(e.metadata_id, "?") for e in tline.events})
            info["kernel_invocations"] = sum(
                len(v) for v in _kernel_invocation_spans(plane).values())
            info["active"] = info["lines"][_COUNTERS_LINE] > 0
            result["devices"][plane.name] = info
            result["kernel_profiling_active"] |= info["active"]

        if not result["devices"]:
            result["note"] = "No device planes found in this trace."
        if not result["kernel_profiling_active"]:
            result["hint"] = _CAPTURE_HINT
        # v7 runtime counter sampling produces separate sampled-counter tracks;
        # their absence on v5p/v6e is expected (silent no-op), not an error.
        result["runtime_counter_sampling"] = (
            "not detected (expected on pre-v7 hardware; requires TPU v7+ and "
            "tpu_enable_periodic_counter_sampling)")
        return json.dumps(result, indent=2)
    except ImportError:
        return xplane_tools._XPLANE_IMPORT_ERROR  # pylint: disable=protected-access
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("check_kernel_profiling failed for run %s", run)
        return f"Error in check_kernel_profiling: {e}"


def list_kernel_invocations(
    run: str, host: str = "", kernel_regex: str = ".*", max_spans: int = 20
) -> str:
    """Lists custom-call (Pallas/Mosaic kernel) invocations in the trace.

    Enumerates kernel executions from the device `XLA Ops` line: op name,
    custom_call_target, invocation count and duration statistics. The
    returned op names are the handles that `get_llo_utilization` and
    `get_kernel_stage_breakdown` accept as `kernel`.

    Args:
        run:          The run name.
        host:         Host name. Defaults to the first host found.
        kernel_regex: Regex filter on the kernel/op name.
        max_spans:    Max invocation spans to return per kernel (default 20).

    Returns:
        JSON: per device plane, a list of kernels with duration stats and
        (capped) invocation spans in absolute picoseconds.
    """
    try:
        xspace = xplane_tools._fetch_xspace(run, host)  # pylint: disable=protected-access
        result: dict = {}
        for plane in _device_planes(xspace):
            spans_by_op = _kernel_invocation_spans(plane, kernel_regex)
            if not spans_by_op:
                continue
            kernels = []
            for op_name, spans in sorted(spans_by_op.items()):
                durations = [e - s for s, e in spans]
                kernels.append({
                    "kernel": op_name,
                    **_duration_stats(durations),
                    "spans_ps": [
                        {"start_ps": s, "end_ps": e} for s, e in spans[:max_spans]
                    ],
                    "spans_truncated": max(0, len(spans) - max_spans),
                })
            result[plane.name] = kernels
        if not result:
            return json.dumps({
                "kernels": [],
                "note": "No custom-call invocations matched. "
                        "Run check_kernel_profiling to verify capture flags.",
            }, indent=2)
        return json.dumps(result, indent=2)
    except ImportError:
        return xplane_tools._XPLANE_IMPORT_ERROR  # pylint: disable=protected-access
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("list_kernel_invocations failed for run %s", run)
        return f"Error in list_kernel_invocations: {e}"


def get_llo_utilization(
    run: str,
    host: str = "",
    kernel: str = "",
    start_time_ps: Optional[int] = None,
    end_time_ps: Optional[int] = None,
    timeline_buckets: int = 0,
) -> str:
    """Per-functional-unit LLO utilization — the kernel bottleneck verdict.

    Aggregates the `_counters_` line (`% util` per ~1us window for MXU,
    Scalar ALU, Vector ALU, Vector Load, Vector Store, XLU, Vector EUP; raw
    counts for Vector Fills / Vector Spills), optionally restricted to a
    kernel's invocation spans or an explicit time range. Answers "which unit
    is the bottleneck": e.g. low MXU + high Vector Load = load/bandwidth
    bound; high Scalar ALU with idle MXU = scalar-core serialization.

    NOTE: values are LLO static slot occupancy over measured time windows —
    label conclusions accordingly (not raw runtime hardware counters).

    Args:
        run:              The run name.
        host:             Host name. Defaults to the first host found.
        kernel:           Kernel/op name regex (from list_kernel_invocations);
                          restricts aggregation to that kernel's spans.
        start_time_ps:    Explicit window start (absolute ps), alternative to
                          `kernel`.
        end_time_ps:      Explicit window end (absolute ps).
        timeline_buckets: If > 0, also return a downsampled per-unit timeline
                          with this many buckets (max 50).

    Returns:
        JSON: per device plane, per-unit {mean, p50, max, samples} of
        `% util`, fills/spills totals, and a computed `verdict` block.
    """
    try:
        xspace = xplane_tools._fetch_xspace(run, host)  # pylint: disable=protected-access
        timeline_buckets = min(timeline_buckets, 50)
        result: dict = {}
        for plane in _device_planes(xspace):
            cline = _find_line(plane, _COUNTERS_LINE)
            if cline is None:
                continue
            windows: list[tuple[int, int]] = []
            if kernel:
                for spans in _kernel_invocation_spans(plane, kernel).values():
                    windows.extend(spans)
                if not windows:
                    result[plane.name] = {
                        "note": f"no custom-call invocations matched kernel={kernel!r}"}
                    continue
            elif start_time_ps is not None or end_time_ps is not None:
                windows = [(start_time_ps or 0, end_time_ps or (1 << 62))]

            names = _event_name_map(plane)
            stat_names = _stat_map(plane)
            samples: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
            counts: dict[str, float] = collections.defaultdict(float)
            for event in cline.events:
                span = _abs_span_ps(cline, event)
                if windows and not _overlaps(span, windows):
                    continue
                unit = names.get(event.metadata_id, "?")
                stats = _event_stats(event, stat_names)
                if "% util" in stats:
                    samples[unit].append((span[0], float(stats["% util"])))
                else:
                    for key in ("fills", "spills"):
                        if key in stats:
                            counts[unit] += float(stats[key])

            units: dict[str, dict] = {}
            for unit, vals in sorted(samples.items()):
                utils = [v for _, v in vals]
                entry = {
                    "samples": len(utils),
                    "mean_util_pct": round(statistics.mean(utils), 2),
                    "p50_util_pct": round(statistics.median(utils), 2),
                    "max_util_pct": round(max(utils), 2),
                }
                if timeline_buckets and len(vals) > 1:
                    vals.sort()
                    t0, t1 = vals[0][0], vals[-1][0]
                    width = max(1, (t1 - t0) // timeline_buckets + 1)
                    buckets: dict[int, list[float]] = collections.defaultdict(list)
                    for t, v in vals:
                        buckets[(t - t0) // width].append(v)
                    entry["timeline_mean_pct"] = [
                        round(statistics.mean(buckets[i]), 1) if i in buckets else None
                        for i in range(timeline_buckets)
                    ]
                units[unit] = entry
            for unit, total in sorted(counts.items()):
                units[unit] = {"total_count": total}

            if not units:
                result[plane.name] = {"note": "no _counters_ samples in window"}
                continue

            util_units = {u: d for u, d in units.items() if "mean_util_pct" in d}
            verdict: dict = {}
            if util_units:
                dominant = max(util_units, key=lambda u: util_units[u]["mean_util_pct"])
                verdict["dominant_unit"] = dominant
                mxu = util_units.get("MXU", {}).get("mean_util_pct", 0.0)
                vload = util_units.get("Vector Load", {}).get("mean_util_pct", 0.0)
                salu = util_units.get("Scalar ALU", {}).get("mean_util_pct", 0.0)
                verdict["mxu_mean_util_pct"] = mxu
                verdict["memory_bound_signal"] = bool(vload > mxu)
                verdict["scalar_bound_signal"] = bool(salu > mxu)
            result[plane.name] = {
                "window": ("kernel spans" if kernel else
                           "explicit range" if windows else "whole trace"),
                "units": units,
                "verdict": verdict,
                "semantics": "LLO static slot occupancy over measured windows",
            }
        if not result:
            return json.dumps({
                "note": "No _counters_ line found on any device plane.",
                "hint": _CAPTURE_HINT,
            }, indent=2)
        return json.dumps(result, indent=2)
    except ImportError:
        return xplane_tools._XPLANE_IMPORT_ERROR  # pylint: disable=protected-access
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("get_llo_utilization failed for run %s", run)
        return f"Error in get_llo_utilization: {e}"


def get_kernel_stage_breakdown(run: str, host: str = "", kernel: str = "") -> str:
    """Mosaic pipeline stage time breakdown for Pallas kernels.

    Aggregates the `XLA TraceMe` line's stage scopes (`ep_wait_in`,
    `ep_copy_in`, `ep_run_kernel`, user `named_scope`s, `ep_copy_out`,
    `ep_wait_out`, ...) into total/mean time per stage and computes
    `wait_ratio` = (ep_wait_in + ep_wait_out) / ep_run_kernel — the
    buffering-effectiveness metric (high ratio = compute starved on DMAs;
    consider more pipeline buffers / larger blocks).

    Args:
        run:    The run name.
        host:   Host name. Defaults to the first host found.
        kernel: Kernel/op name regex; restricts to that kernel's spans.

    Returns:
        JSON: per device plane, per-stage {count, total_us, mean_us},
        stages classified as pipeline vs user named_scope, and wait_ratio.
    """
    try:
        xspace = xplane_tools._fetch_xspace(run, host)  # pylint: disable=protected-access
        result: dict = {}
        for plane in _device_planes(xspace):
            tline = _find_line(plane, _TRACEME_LINE)
            if tline is None:
                continue
            windows: list[tuple[int, int]] = []
            if kernel:
                for spans in _kernel_invocation_spans(plane, kernel).values():
                    windows.extend(spans)
                if not windows:
                    result[plane.name] = {
                        "note": f"no custom-call invocations matched kernel={kernel!r}"}
                    continue

            names = _event_name_map(plane)
            durations: dict[str, list[int]] = collections.defaultdict(list)
            for event in tline.events:
                span = _abs_span_ps(tline, event)
                if windows and not _overlaps(span, windows):
                    continue
                durations[names.get(event.metadata_id, "?")].append(event.duration_ps)

            if not durations:
                result[plane.name] = {"note": "no XLA TraceMe events in window"}
                continue

            stages = {
                stage: {
                    **_duration_stats(durs),
                    "kind": "pipeline" if stage in _EP_STAGES else "named_scope",
                }
                for stage, durs in sorted(durations.items())
            }
            total = lambda s: sum(durations.get(s, []))  # noqa: E731
            run_total = total("ep_run_kernel")
            wait_total = total("ep_wait_in") + total("ep_wait_out")
            entry: dict = {"stages": stages}
            if run_total:
                entry["wait_ratio"] = round(wait_total / run_total, 3)
                entry["wait_ratio_meaning"] = (
                    "(ep_wait_in+ep_wait_out)/ep_run_kernel; >~0.3 suggests "
                    "DMA-starved compute (buffering/block-size lever)")
            result[plane.name] = entry
        if not result:
            return json.dumps({
                "note": "No XLA TraceMe line found on any device plane.",
                "hint": _CAPTURE_HINT,
            }, indent=2)
        return json.dumps(result, indent=2)
    except ImportError:
        return xplane_tools._XPLANE_IMPORT_ERROR  # pylint: disable=protected-access
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("get_kernel_stage_breakdown failed for run %s", run)
        return f"Error in get_kernel_stage_breakdown: {e}"
