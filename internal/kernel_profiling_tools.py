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


def _weighted_median(pairs: list[tuple[int, float]]) -> float:
    """Median of `value` weighted by `weight` — the value at which half the
    total weight lies below. Plain statistics.median treats a 78-ps sample and
    a 4-us sample as equals; this does not."""
    if not pairs:
        raise ValueError("no samples")
    ordered = sorted(pairs, key=lambda p: p[1])
    total = sum(w for w, _ in ordered)
    if total <= 0:
        return statistics.median([v for _, v in ordered])
    seen = 0.0
    for w, v in ordered:
        seen += w
        if seen >= total / 2.0:
            return v
    return ordered[-1][1]


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
        timeline_buckets = min(int(timeline_buckets), 50)
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
                    samples[unit].append(
                        (span[0], max(0, span[1] - span[0]), float(stats["% util"])))
                else:
                    for key in ("fills", "spills"):
                        if key in stats:
                            counts[unit] += float(stats[key])

            # Sampling intervals in a real trace span ~9 orders of magnitude
            # (78 ps to 4.19 us measured on a v6e Pallas capture), so an
            # UNWEIGHTED mean lets a 78-ps blip at 79% outvote a 4-us stretch
            # at 0%. Measured error on that capture: MXU 39.24% unweighted vs
            # 0.2647% duration-weighted, a 148x overstatement of the headline
            # number pasted into experiment pages. Weight every statistic by
            # how long the sample actually covered.
            #
            # Guard first: some exporters write an absolute end timestamp into
            # duration_ps, yielding samples longer than the whole trace (three
            # _counters_ events at ~81.9 ms in a 605 us capture). Those would
            # dominate any duration weighting outright.
            # Derive the reference span from sample START times only. Using
            # start+duration would let a malformed sample define the very span
            # it is checked against, so the guard could never fire on the case
            # it exists for.
            _starts = [s for vals in samples.values() for s, _, _ in vals]
            trace_span = max(1, (max(_starts) - min(_starts)) if _starts else 1)

            units: dict[str, dict] = {}
            for unit, vals in sorted(samples.items()):
                # A single sample may not exceed the whole observed span.
                good = [(s, d, v) for s, d, v in vals if 0 < d <= trace_span]
                malformed = len(vals) - len(good)
                utils = [v for _, _, v in vals]
                entry = {
                    "samples": len(utils),
                    "max_util_pct": round(max(utils), 2),
                    "mean_util_pct_unweighted": round(statistics.mean(utils), 2),
                    "p50_util_pct_unweighted": round(statistics.median(utils), 2),
                }
                if good:
                    tot = sum(d for _, d, _ in good)
                    entry["mean_util_pct"] = round(
                        sum(d * v for _, d, v in good) / tot, 2)
                    entry["p50_util_pct"] = round(
                        _weighted_median([(d, v) for _, d, v in good]), 2)
                    entry["weighted_by"] = "sample duration_ps"
                    entry["covered_ps"] = tot
                else:
                    # No usable durations — report the unweighted figure but
                    # say so rather than passing it off as a weighted mean.
                    entry["mean_util_pct"] = round(statistics.mean(utils), 2)
                    entry["p50_util_pct"] = round(statistics.median(utils), 2)
                    entry["weighted_by"] = ("none — no sample carried a usable "
                                            "duration; values are unweighted")
                if malformed:
                    entry["malformed_duration_samples"] = malformed
                    entry["malformed_note"] = (
                        "%d sample(s) reported a duration longer than the whole "
                        "observed span (%d ps) — an exporter writing an absolute "
                        "end into duration_ps; excluded from the weighting"
                        % (malformed, trace_span))
                if timeline_buckets and len(vals) > 1:
                    vals.sort()
                    t0, t1 = vals[0][0], vals[-1][0]
                    width = max(1, (t1 - t0) // timeline_buckets + 1)
                    buckets: dict[int, list[tuple[int, float]]] = (
                        collections.defaultdict(list))
                    for t, d, v in (good or [(s, 1, v) for s, _, v in vals]):
                        buckets[(t - t0) // width].append((d, v))
                    entry["timeline_mean_pct"] = [
                        round(sum(d * v for d, v in buckets[i])
                              / max(1, sum(d for d, _ in buckets[i])), 1)
                        if i in buckets else None
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
                # A unit absent from util_units had NO samples in this window
                # (wrong plane kind, renamed unit on another TPU generation, or
                # simply no overlap). Defaulting it to 0.0 asserted "the MXU was
                # idle" / "not memory bound" from no evidence at all, and both
                # booleans were then derived from that fabricated zero.
                mxu = util_units.get("MXU", {}).get("mean_util_pct")
                vload = util_units.get("Vector Load", {}).get("mean_util_pct")
                salu = util_units.get("Scalar ALU", {}).get("mean_util_pct")
                missing = [n for n, v in (("MXU", mxu), ("Vector Load", vload),
                                          ("Scalar ALU", salu)) if v is None]
                verdict["mxu_mean_util_pct"] = mxu
                if missing:
                    verdict["status"] = "indeterminate"
                    verdict["missing_units"] = missing
                    verdict["note"] = (
                        "no samples for %s in this window — bound signals are "
                        "not derivable and are omitted rather than defaulted "
                        "to zero" % ", ".join(missing))
                else:
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


# ---------------------------------------------------------------------------
# Device vs wall dual report
# ---------------------------------------------------------------------------

_WALL_KEY_RE = re.compile(r"(^|_)(wall_)?p50(_ms|_us|_ns)?$|^wall_ms$|^wall_us$")

_WALL_RATIO_LABEL = (
    "wall_ratio (deployable — what a caller of the op actually gets)")
_DEVICE_RATIO_LABEL = (
    "device_ratio (device-framing — overstates the win when dispatch/host "
    "overhead dominates the wall time)")


def _extract_wall_ms(obj, path: str = "") -> list[tuple[str, float]]:
    """Finds wall-p50-like numeric leaves in a measurement JSON.

    Accepts the common shapes: {"wall_p50_ms": x}, {"p50_ms": x},
    {"p50_us": x}, {"wall": {"p50_ms": x}}, lists of runs, etc. Returns
    [(json_path, value_ms), ...] in document order.
    """
    found: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(v, (int, float)) and _WALL_KEY_RE.search(str(k)):
                scale = (1e-3 if str(k).endswith("_us")
                         else 1e-6 if str(k).endswith("_ns") else 1.0)
                found.append((p, float(v) * scale))
            else:
                found.extend(_extract_wall_ms(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_extract_wall_ms(v, f"{path}[{i}]"))
    return found


def _floor_check(claim_ms: Optional[float], floor_ms: float,
                 what: str) -> Optional[dict]:
    if claim_ms is None or not floor_ms or claim_ms >= floor_ms:
        return None
    return {
        "claim": what,
        "claim_ms": claim_ms,
        "floor_ms": floor_ms,
        "verdict": "PHYSICALLY_IMPOSSIBLE",
        "detail": (f"{what} of {claim_ms:.6g} ms is "
                   f"{floor_ms / claim_ms:.1f}x below the stated physical "
                   f"floor of {floor_ms:.6g} ms — the measurement or the "
                   f"framing is wrong, not the hardware"),
    }


def _pop_nested(node, key: str) -> list:
    """Remove every occurrence of `key` at any depth, returning what was
    removed. Companion to _extract_wall_ms, which also recurses — a top-level
    pop alone leaves nested baseline subtrees in the tree to be mistaken for
    the candidate."""
    found = []
    if isinstance(node, dict):
        if key in node:
            found.append(node.pop(key))
        for v in list(node.values()):
            found.extend(_pop_nested(v, key))
    elif isinstance(node, list):
        for v in node:
            found.extend(_pop_nested(v, key))
    return found


def _device_p50_ms(run: str, host: str, kernel: str) -> Optional[float]:
    """p50 device-busy per invocation (ms) for the matching kernel spans."""
    xspace = xplane_tools._fetch_xspace(run, host)  # pylint: disable=protected-access
    durations: list[int] = []
    for plane in _device_planes(xspace):
        for spans in _kernel_invocation_spans(plane, kernel or ".*").values():
            durations.extend(e - s for s, e in spans)
    if not durations:
        return None
    return statistics.median(durations) / 1e9  # ps -> ms


def get_device_wall_report(
    run: str,
    host: str = "",
    kernel: str = "",
    measure_json: str = "",
    baseline_run: str = "",
    floor_ms: float = 0.0,
) -> str:
    """Device-busy vs wall-clock dual report — keeps both framings labeled.

    Device time (from the trace) and wall time (from the caller's own
    measurement) routinely diverge by 1.6-27x on dispatch-bound kernels;
    reporting only one framing is the classic way a kernel "win" evaporates
    in deployment. This tool reports both, labels both ratios, and (with
    `floor_ms`) stamps sub-physical claims PHYSICALLY_IMPOSSIBLE.

    Args:
        run:          Run name whose trace provides device-busy time.
        host:         Host name. Defaults to the first host found.
        kernel:       Kernel/op regex (see list_kernel_invocations) to
                      restrict device spans; default all instrumented ops.
        measure_json: Path to a wall-clock measurement JSON (e.g. a
                      benchmark-harness/kernelgate result). The first
                      wall/p50-like key found is the candidate wall p50;
                      a `baseline` subtree provides the baseline wall p50.
                      Accepted key shapes: wall_p50_ms / p50_ms / p50_us /
                      {"wall": {"p50_ms": ...}}.
        baseline_run: Optional second run for the baseline device-busy p50.
        floor_ms:     Optional physical floor (e.g. HBM roofline time). Any
                      claim below it is stamped PHYSICALLY_IMPOSSIBLE with
                      the multiple below floor.

    Returns:
        JSON: device/wall p50s, gap_pct, labeled wall_ratio + device_ratio
        (when a baseline exists), floor audit, and framing caveats.
    """
    try:
        floor_ms = float(floor_ms)
        result: dict = {"run": run, "kernel": kernel or "(all instrumented)"}

        device_ms = _device_p50_ms(run, host, kernel)
        result["device_busy_p50_ms"] = device_ms
        if device_ms is None:
            result["device_note"] = (
                "no instrumented kernel spans found in trace; "
                + _CAPTURE_HINT)

        wall_ms = None
        baseline_wall_ms = None
        if measure_json:
            with open(measure_json, encoding="utf-8") as f:
                doc = json.load(f)
            # `pop` only removes a TOP-LEVEL "baseline", but _extract_wall_ms
            # recurses — so for a nested doc such as
            #   {"results": {"baseline": {...}, "candidate": {...}}}
            # the baseline's p50 stayed in the tree and, being first in
            # document order, was reported as the CANDIDATE's wall time.
            baseline_doc = doc.pop("baseline", None) if isinstance(
                doc, dict) else None
            nested_baselines = _pop_nested(doc, "baseline")
            if baseline_doc is None and nested_baselines:
                baseline_doc = nested_baselines[0]
            walls = _extract_wall_ms(doc)
            if walls:
                result["wall_source"] = {"file": measure_json,
                                         "key": walls[0][0]}
                wall_ms = walls[0][1]
                if len(walls) > 1:
                    # Don't silently take [0] when the document offers several.
                    result["wall_ambiguous"] = {
                        "note": "multiple wall/p50-like keys matched; used the "
                                "first in document order — confirm it is the "
                                "candidate's",
                        "candidates": [{"key": k, "value_ms": v}
                                       for k, v in walls[:8]],
                    }
            else:
                result["wall_note"] = (
                    f"no wall/p50-like key found in {measure_json!r} "
                    "(accepted: wall_p50_ms, p50_ms, p50_us, wall.p50_ms)")
            if baseline_doc is not None:
                bwalls = _extract_wall_ms(baseline_doc)
                if bwalls:
                    baseline_wall_ms = bwalls[0][1]
        result["wall_p50_ms"] = wall_ms

        if wall_ms is not None and device_ms is not None:
            gap_pct = round(100.0 * (wall_ms - device_ms) / wall_ms, 1)
            result["gap_pct"] = gap_pct
            if device_ms > wall_ms:
                result["framing"] = (
                    "device-busy EXCEEDS wall — overlapping invocations or "
                    "mismatched aggregation windows; the device framing is "
                    "unreliable here, trust wall")
            elif abs(gap_pct) < 5.0:
                result["framing"] = (
                    "framings agree (gap < 5%) — dispatch overhead is "
                    "negligible; either number is honest")
            else:
                result["framing"] = (
                    f"wall is {wall_ms / device_ms:.2f}x device-busy — "
                    "dispatch/host overhead owns the difference; quote the "
                    "wall number for deployable claims")

        baseline_device_ms = None
        if baseline_run:
            baseline_device_ms = _device_p50_ms(baseline_run, host, kernel)
            result["baseline_run"] = baseline_run
            result["baseline_device_busy_p50_ms"] = baseline_device_ms
        if baseline_wall_ms is not None:
            result["baseline_wall_p50_ms"] = baseline_wall_ms

        ratios: dict = {}
        if wall_ms and baseline_wall_ms:
            ratios["wall_ratio"] = round(baseline_wall_ms / wall_ms, 2)
            ratios["wall_ratio_label"] = _WALL_RATIO_LABEL
        if device_ms and baseline_device_ms:
            ratios["device_ratio"] = round(
                baseline_device_ms / device_ms, 2)
            ratios["device_ratio_label"] = _DEVICE_RATIO_LABEL
        if ratios:
            result["ratios"] = ratios
            if "wall_ratio" in ratios and "device_ratio" in ratios and (
                    ratios["device_ratio"] > 1.5 * ratios["wall_ratio"]):
                result["ratio_warning"] = (
                    "device_ratio overstates wall_ratio by >1.5x — the "
                    "device framing is inflating this win (dispatch-bound "
                    "regime); report wall_ratio")

        floor_audit = [a for a in (
            _floor_check(wall_ms, floor_ms, "wall p50"),
            _floor_check(device_ms, floor_ms, "device-busy p50"),
        ) if a]
        if floor_ms:
            result["floor_ms"] = floor_ms
            result["floor_audit"] = floor_audit or [
                {"verdict": "OK", "detail": "no claim below the floor"}]

        result["caveat"] = (
            "device-busy comes from instrumented custom-call spans in the "
            "trace (LLO kernel-profiling flags); wall comes from the "
            "caller's measurement file. Neither alone is 'the' time: wall "
            "is what deploys, device is what the kernel costs on-chip.")
        return json.dumps(result, indent=2)
    except ImportError:
        return xplane_tools._XPLANE_IMPORT_ERROR  # pylint: disable=protected-access
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("get_device_wall_report failed for run %s", run)
        return f"Error in get_device_wall_report: {e}"
