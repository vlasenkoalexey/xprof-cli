"""Whole-model analysis tools over xprof converters.

Adds the converter-backed analyses the original MCP never surfaced:
roofline (per-op compute/memory-bound classification), pod/megascale
collective stats, per-buffer memory attribution, host input-pipeline
analysis, xprof's own smart suggestions, framework op stats, and a
consolidated KPI summary.

Every tool returns JSON text. The roofline tool embeds an explicit
`caveats` block — its numbers are XLA cost-model estimates with known
blind spots (custom calls, communication), and downstream agents must see
those limits next to the data, not discover them the hard way.
"""

import json
import logging
from typing import Any, Optional

from xprof_mcp.internal import xprof_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# gviz DataTable helpers (the converters emit google.visualization tables)
# ---------------------------------------------------------------------------


def _datatable_records(table: dict, limit: int = 0) -> list[dict]:
    """Converts one gviz DataTable ({cols, rows, p}) to a list of dicts."""
    col_ids = [col.get("id") or col.get("label") for col in table.get("cols", [])]
    records = []
    for row in table.get("rows", []):
        cells = row.get("c", [])
        rec = {}
        for i, col_id in enumerate(col_ids):
            if i < len(cells) and cells[i] is not None:
                rec[col_id] = cells[i].get("v")
        records.append(rec)
        if limit and len(records) >= limit:
            break
    return records


def _fetch_json(tool: str, run: str, host: str = "ALL_HOSTS", **kwargs) -> Any:
    client = xprof_client.get_client()
    return json.loads(client.fetch(tool, run, host=host, **kwargs))


def _error(tool: str, run: str, exc: Exception) -> str:
    logger.warning("%s failed for run %s", tool, run, exc_info=True)
    return json.dumps(
        {"error": f"{type(exc).__name__}: {exc}", "tool": tool, "run": run},
        indent=2,
    )


# ---------------------------------------------------------------------------
# Roofline
# ---------------------------------------------------------------------------

_ROOFLINE_CAVEATS = [
    "FLOPs and bytes are XLA cost-model estimates stamped at compile time; "
    "only TIME is measured. Treat utilization numbers as model-derived.",
    "custom_call ops (Pallas/Mosaic kernels, e.g. splash attention, "
    "segment_matmul, tokamax CE) have no cost-model FLOPs: their FLOP "
    "utilization reads near zero and their operational intensity is "
    "meaningless regardless of actual efficiency. Judge hand-written "
    "kernels with get_llo_utilization / get_kernel_stage_breakdown instead.",
    "bound_by classifies compute vs memory tiers (HBM/VMEM/CMEM) ONLY — "
    "there is no communication class. Collectives (~0 cost-model FLOPs, "
    "nonzero bytes) get mis-bucketed as HBM-bound. For ICI/DCN attribution "
    "use get_pod_viewer / get_megascale_stats.",
    "The 'Program' row folds idle time into its denominator, so its "
    "utilization reads pessimistically low; prefer the op-level rows.",
    "bytes_accessed is the op's logical traffic, not measured HBM traffic "
    "(fusion and VMEM residency are invisible), so memory-vs-compute calls "
    "near the ridge point are soft.",
    "On TPU v5p/v6e there is no hardware-counter cross-check (counter "
    "sampling is v7+ only): the cost-model series is the only series.",
]

# The load-bearing subset of the 37 roofline columns.
_ROOFLINE_FIELDS = (
    "step",
    "rank",
    "category",
    "operation",
    "occurrences",
    "total_time",
    "total_self_time_percent",
    "dma_stall_percent",
    "measured_flop_rate",
    "hbm_bw",
    "operational_intensity",
    "hbm_operational_intensity",
    "bottleneck_operational_intensity",
    "bound_by",
    "roofline_efficiency",
    "compute_efficiency",
    "max_mem_bw_utilization",
)


def get_roofline_model(
    run: str,
    host: str = "",
    step_filter: str = "Total",
    limit: int = 25,
) -> str:
    """Per-op roofline analysis: compute- vs memory-bound classification.

    For each op: operational intensity (FLOP/byte), achieved vs peak FLOP
    rate, per-memory-tier bandwidth utilization, and a `bound_by` verdict
    (compute / HBM / VMEM read/write / CMEM). Device peaks and ridge
    points are included under `device`.

    READ THE `caveats` FIELD BEFORE ACTING ON THE NUMBERS — FLOPs/bytes
    are cost-model estimates and blind to custom calls + communication.

    Args:
        run: Profile run name (see list_runs).
        host: Specific host, or empty for all hosts.
        step_filter: Row scope — "Total" (whole profile, default),
            "Average" (per-step average), or "" for every row including
            per-step records.
        limit: Max op rows returned (ranked by total time).

    Returns:
        JSON: {run, device, records, diagnostics, caveats}.
    """
    try:
        payload = _fetch_json("roofline_model", run, host=host or "ALL_HOSTS")
        roofline_table = payload[0] if payload else {}
        diagnostics = payload[1] if len(payload) > 1 else {}

        records = _datatable_records(roofline_table)
        if step_filter:
            records = [r for r in records if r.get("step") == step_filter]
        slim = [
            {k: r.get(k) for k in _ROOFLINE_FIELDS if k in r} for r in records
        ][: limit or None]

        device = dict(roofline_table.get("p", {}))
        return json.dumps(
            {
                "run": run,
                "device": device,
                "records": slim,
                "record_count_total": len(records),
                "diagnostics": _datatable_records(diagnostics),
                "caveats": _ROOFLINE_CAVEATS,
            },
            indent=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_roofline_model", run, e)


# ---------------------------------------------------------------------------
# Communication / multi-chip
# ---------------------------------------------------------------------------


def get_pod_viewer(run: str, host: str = "") -> str:
    """Pod-level view: per-core step breakdown and collective/ICI stats.

    The communication attribution the roofline tool cannot provide (it has
    no comm-bound class). Empty podStatsMap usually means a single-host
    capture without step-level pod stats.

    Args:
        run: Profile run name.
        host: Specific host, or empty for all hosts.

    Returns:
        JSON passthrough of the pod_viewer converter output.
    """
    try:
        payload = _fetch_json("pod_viewer", run, host=host or "ALL_HOSTS")
        return json.dumps({"run": run, "pod_viewer": payload}, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_pod_viewer", run, e)


def get_megascale_stats(run: str, host: str = "", limit: int = 50) -> str:
    """Multi-slice (DCN) collective stats: per-rendezvous latencies/sizes.

    Relevant only for multi-slice workloads (data goes over DCN); empty on
    single-slice runs. For within-slice ICI collectives use get_pod_viewer.

    Args:
        run: Profile run name.
        host: Specific host, or empty for all hosts.
        limit: Max rows per table.

    Returns:
        JSON: {run, tables: [records...]}.
    """
    try:
        payload = _fetch_json("megascale_stats", run, host=host or "ALL_HOSTS")
        tables = payload if isinstance(payload, list) else [payload]
        return json.dumps(
            {
                "run": run,
                "tables": [_datatable_records(t, limit=limit) for t in tables],
            },
            indent=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_megascale_stats", run, e)


# ---------------------------------------------------------------------------
# Memory attribution
# ---------------------------------------------------------------------------


def get_memory_viewer(
    run: str, module_name: str = "", memory_space: str = "0"
) -> str:
    """Per-buffer HBM attribution for one HLO module: which tensors hold peak.

    Answers "WHICH buffer owns the peak", not just the peak number
    (get_memory_profile). Lists modules when module_name is omitted.

    Args:
        run: Profile run name.
        module_name: HLO module (see list_hlo_modules); empty lists modules.
        memory_space: "0" = HBM (default); other spaces per device.

    Returns:
        JSON: buffer allocation table + peak stats for the module, or the
        available module list.
    """
    try:
        client = xprof_client.get_client()
        if not module_name:
            modules = client.get_hlo_module_list(run)
            return json.dumps(
                {
                    "run": run,
                    "modules": modules.split(",") if modules else [],
                    "tip": "Pass module_name=<one of these> for the buffer map.",
                },
                indent=2,
            )
        payload = json.loads(
            client.fetch(
                "memory_viewer",
                run,
                module_name=module_name,
                memory_space=memory_space,
            )
        )
        return json.dumps(
            {"run": run, "module": module_name, "memory_viewer": payload},
            indent=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_memory_viewer", run, e)


# ---------------------------------------------------------------------------
# Host / input pipeline
# ---------------------------------------------------------------------------


def get_input_pipeline(run: str, host: str = "", limit: int = 30) -> str:
    """Host-vs-device input-pipeline analysis: is the device data-starved?

    Decomposes step time into device compute vs host/input stalls — the
    coarse `bottleneck_category` from get_overview, with the host side
    broken down (enqueue, preprocessing, I/O wait).

    Args:
        run: Profile run name.
        host: Specific host, or empty for all hosts.
        limit: Max rows per table.

    Returns:
        JSON: {run, tables: [records...]}.
    """
    try:
        payload = _fetch_json(
            "input_pipeline_analyzer", run, host=host or "ALL_HOSTS"
        )
        tables = payload if isinstance(payload, list) else [payload]
        out = []
        for t in tables:
            out.append(
                {
                    "properties": t.get("p", {}),
                    "records": _datatable_records(t, limit=limit),
                }
            )
        return json.dumps({"run": run, "tables": out}, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_input_pipeline", run, e)


# ---------------------------------------------------------------------------
# Framework op stats / smart suggestions / KPIs
# ---------------------------------------------------------------------------


def get_framework_op_stats(run: str, host: str = "", limit: int = 40) -> str:
    """Device time attributed to framework-level op names (JAX/PyTorch/TF).

    Useful on lanes where raw HLO names are opaque (torch_tpu, torchax):
    maps time back to the op names the model code actually calls.

    Args:
        run: Profile run name.
        host: Specific host, or empty for all hosts.
        limit: Max rows returned.

    Returns:
        JSON: {run, records}.
    """
    try:
        payload = _fetch_json(
            "framework_op_stats", run, host=host or "ALL_HOSTS"
        )
        table = payload[0] if isinstance(payload, list) else payload
        return json.dumps(
            {"run": run, "records": _datatable_records(table, limit=limit)},
            indent=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_framework_op_stats", run, e)


def get_utilization_viewer(run: str, host: str = "", limit: int = 100) -> str:
    """Sampled utilization timeline: achieved vs peak per node over time.

    Complements the single-number duty cycle from get_overview with a
    time-resolved view — reveals utilization phases (warmup, steady-state,
    stragglers) a scalar average hides.

    Args:
        run: Profile run name.
        host: Specific host, or empty for all hosts.
        limit: Max sample rows returned.

    Returns:
        JSON: {run, records}.
    """
    try:
        payload = _fetch_json(
            "utilization_viewer", run, host=host or "ALL_HOSTS"
        )
        table = payload[0] if isinstance(payload, list) else payload
        return json.dumps(
            {"run": run, "records": _datatable_records(table, limit=limit)},
            indent=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_utilization_viewer", run, e)


def get_smart_suggestions(run: str, host: str = "") -> str:
    """xprof's own automated bottleneck triage suggestions.

    Cheap first-pass second opinion; may be empty for many profiles.

    Args:
        run: Profile run name.
        host: Specific host, or empty for all hosts.

    Returns:
        JSON passthrough: {run, suggestions}.
    """
    try:
        payload = _fetch_json("smart_suggestion", run, host=host or "ALL_HOSTS")
        return json.dumps({"run": run, **payload}, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_smart_suggestions", run, e)


def get_perf_counters(run: str, host: str = "", limit: int = 100) -> str:
    """Hardware performance counters (TPU v7+/Ironwood; GPU).

    Measured HW counters — the ground truth that cross-checks the
    roofline's cost-model estimates. IMPORTANT: counter sampling is
    silently absent on TPU v5p/v6e (empty result ≠ zero activity; it
    means the hardware/runtime doesn't sample counters). On v7+ the
    capture must enable counter sampling (see docs/KERNEL_PROFILING.md).

    Args:
        run: Profile run name.
        host: Specific host, or empty for all hosts.
        limit: Max rows per table.

    Returns:
        JSON: {run, tables, note} — tables empty on pre-v7 TPU captures.
    """
    try:
        payload = _fetch_json("perf_counters", run, host=host or "ALL_HOSTS")
        tables = payload if isinstance(payload, list) else [payload]
        records = [
            _datatable_records(t, limit=limit)
            for t in tables
            if isinstance(t, dict)
        ]
        empty = not any(records)
        return json.dumps(
            {
                "run": run,
                "tables": records,
                "note": (
                    "EMPTY: no counter samples in this capture. On TPU "
                    "v5p/v6e counter sampling is silently unavailable — "
                    "this is expected, not a capture error. On v7+ enable "
                    "counter sampling at capture time."
                )
                if empty
                else "Measured HW counters — ground truth vs the "
                "roofline's cost-model estimates.",
            },
            indent=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_perf_counters", run, e)


def get_kpi_metrics(run: str) -> str:
    """Consolidated KPI summary: step time, duty cycle, MFU-ish, peak HBM.

    One-call header for experiment pages: pulls the headline numbers from
    the overview and memory profile. The roofline utilization figure here
    is the PROGRAM-LEVEL scalar — idle-folded and cost-model-derived (see
    get_roofline_model caveats); use it for trend, not truth.

    Args:
        run: Profile run name.

    Returns:
        JSON: {run, step_time_ms, duty_cycle_percent, mxu_utilization_percent,
        roofline_utilization, peak_hbm_gib, device}.
    """
    try:
        from xprof_mcp.tools import get_memory_profile_tool  # pylint: disable=g-import-not-at-top
        from xprof_mcp.tools import get_overview_tool  # pylint: disable=g-import-not-at-top

        overview = json.loads(get_overview_tool.get_overview(run))
        if "error" in overview:
            return json.dumps(
                {"error": overview["error"], "tool": "get_kpi_metrics"}, indent=2
            )
        perf = overview.get("performance_summary", {})
        env = overview.get("run_environment", {})

        peak_hbm: Optional[Any] = None
        try:
            memory = json.loads(
                get_memory_profile_tool.get_memory_profile(run)
            )
            peak_hbm = memory.get("peak_memory_usage_gib")
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        return json.dumps(
            {
                "run": run,
                "step_time_ms": perf.get("steptime_ms_average"),
                "duty_cycle_percent": perf.get("device_duty_cycle_percent"),
                "mxu_utilization_percent": perf.get("mxu_utilization_percent"),
                "roofline_utilization": perf.get(
                    "flop_rate_utilization_relative_to_roofline"
                ),
                "peak_hbm_gib": peak_hbm,
                "device": {
                    "device_type": env.get("device_type"),
                    "device_core_count": env.get("device_core_count"),
                },
                "note": (
                    "roofline_utilization is the program-level cost-model "
                    "scalar; see get_roofline_model caveats before acting on it."
                ),
            },
            indent=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return _error("get_kpi_metrics", run, e)
