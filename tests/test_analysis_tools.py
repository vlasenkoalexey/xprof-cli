"""Golden/schema tests for the whole-model analysis tools (local mode).

Runs every converter-backed tool against the committed v6e fixture trace
(tests/fixtures/logdir, run 'testrun') with XPROF_MODE=local — i.e. fully
in-process, no xprof server. Assertions are schema-level (keys present,
types, stable structural facts), not float snapshots.

Includes the roofline CAVEAT CONTRACT: get_roofline_model output must
carry its trust-boundary caveats (cost-model FLOPs, custom-call
blindness, no communication-bound class). If a refactor drops them, these
tests fail — agents acting on roofline numbers without the caveats is the
failure mode this guards against.
"""

import json

import pytest

from tests.conftest import FIXTURE_RUN

pytest.importorskip("xprof.convert.raw_to_tool_data",
                    reason="xprof pip converters required for local mode")

from xprof_mcp.tools import analysis_tools  # noqa: E402
from xprof_mcp.tools import get_overview_tool  # noqa: E402
from xprof_mcp.tools import get_top_hlo_ops_tool  # noqa: E402

MODULE_NAME = "jit_matmul_optimized(17342868964687645945)"


# ---------------------------------------------------------------------------
# Roofline + caveat contract
# ---------------------------------------------------------------------------


def test_roofline_model_schema(local_mode):
    out = json.loads(analysis_tools.get_roofline_model(FIXTURE_RUN))
    assert "error" not in out
    assert out["run"] == FIXTURE_RUN
    # Device envelope: peaks + ridge points present.
    assert "peak_flop_rate" in out["device"]
    assert "hbm_ridge_point" in out["device"]
    # Records: Program row + at least one op row, key fields present.
    assert len(out["records"]) >= 2
    ops = [r for r in out["records"] if r.get("category") != "Program"]
    assert ops, "expected at least one non-Program op record"
    for field in ("operation", "bound_by", "roofline_efficiency"):
        assert field in ops[0]


def test_roofline_caveats_contract(local_mode):
    out = json.loads(analysis_tools.get_roofline_model(FIXTURE_RUN))
    caveats = " ".join(out["caveats"]).lower()
    assert len(out["caveats"]) >= 5
    # The three load-bearing trust boundaries must be stated.
    assert "cost-model" in caveats
    assert "custom_call" in caveats
    assert "communication" in caveats and "hbm-bound" in caveats


def test_roofline_step_filter(local_mode):
    total = json.loads(
        analysis_tools.get_roofline_model(FIXTURE_RUN, step_filter="Total")
    )
    everything = json.loads(
        analysis_tools.get_roofline_model(FIXTURE_RUN, step_filter="")
    )
    assert everything["record_count_total"] >= len(total["records"])
    assert all(r["step"] == "Total" for r in total["records"])


# ---------------------------------------------------------------------------
# Communication / memory / host tools
# ---------------------------------------------------------------------------


def test_pod_viewer_schema(local_mode):
    out = json.loads(analysis_tools.get_pod_viewer(FIXTURE_RUN))
    assert "error" not in out
    assert "podStatsSequence" in out["pod_viewer"]


def test_megascale_stats_schema(local_mode):
    out = json.loads(analysis_tools.get_megascale_stats(FIXTURE_RUN))
    assert "error" not in out
    assert isinstance(out["tables"], list)  # empty on single-slice fixture


def test_memory_viewer_lists_modules_without_module_name(local_mode):
    out = json.loads(analysis_tools.get_memory_viewer(FIXTURE_RUN))
    assert MODULE_NAME in out["modules"]


def test_memory_viewer_buffer_map(local_mode):
    out = json.loads(
        analysis_tools.get_memory_viewer(FIXTURE_RUN, module_name=MODULE_NAME)
    )
    assert "error" not in out
    mv = out["memory_viewer"]
    # Per-buffer attribution tables — the reason this tool exists.
    assert "maxHeap" in mv
    assert "peakHeapMib" in mv


def test_input_pipeline_schema(local_mode):
    out = json.loads(analysis_tools.get_input_pipeline(FIXTURE_RUN))
    assert "error" not in out
    assert out["tables"] and out["tables"][0]["records"]


def test_framework_op_stats_schema(local_mode):
    out = json.loads(analysis_tools.get_framework_op_stats(FIXTURE_RUN))
    assert "error" not in out
    assert out["records"], "expected framework op records"
    assert "operation" in out["records"][0]


def test_utilization_viewer_schema(local_mode):
    out = json.loads(analysis_tools.get_utilization_viewer(FIXTURE_RUN))
    assert "error" not in out
    assert isinstance(out["records"], list)
    if out["records"]:
        assert "achieved" in out["records"][0]


def test_detect_unfused_reshapes_schema(local_mode):
    from xprof_mcp.tools import detect_tools

    out = json.loads(detect_tools.detect_unfused_reshapes(FIXTURE_RUN))
    assert "error" not in out
    assert "bottlenecks_found" in out
    assert isinstance(out["inefficient_ops"], list)


def test_smart_suggestions_schema(local_mode):
    out = json.loads(analysis_tools.get_smart_suggestions(FIXTURE_RUN))
    assert "error" not in out
    assert "suggestions" in out


def test_kpi_metrics_schema(local_mode):
    out = json.loads(analysis_tools.get_kpi_metrics(FIXTURE_RUN))
    assert "error" not in out
    for key in (
        "step_time_ms",
        "duty_cycle_percent",
        "mxu_utilization_percent",
        "roofline_utilization",
        "device",
    ):
        assert key in out
    # The program-level-scalar warning must ride along.
    assert "caveat" in out["note"] or "roofline" in out["note"]


# ---------------------------------------------------------------------------
# Pre-existing summary tools keep working through the local client
# ---------------------------------------------------------------------------


def test_get_overview_local(local_mode):
    out = json.loads(get_overview_tool.get_overview(FIXTURE_RUN))
    assert "error" not in out
    assert "performance_summary" in out and "run_environment" in out


def test_get_top_hlo_ops_local(local_mode):
    out = json.loads(get_top_hlo_ops_tool.get_top_hlo_ops(FIXTURE_RUN))
    assert "error" not in out
    assert "top_by_time" in out
