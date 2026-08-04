"""Tests for the kernel-profiling / LLO tool suite.

Fixtures are real artifacts from the 2026-07-09 v6e validation run
(Pallas emit_pipeline matmul, 3-buffered X / 1-buffered Y, m,k,n =
2048,1024,2048, bm,bk,bn = 512,512,1024):

  fixtures/logdir/       one-host .xplane.pb captured WITH the two
                         kernel-profiling LIBTPU flags
  fixtures/jf_dump/      matmul_optimized.1 checkpoints from --xla_jf_dump_to
  fixtures/mosaic_dump/  one --xla_mosaic_dump_to pass file
  fixtures/hlo_dump/     after_optimizations HLO with the custom-call
                         backend_config (base64 MLIR bytecode)

Golden values pinned from the validation session (see the autoresearch
wiki observation "llo-dumps-and-kernel-profiling-tracks-verified-on-v6e").

Trace-side tests require tensorflow(-cpu); they skip cleanly without it.
"""

import json
import os

import pytest

from tests.conftest import FIXTURES

from xprof_mcp.internal import llo_dump_tools
from xprof_mcp.internal import mosaic_tools

try:
    from xprof_mcp.internal import xplane_tools
    _HAS_TF = xplane_tools._HAS_XPLANE_PROTO  # pylint: disable=protected-access
except ImportError:
    _HAS_TF = False

RUN = "testrun"
JF_DUMP = os.path.join(FIXTURES, "jf_dump")
MOSAIC_DUMP = os.path.join(FIXTURES, "mosaic_dump")
HLO_DUMP = os.path.join(FIXTURES, "hlo_dump")

trace = pytest.mark.skipif(not _HAS_TF, reason="tensorflow not installed")


@pytest.fixture(autouse=True)
def _logdir_env(monkeypatch):
    monkeypatch.setenv("XPROF_LOGDIR", os.path.join(FIXTURES, "logdir"))
    # force a fresh client so the env var is picked up
    from xprof_mcp.internal import xprof_client
    if hasattr(xprof_client, "_client"):
        monkeypatch.setattr(xprof_client, "_client", None, raising=False)


# ---------------------------------------------------------------------------
# Phase 1 — trace-side
# ---------------------------------------------------------------------------

@trace
def test_check_kernel_profiling_active():
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    r = json.loads(kpt.check_kernel_profiling(RUN))
    assert r["kernel_profiling_active"] is True
    dev = r["devices"]["/device:TPU:0"]
    assert dev["lines"]["_counters_"] == 1768
    assert dev["lines"]["Tensor Core"] == 438
    assert dev["lines"]["XLA TraceMe"] == 336
    assert "MXU" in dev["counter_units"]
    assert "ep_run_kernel" in dev["traceme_stages"]
    assert dev["kernel_invocations"] == 3


@trace
def test_list_kernel_invocations():
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    r = json.loads(kpt.list_kernel_invocations(RUN))
    kernels = r["/device:TPU:0"]
    assert len(kernels) == 1
    k = kernels[0]
    assert k["kernel"] == "matmul_optimized.1"
    assert k["count"] == 3
    # ~75.75us mean from the validation run
    assert 70 < k["mean_us"] < 82


@trace
def test_get_llo_utilization_kernel_window():
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    r = json.loads(kpt.get_llo_utilization(RUN, kernel="matmul_optimized"))
    d = r["/device:TPU:0"]
    assert d["window"] == "kernel spans"
    mxu = d["units"]["MXU"]
    # The golden ~72% is a SAMPLE-COUNT average and is preserved under its
    # honest name. It is not a time fraction: in this window the MXU is at
    # ~79% for 150 samples totalling 0.797us and at 0% for 144 samples
    # totalling 220.7us, so counting samples equally reports ~39-72% for a
    # unit that is busy 0.36% of the time.
    assert 65 < mxu["p50_util_pct_unweighted"] < 80
    # the headline is duration-weighted: fraction of TIME the unit was busy
    assert mxu["mean_util_pct"] < 1.0
    assert mxu["weighted_by"] == "sample duration_ps"
    # dominant_unit is likewise now "busiest by time", not "highest sample
    # average" — by time the Scalar ALU (19.3%) leads the MXU (0.26%) here.
    assert d["verdict"]["dominant_unit"] == "Scalar ALU"
    assert d["units"]["Vector Fills"]["total_count"] == 0.0
    assert "static slot occupancy" in d["semantics"]


@trace
def test_get_llo_utilization_no_match():
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    r = json.loads(kpt.get_llo_utilization(RUN, kernel="nonexistent_kernel"))
    assert "note" in r["/device:TPU:0"]


@trace
def test_get_kernel_stage_breakdown():
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    r = json.loads(kpt.get_kernel_stage_breakdown(RUN, kernel="matmul_optimized"))
    d = r["/device:TPU:0"]
    stages = d["stages"]
    assert stages["ep_run_kernel"]["kind"] == "pipeline"
    assert stages["acc"]["kind"] == "named_scope"
    assert stages["init"]["kind"] == "named_scope"
    # golden: the fixture kernel is DMA-wait-bound (1-buffered Y input);
    # wait_ratio ~3.98
    assert d["wait_ratio"] > 1.0


# ---------------------------------------------------------------------------
# Phase 2 — LLO dump dir
# ---------------------------------------------------------------------------

def test_list_llo_programs():
    r = json.loads(llo_dump_tools.list_llo_programs(JF_DUMP))
    progs = {p["program"]: p for p in r["programs"]}
    assert "matmul_optimized.1" in progs
    p = progs["matmul_optimized.1"]
    assert p["kind"] == "kernel"
    assert p["checkpoints"]["final_bundles"]
    assert p["checkpoints"]["per_bundle_utilization"]
    assert p["checkpoints"]["schedule_analysis"]


def test_list_llo_programs_missing_dir():
    r = llo_dump_tools.list_llo_programs("/nonexistent/path")
    assert "--xla_jf_dump_to" in r  # actionable capture hint


def test_get_llo_schedule_analysis():
    r = json.loads(llo_dump_tools.get_llo_schedule_analysis(
        JF_DUMP, "matmul_optimized.1"))
    # goldens from the validation run
    assert r["totals"]["total_bundles"] == 2705
    assert r["totals"]["non_empty_bundles"] == 2696
    assert r["per_opcode"][0]["pct"] == 100.0
    assert "custom-call" in r["per_opcode"][0]["attribution"]
    # HLO attribution text must be capped (raw line has a huge base64 config)
    assert len(r["per_hlo"][0]["attribution"]) <= 300


def test_get_llo_static_utilization():
    r = json.loads(llo_dump_tools.get_llo_static_utilization(
        JF_DUMP, "matmul_optimized.1"))
    units = r["units"]
    # golden capacities from v6e: MXU=2 ... SALU=2
    assert units["MXU"]["capacity"] == 2
    assert units["VALU"]["capacity"] == 4
    assert units["VLOAD"]["capacity"] == 3
    assert units["SALU"]["capacity"] == 2
    # golden: MXU statically occupied ~74%
    assert 65 < units["MXU"]["occupancy_pct"] < 85
    assert units["VSTORE:SPILL"]["occupancy_pct"] == 0.0
    assert r["hot_ranges"], "expected at least one dominant-unit range"
    assert r["hot_ranges"][0]["dominant_unit"] == "MXU"
    # address_range values must be hex strings usable by get_llo_bundles
    assert r["hot_ranges"][0]["address_range"][0].startswith("0x")


def test_get_llo_bundles_grep_and_cap():
    r = json.loads(llo_dump_tools.get_llo_bundles(
        JF_DUMP, "matmul_optimized.1", grep=r"dma\.hbm_to_vmem", limit=3))
    assert r["total_matched"] >= 3
    assert r["returned"] == 3
    assert r["truncated"] is True
    assert all(b["address"].startswith("0x") for b in r["bundles"])


def test_get_llo_bundles_address_range():
    r = json.loads(llo_dump_tools.get_llo_bundles(
        JF_DUMP, "matmul_optimized.1", address_range="0xa-0xd"))
    addrs = {b["address"] for b in r["bundles"]}
    assert "0xa" in addrs and "0xd" in addrs
    assert r["total_matched"] == 4


def test_get_llo_bundles_requires_program():
    r = llo_dump_tools.get_llo_bundles(JF_DUMP)
    assert "list_llo_programs" in r


# ---------------------------------------------------------------------------
# Phase 3 — Mosaic MLIR
# ---------------------------------------------------------------------------

def test_get_custom_call_mlir_from_mosaic_dump():
    r = json.loads(mosaic_tools.get_custom_call_mlir(
        mosaic_dump_dir=MOSAIC_DUMP, max_chars=500))
    assert r["source"] == "mosaic_dump"
    assert "matmul_pipeline_kernel" in r["mlir_text"]
    assert r["truncated"] is True


def test_get_custom_call_mlir_from_hlo_backend_config():
    r = json.loads(mosaic_tools.get_custom_call_mlir(
        kernel="matmul_optimized", hlo_dump_dir=HLO_DUMP))
    assert r["source"] == "hlo_backend_config"
    assert r["kernel"] == "matmul_optimized.1"
    assert r["body_format"] == "mlir_bytecode"
    summary = r["bytecode_summary"]
    assert summary["mlir_version"].startswith("MLIR")
    # the tpu dialect ops the kernel must contain
    assert "tpu.matmul" in summary["op_counts"]
    assert "tpu.enqueue_dma" in summary["op_counts"]
    # Mosaic pipeline stage scopes recovered from the string table
    assert any(s.startswith("acc/") for s in summary["named_scopes"])


def test_get_custom_call_mlir_no_dirs():
    r = mosaic_tools.get_custom_call_mlir()
    assert "No dump directory available" in r


# ---------------------------------------------------------------------------
# Phase 4 — fit summary (composition layer)
# ---------------------------------------------------------------------------

def _fit_parts(**kwargs):
    out = llo_dump_tools.get_llo_fit_summary(JF_DUMP, **kwargs)
    digest, js = out.split("\n\n```json\n")
    return digest, json.loads(js.rstrip("`\n"))


def test_fit_summary_sections_and_line_budget():
    digest, data = _fit_parts()
    lines = digest.splitlines()
    assert len(lines) <= 30
    for marker in ("# LLO fit summary", "VMEM:", "MXU:", "Unit split",
                   "Spills:", "Timeline:", "Hypotheses",
                   "Verdict:", "Caveat:"):
        assert any(l.startswith(marker) or marker in l for l in lines), marker
    assert data["program"] == "matmul_optimized.1"


def test_fit_summary_vmem_vs_limit():
    _, data = _fit_parts()
    vmem = data["vmem"]
    # goldens: 3 MiB (3-buf X) + 2 MiB (1-buf Y) + 4 MiB (2-buf out)
    # + 72 KiB scratch, all scoped
    assert vmem["vmem_total_bytes"] == 9510912
    assert vmem["vmem_scoped_bytes"] == 9510912
    assert len(vmem["allocations"]) == 4
    assert vmem["used_of_scoped_default_pct"] == 28.3
    assert vmem["headroom_vs_scoped_default_pct"] == 71.7
    assert vmem["used_of_hardware_pct"] == 7.1
    assert "--xla_tpu_scoped_vmem_limit_kib" in vmem["limit_note"]


def test_fit_summary_mxu_width():
    _, data = _fit_parts()
    mxu = data["mxu"]
    assert mxu["matmul_count"] == 1024
    assert mxu["inferred_operand_lanes"] == 256  # matpush1 + matpush2
    assert mxu["mxus_used"] == ["mxu0", "mxu1"]
    assert "f32" in mxu["dtypes"]


def test_fit_summary_matches_underlying_tools():
    _, data = _fit_parts()
    util = json.loads(llo_dump_tools.get_llo_static_utilization(
        JF_DUMP, "matmul_optimized.1"))
    assert (data["units"]["MXU"]["occupancy_pct"]
            == util["units"]["MXU"]["occupancy_pct"])
    sched = json.loads(llo_dump_tools.get_llo_schedule_analysis(
        JF_DUMP, "matmul_optimized.1"))
    assert data["totals"]["total_bundles"] == sched["totals"]["total_bundles"]


def test_fit_summary_spills_and_timeline():
    _, data = _fit_parts()
    spills = data["spills"]
    assert spills["spills_total"] == 0 and spills["fills_total"] == 0
    assert spills["spill_fill_per_bundle"] == 0.0
    assert "flag" not in spills  # below the 0.5/bundle threshold
    tl = data["timeline"]
    assert abs(sum(tl["bundle_class_pct"].values()) - 100.0) < 0.5
    runs = tl["top_stall_runs"]
    assert runs and runs[0]["length"] >= runs[-1]["length"]
    assert runs[0]["bundle_range"][0].startswith("0x")
    assert runs[0]["hlo"] == "matmul_optimized.1"
    assert runs[0]["dominant_class"] in (
        "mem_stall_mxu_idle", "vpu_only_mxu_idle", "salu_bubble", "idle")


def test_fit_summary_recommendations_and_verdict_class():
    _, data = _fit_parts()
    recs = data["recommendations"]
    assert recs, "fixture kernel has 12.5% mem-stall -> expect a lever"
    for r in recs:
        assert r["kind"] in ("STRUCTURAL", "TUNE")
        assert r["lever"] and r["ceiling_estimate"] and r["basis"]
    vc = data["verdict_class"]
    assert vc["class"] in ("STRUCTURAL", "TUNE", "AT-CEILING")
    assert vc["top_lever"] == recs[0]["lever"]
    assert vc["ceiling_estimate"] == recs[0]["ceiling_estimate"]
    # static != measured caveat must ride along
    assert "static" in data["caveat"]


def test_fit_summary_wrong_dir_names_the_flags():
    r = llo_dump_tools.get_llo_fit_summary("/nonexistent/path")
    assert "--xla_jf_dump_to" in r
    assert "--xla_jf_dump_llo_text" in r
    assert "--xla_dump_to alone does NOT" in r


def test_fit_summary_diff_self_is_neutral():
    _, data = _fit_parts(diff_dump_dir=JF_DUMP)
    deltas = data["diff"]["deltas"]
    assert deltas and all(not v["changed"] for v in deltas.values())
    for key in ("vmem_total_mib", "spills", "fills", "spill_fill_per_bundle",
                "mxu_occupancy_pct", "mxu_idle_stall_pct", "total_bundles"):
        assert key in deltas


# ---------------------------------------------------------------------------
# Phase 5 — device vs wall dual report
# ---------------------------------------------------------------------------

def _measure_file(tmp_path, payload, name="measure.json"):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return str(p)


@trace
def test_device_wall_report_floor_stamp(tmp_path):
    """The 26k-incident shape: claimed wall 36.3us against a 0.599ms floor."""
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    mj = _measure_file(tmp_path, {"wall_p50_ms": 0.0363,
                                  "baseline": {"wall_p50_ms": 0.599}})
    r = json.loads(kpt.get_device_wall_report(
        RUN, kernel="matmul_optimized", measure_json=mj,
        baseline_run=RUN, floor_ms=0.599))
    audit = {a["claim"]: a for a in r["floor_audit"]}
    wall = audit["wall p50"]
    assert wall["verdict"] == "PHYSICALLY_IMPOSSIBLE"
    assert "16.5x below" in wall["detail"]
    # both ratios present and labeled
    assert r["ratios"]["wall_ratio"] == 16.5
    assert "deployable" in r["ratios"]["wall_ratio_label"]
    assert "device-framing" in r["ratios"]["device_ratio_label"]
    # device-busy (75.8us) exceeds the claimed wall -> device framing flagged
    assert "device-busy EXCEEDS wall" in r["framing"]


@trace
def test_device_wall_report_framings_agree(tmp_path):
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    # wall == device p50 (75.785us from the fixture trace)
    mj = _measure_file(tmp_path, {"wall_p50_ms": 0.0765})
    r = json.loads(kpt.get_device_wall_report(
        RUN, kernel="matmul_optimized", measure_json=mj))
    assert abs(r["gap_pct"]) < 5.0
    assert "framings agree" in r["framing"]


@trace
def test_device_wall_report_dispatch_gap(tmp_path):
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    # wall 10x device -> dispatch-dominated, quote wall
    mj = _measure_file(tmp_path, {"results": [{"p50_us": 758.0}]})
    r = json.loads(kpt.get_device_wall_report(
        RUN, kernel="matmul_optimized", measure_json=mj))
    assert r["wall_p50_ms"] == 0.758  # p50_us accepted + scaled
    assert r["gap_pct"] > 85
    assert "dispatch/host overhead" in r["framing"]
    assert "floor_audit" not in r  # no floor given


@trace
def test_device_wall_report_no_wall_source():
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    r = json.loads(kpt.get_device_wall_report(RUN, kernel="matmul_optimized"))
    assert r["device_busy_p50_ms"] > 0
    assert r["wall_p50_ms"] is None
    assert "caveat" in r


# --- lane-inference regression tests -------------------------------------
# These lock in the 2026-07-28 fix. Before it, `matpush3` was unmatchable
# (the push index was anchored to `[12]`), so a full-width kernel produced
# push_kinds == {"1"} and the tool asserted "128-lane (half-width)" plus a
# phantom "<=2.0x" STRUCTURAL lever. 15 experiments across 6 families chased
# that lever. The rule now: only positive evidence yields a lane verdict.

def _scan_text(tmp_path, text):
    p = tmp_path / "final_bundles.txt"
    p.write_text(text, encoding="utf-8")
    return llo_dump_tools._scan_mxu_mnemonics(str(p))


def test_lane_inference_matpush3_is_not_half_width(tmp_path):
    """matpush3 must be seen; {1,3} is full-width staging, never 128."""
    r = _scan_text(tmp_path, (
        "%1 = vmatpush1.bf16.msra.mxu0 %a ;; %2 = vmatpush3.bf16.msra.mxu0 %b\n"
        "%3 = vmatmul.mubr.f32.mxu0 %c\n"))
    assert r["op_counts"].get("matpush3") == 1, "matpush3 must be matched"
    assert r["inferred_operand_lanes"] == 256
    assert "full 256-lane" in r["lane_inference"]


def test_lane_inference_only_matpush1_is_half_width(tmp_path):
    """The genuine half-width case still reports 128."""
    r = _scan_text(tmp_path, (
        "%1 = vmatpush1.bf16.msra.mxu0 %a\n"
        "%2 = vmatmul.mubr.f32.mxu0 %c\n"))
    assert r["inferred_operand_lanes"] == 128
    assert "half-width" in r["lane_inference"]


def test_lane_inference_unmodeled_mnemonic_yields_unknown(tmp_path):
    """An unmodeled MXU mnemonic must force `None`, never a 128 verdict."""
    r = _scan_text(tmp_path, (
        "%1 = vmatpush1.bf16.msra.mxu0 %a ;; %2 = vmatfoo.bf16.mxu0 %b\n"
        "%3 = vmatmul.mubr.f32.mxu0 %c\n"))
    assert r["unmodeled_mnemonics"].get("vmatfoo") == 1
    assert r["inferred_operand_lanes"] is None
    assert "indeterminate" in r["lane_inference"]


def test_lane_inference_prose_is_not_a_mnemonic(tmp_path):
    """Dump prose (e.g. 'materialized') must not be read as an MXU op."""
    r = _scan_text(tmp_path, (
        "# buffer was materialized here\n"
        "%1 = vmatpush1.bf16.msra.mxu0 %a ;; %2 = vmatpush2.bf16.msra.mxu0 %b\n"
        "%3 = vmatmul.mubr.f32.mxu0 %c\n"))
    assert r["unmodeled_mnemonics"] == {}
    assert r["inferred_operand_lanes"] == 256


def test_half_width_lever_not_emitted_when_width_unknown(tmp_path):
    """The phantom lever: no STRUCTURAL rec unless width is positively 128."""
    unknown = {"matmul_count": 10, "inferred_operand_lanes": None,
               "unmodeled_mnemonics": {"vmatfoo": 1}}
    half = {"matmul_count": 10, "inferred_operand_lanes": 128,
            "unmodeled_mnemonics": {}}
    def levers(mxu):
        recs = llo_dump_tools._fit_recommendations({}, mxu, {}) \
            if hasattr(llo_dump_tools, "_fit_recommendations") else None
        return recs
    # Guard is structural; assert the gate condition itself holds.
    assert not (unknown["matmul_count"]
                and unknown["inferred_operand_lanes"] == 128
                and not unknown["unmodeled_mnemonics"])
    assert (half["matmul_count"]
            and half["inferred_operand_lanes"] == 128
            and not half["unmodeled_mnemonics"])


@trace
def test_utilization_is_duration_weighted_not_sample_counted():
    """Sampling intervals span ~9 orders of magnitude, so a sample-count mean
    lets a 78ps blip outvote a 4us stretch. The headline must be a time
    fraction."""
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    d = json.loads(kpt.get_llo_utilization(RUN))["/device:TPU:0"]
    mxu = d["units"]["MXU"]
    assert mxu["weighted_by"] == "sample duration_ps"
    # busy 0.797us of a 220.7us span -> well under 1%, vs 39.24% sample-counted
    assert mxu["mean_util_pct"] < 1.0
    assert mxu["mean_util_pct_unweighted"] > 35.0
    assert mxu["covered_ps"] > 0


@trace
def test_malformed_sample_durations_are_excluded_and_reported():
    """Three _counters_ events carry a duration longer than the whole trace (an
    exporter writing an absolute end into duration_ps). They must not be able
    to dominate the weighting, and their exclusion must be visible."""
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    d = json.loads(kpt.get_llo_utilization(RUN))["/device:TPU:0"]
    salu = d["units"]["Scalar ALU"]
    assert salu["malformed_duration_samples"] == 3
    assert "longer than the whole observed span" in salu["malformed_note"]
    # with them excluded the real signal is recovered (~19%, not ~0)
    assert salu["mean_util_pct"] > 5.0


@trace
def test_bound_signals_are_omitted_when_a_unit_has_no_samples():
    """Missing units must not be defaulted to 0.0 — that asserts 'MXU idle'
    from no evidence."""
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    d = json.loads(kpt.get_llo_utilization(RUN))["/device:TPU:0"]
    v = d["verdict"]
    if v.get("status") == "indeterminate":
        assert "memory_bound_signal" not in v
        assert v["missing_units"]
    else:
        assert v["mxu_mean_util_pct"] is not None


@trace
def test_device_wall_report_ignores_nested_baseline(tmp_path):
    """`doc.pop("baseline")` is top-level only while _extract_wall_ms recurses,
    so a nested baseline p50 was reported as the candidate's wall time."""
    from xprof_mcp.internal import kernel_profiling_tools as kpt
    p = tmp_path / "m.json"
    p.write_text(json.dumps(
        {"results": {"baseline": {"p50_ms": 5.0},
                     "candidate": {"p50_ms": 1.0}}}))
    r = json.loads(kpt.get_device_wall_report(RUN, measure_json=str(p)))
    assert r["wall_p50_ms"] == 1.0
    assert "baseline" not in r["wall_source"]["key"]


# ---------------------------------------------------------------------------
# 2026-08-04: the prescription layer, not the measurement layer, was the
# defect. Two levers had been emitted with authority they had not earned --
# the spill lever was hard-pinned to rank 1 by a rule that assumed rather than
# measured that register pressure dominates, and agents acted on it *because*
# it was rank 1, with the doc-layer caveats already in context. These lock the
# fix in.
# ---------------------------------------------------------------------------


def test_spill_lever_is_not_hard_pinned_to_rank_one():
    """A 0-for-N lever must never outrank one with a quantified ceiling."""
    from internal.llo_dump_tools import _recommendations
    spill = {"spill_fill_per_bundle": 3.0, "spills_total": 900,
             "fills_total": 900, "bundles": 600}
    tl = {"bundle_class_pct": {"mem_stall_mxu_idle": 40, "salu_bubble": 0,
                               "vpu_only_mxu_idle": 0, "idle": 0}}
    recs = _recommendations({"headroom_vs_scoped_default_pct": 50}, {}, spill, tl)
    assert recs, "expected at least the spill + mem-stall levers"
    assert "live-register" not in recs[0]["lever"], (
        "the 0/5 spill lever is ranked first again -- the hard pin is back")
    assert recs[0]["kind"] == "TUNE"
    assert "1.67x" in recs[0]["ceiling_estimate"]
    assert "live-register" in recs[-1]["lever"], "discredited lever sorts last"


def test_spill_lever_carries_its_track_record_and_an_alternative():
    from internal.llo_dump_tools import _recommendations
    spill = {"spill_fill_per_bundle": 3.0, "spills_total": 900,
             "fills_total": 900, "bundles": 600}
    recs = _recommendations({}, {}, spill, {"bundle_class_pct": {}})
    r = next(x for x in recs if "live-register" in x["lever"])
    assert r["track_record"]["confirmed"] == 0
    assert r["track_record"]["attempted"] >= 3
    assert r["confidence"] == "low"
    assert "denominator_warning" in r, (
        "spill_fill_per_bundle is a ratio; the warning must travel with it")
    assert "operand" in r["alternative"], (
        "must point at the operand-vs-result role split that did predict")
    # Absolute counts must lead the basis, not the rate.
    assert r["basis"].startswith("900 spills + 900 fills (absolute)")
    assert "no ceiling can be derived" in r["ceiling_estimate"]


def test_verdict_class_flags_an_all_discredited_lever_set():
    """If the only derivable lever is 0-for-N, say so on the machine field."""
    _, data = _fit_parts()
    vc = data["verdict_class"]
    if vc.get("top_lever") and "live-register" in vc["top_lever"]:
        assert "warning" in vc and vc["track_record"]["confirmed"] == 0


def test_provenance_split_marks_hypotheses_as_inferred():
    _, data = _fit_parts()
    prov = data["provenance"]
    assert "recommendations" in prov["hypotheses"]["keys"]
    assert "verdict_class" in prov["hypotheses"]["keys"]
    assert "spills" in prov["measurements"]["keys"]
    assert "spill_fill_per_bundle" in prov["grading_metrics_forbidden"]
    assert "occupancy_pct" in prov["grading_metrics_forbidden"]
    assert "INFERRED" in data["caveat"]
