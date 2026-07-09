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
    # golden: p50 ~72% during the kernel
    assert 65 < mxu["p50_util_pct"] < 80
    assert d["verdict"]["dominant_unit"] == "MXU"
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
