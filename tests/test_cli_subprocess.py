"""End-to-end CLI smoke tests: real subprocess, real fixture, no server.

Each invocation pays the tensorflow import (~2s), so this file keeps the
subprocess count low; per-tool coverage lives in test_analysis_tools.py.
"""

import json
import os
import subprocess
import sys

import pytest

from tests.conftest import FIXTURE_LOGDIR, FIXTURE_RUN, _REPO_ROOT

pytest.importorskip("xprof.convert.raw_to_tool_data",
                    reason="xprof pip converters required for local mode")
pytest.importorskip("fire")


@pytest.fixture(scope="module")
def cli_env(tmp_path_factory):
    """Env whose PYTHONPATH exposes the repo as the `xprof_mcp` package
    regardless of the checkout directory name."""
    pkg_parent = tmp_path_factory.mktemp("pkg")
    link = pkg_parent / "xprof_mcp"
    link.symlink_to(_REPO_ROOT)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pkg_parent)
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env.pop("XPROF_MODE", None)  # CLI must default to local on its own
    return env


def _run_cli(env, *args):
    return subprocess.run(
        [sys.executable, "-m", "xprof_mcp.cli.main", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_get_overview_success_and_cache_hit(cli_env):
    args = (
        "get_overview",
        f"--logdir={FIXTURE_LOGDIR}",
        f"--run={FIXTURE_RUN}",
        "--bypass_cache=True",  # first call: deterministic cold path
    )
    cold = _run_cli(cli_env, *args)
    assert cold.returncode == 0, cold.stderr[-800:]
    parsed = json.loads(cold.stdout)
    assert parsed["run"] == FIXTURE_RUN
    assert "performance_summary" in parsed

    warm = _run_cli(
        cli_env,
        "get_overview",
        f"--logdir={FIXTURE_LOGDIR}",
        f"--run={FIXTURE_RUN}",
    )
    assert warm.returncode == 0
    assert json.loads(warm.stdout) == parsed, "cache must return same payload"


def test_roofline_caveats_reach_stdout(cli_env):
    proc = _run_cli(
        cli_env,
        "get_roofline_model",
        f"--logdir={FIXTURE_LOGDIR}",
        f"--run={FIXTURE_RUN}",
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    out = json.loads(proc.stdout)
    assert "caveats" in out and len(out["caveats"]) >= 5


def test_unknown_run_exits_nonzero(cli_env):
    proc = _run_cli(
        cli_env,
        "get_overview",
        f"--logdir={FIXTURE_LOGDIR}",
        "--run=definitely-not-a-run",
    )
    assert proc.returncode == 1
    # Error body is machine-readable JSON (stdout for tool-level errors).
    parsed = json.loads(proc.stdout or proc.stderr)
    assert "error" in parsed


def test_llo_fit_summary_cli_matches_direct_call(cli_env):
    """CLI frontend parity: subprocess output == direct registry call."""
    from tests.conftest import FIXTURES
    from xprof_mcp.internal import llo_dump_tools

    jf_dump = os.path.join(FIXTURES, "jf_dump")
    proc = _run_cli(
        cli_env,
        "get_llo_fit_summary",
        f"--dump_dir={jf_dump}",
        "--bypass_cache=True",
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    direct = llo_dump_tools.get_llo_fit_summary(jf_dump)
    assert proc.stdout.rstrip("\n") == direct.rstrip("\n")
    # digest stays within the context budget on the CLI too
    digest = proc.stdout.split("\n\n```json")[0]
    assert len(digest.splitlines()) <= 30
