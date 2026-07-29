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


# --- regression: per-annotation arg coercion ---------------------------------
# The wrapper used to stringify EVERY int/float arg so that a run named
# "12345" survived fire's numeric parsing. That also handed the string "5" to
# int-annotated parameters, so `get_llo_fit_summary --top_stalls 5` died on
# `runs[:top_stalls]` with "slice indices must be integers".

def test_int_annotated_params_reach_the_tool_as_int(monkeypatch):
    # NB: cli.main does `from xprof_mcp import tool_registry`, so the patch
    # must target THAT module object — patching a bare `import tool_registry`
    # hits a different instance, leaves the cached branch live, and _run_salt
    # then builds the xprof_client singleton with no logdir, breaking every
    # later test that expects a fixture-bound client.
    from xprof_mcp import tool_registry
    from cli.main import _make_command
    monkeypatch.setattr(tool_registry, "UNCACHED_TOOLS",
                        frozenset(tool_registry.UNCACHED_TOOLS | {"tool"}))

    seen = {}

    def tool(run: str = "", top_stalls: int = 5, ratio: float = 1.0):
        seen.update(run=run, top_stalls=top_stalls, ratio=ratio)
        return None

    cmd = _make_command("tool", tool)
    cmd(run=12345, top_stalls=5, ratio=2, bypass_cache=True)

    # numeric-looking NAME must still be re-stringified ...
    assert seen["run"] == "12345" and isinstance(seen["run"], str)
    # ... while genuinely numeric params keep their declared type
    assert seen["top_stalls"] == 5 and isinstance(seen["top_stalls"], int)
    assert seen["ratio"] == 2.0 and isinstance(seen["ratio"], float)


def test_int_coercion_applies_to_positional_args(monkeypatch):
    # NB: cli.main does `from xprof_mcp import tool_registry`, so the patch
    # must target THAT module object — patching a bare `import tool_registry`
    # hits a different instance, leaves the cached branch live, and _run_salt
    # then builds the xprof_client singleton with no logdir, breaking every
    # later test that expects a fixture-bound client.
    from xprof_mcp import tool_registry
    from cli.main import _make_command
    monkeypatch.setattr(tool_registry, "UNCACHED_TOOLS",
                        frozenset(tool_registry.UNCACHED_TOOLS | {"tool"}))

    seen = {}

    def tool(run: str = "", top_stalls: int = 5):
        seen.update(run=run, top_stalls=top_stalls)
        return None

    _make_command("tool", tool)(12345, 5, bypass_cache=True)
    assert isinstance(seen["run"], str) and isinstance(seen["top_stalls"], int)


def test_unannotated_params_keep_legacy_stringify(monkeypatch):
    # NB: cli.main does `from xprof_mcp import tool_registry`, so the patch
    # must target THAT module object — patching a bare `import tool_registry`
    # hits a different instance, leaves the cached branch live, and _run_salt
    # then builds the xprof_client singleton with no logdir, breaking every
    # later test that expects a fixture-bound client.
    from xprof_mcp import tool_registry
    from cli.main import _make_command
    monkeypatch.setattr(tool_registry, "UNCACHED_TOOLS",
                        frozenset(tool_registry.UNCACHED_TOOLS | {"tool"}))

    seen = {}

    def tool(thing=""):
        seen["thing"] = thing
        return None

    _make_command("tool", tool)(thing=7, bypass_cache=True)
    assert seen["thing"] == "7"
