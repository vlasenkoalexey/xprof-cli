"""Parity guard: in-process conversion must match the HTTP xprof server.

Boots a real `xprof --logdir=<fixtures> --port=<random>` and compares tool
payloads fetched via OSSXprofClient (HTTP) against LocalXprofClient
(in-process converters) for the same run. This is the structural guarantee
that the XPROF_MODE=local migration did not change analysis semantics.

Skips cleanly when the `xprof` server binary is not installed.
"""

import json
import shutil
import socket
import subprocess
import time

import pytest

from tests.conftest import FIXTURE_LOGDIR, FIXTURE_RUN

pytest.importorskip("xprof.convert.raw_to_tool_data",
                    reason="xprof pip converters required for local mode")

XPROF_BIN = shutil.which("xprof")

pytestmark = pytest.mark.skipif(
    XPROF_BIN is None, reason="xprof server binary not on PATH"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def xprof_server():
    port = _free_port()
    proc = subprocess.Popen(
        [XPROF_BIN, "--logdir", FIXTURE_LOGDIR, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        import requests

        deadline = time.time() + 90
        url = f"http://127.0.0.1:{port}/data/plugin/profile/runs"
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"xprof server died on startup (rc={proc.returncode})")
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(1.0)
        else:
            pytest.fail("xprof server did not become ready within 90s")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def clients(xprof_server, monkeypatch):
    from xprof_mcp.internal import local_client, xprof_client

    monkeypatch.setenv("XPROF_LOGDIR", FIXTURE_LOGDIR)
    http = xprof_client.OSSXprofClient(base_url=xprof_server)
    local = local_client.LocalXprofClient()
    return http, local


def test_runs_parity(clients):
    http, local = clients
    assert sorted(http.get_runs()) == sorted(local.get_runs())


@pytest.mark.parametrize("tool", ["overview_page", "op_profile", "hlo_stats"])
def test_tool_payload_parity(clients, tool):
    http, local = clients
    http_data = json.loads(http.fetch(tool, FIXTURE_RUN, host="ALL_HOSTS"))
    local_data = json.loads(local.fetch(tool, FIXTURE_RUN, host="ALL_HOSTS"))

    if tool == "overview_page":
        # Compare the analysis substance; the server may append
        # environment-dependent presentation tables.
        http_sub = _overview_substance(http_data)
        local_sub = _overview_substance(local_data)
        assert local_sub == http_sub
    else:
        assert local_data == http_data


def _overview_substance(payload):
    """Extracts performance_summary / run_environment-bearing tables."""
    if isinstance(payload, dict):
        return payload
    out = []
    for table in payload:
        if isinstance(table, dict) and table.get("p"):
            out.append(table["p"])
    return out
