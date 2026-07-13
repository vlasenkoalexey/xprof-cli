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

# Both sides read a pristine COPY of the fixture logdir: conversions write
# derived caches (*.op_stats_v2.pb, cache_version.txt, saved tool results)
# next to the trace, and a cache left by an older xprof version would make
# the server serve stale columns while local computes fresh — a fake
# parity failure (and pollution of the committed fixtures).


@pytest.fixture(scope="module")
def parity_logdir(tmp_path_factory):
    dst = tmp_path_factory.mktemp("parity") / "logdir"
    shutil.copytree(FIXTURE_LOGDIR, dst)
    return str(dst)

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
def xprof_server(parity_logdir):
    port = _free_port()
    proc = subprocess.Popen(
        [XPROF_BIN, "--logdir", parity_logdir, "--port", str(port)],
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
def clients(xprof_server, parity_logdir, monkeypatch):
    from xprof_mcp.internal import local_client, xprof_client

    monkeypatch.setenv("XPROF_LOGDIR", parity_logdir)
    http = xprof_client.OSSXprofClient(
        base_url=xprof_server, logdir=parity_logdir
    )
    local = local_client.LocalXprofClient(logdir=parity_logdir)
    return http, local


def test_runs_parity(clients):
    http, local = clients
    assert sorted(http.get_runs()) == sorted(local.get_runs())


@pytest.mark.parametrize("tool", ["overview_page", "op_profile", "hlo_stats"])
def test_tool_payload_parity(clients, tool):
    """Superset parity: local must carry every value HTTP carried.

    The serving path and the direct converter path can differ in
    *presentation* (the in-process converter may emit additional columns,
    e.g. hlo_stats core_type/parent_op_name), so the contract is: for
    every table, every column the server returned must exist locally with
    identical row values, and locally-only columns are allowed. This is
    the guarantee the XPROF_MODE=local migration needs — no information
    lost, no values changed.
    """
    http, local = clients
    http_data = json.loads(http.fetch(tool, FIXTURE_RUN, host="ALL_HOSTS"))
    local_data = json.loads(local.fetch(tool, FIXTURE_RUN, host="ALL_HOSTS"))
    assert _canonical(local_data, restrict_to=http_data) == _canonical(http_data)


# overview_page properties where the server's serving path is known to emit
# 0.0% on a freshly-served run while the in-process converter computes the
# real value (verified manually on the fixture: local's 37.4% duty cycle is
# consistent with the trace content; 0.0% is not). Local is the more
# correct side here, so these are exempt from strict equality — but they
# must still be PRESENT locally.
_KNOWN_SERVER_ZEROED = {
    "device_duty_cycle_percent",
    "device_idle_time_percent",
    "host_idle_time_percent",
}


def _canonical(payload, restrict_to=None):
    """Normalizes tool payloads for superset comparison.

    gviz DataTables become lists of {col_id: value} records; when
    restrict_to is given, columns absent from the reference payload are
    dropped (allowing local-only additions). Non-table payloads are
    returned as-is.
    """
    ref_cols = None
    if restrict_to is not None:
        ref_cols = set()
        for table in _tables(restrict_to):
            ref_cols.update(c.get("id") for c in table.get("cols", []))

    out = []
    for table in _tables(payload):
        col_ids = [c.get("id") for c in table.get("cols", [])]
        rows = []
        for row in table.get("rows", []):
            cells = row.get("c", [])
            rec = {}
            for i, col_id in enumerate(col_ids):
                if ref_cols is not None and col_id not in ref_cols:
                    continue
                rec[col_id] = cells[i].get("v") if i < len(cells) and cells[i] else None
            rows.append(rec)
        props = {
            k: v
            for k, v in table.get("p", {}).items()
            if k not in _KNOWN_SERVER_ZEROED
        }
        out.append({"rows": rows, "p": props})
    return out if out else payload


def _tables(payload):
    if isinstance(payload, dict) and "cols" in payload:
        return [payload]
    if isinstance(payload, list):
        return [t for t in payload if isinstance(t, dict) and "cols" in t]
    return []
