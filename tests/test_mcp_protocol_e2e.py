"""MCP-protocol integration test: drives the server over stdio and exercises
the kernel-profiling + LLO dump suites end-to-end against the fixtures.

Requires the `mcp` package (and tensorflow for the trace-side calls);
skips cleanly when either is missing.
"""

import json
import os
import sys

import pytest

from tests.conftest import FIXTURES

mcp_lib = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

try:
    from xprof_mcp.internal import xplane_tools
    _HAS_TF = xplane_tools._HAS_XPLANE_PROTO  # pylint: disable=protected-access
except ImportError:
    _HAS_TF = False

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "testrun"


async def _call(session, name, args):
    res = await session.call_tool(name, args)
    return json.loads(res.content[0].text)


@pytest.mark.skipif(not _HAS_TF, reason="tensorflow not installed")
@pytest.mark.anyio
async def test_kernel_profiling_and_llo_dumps_over_mcp(tmp_path):
    # the package dir may be hyphenated (`xprof-mcp`); alias it for -m import
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "xprof_mcp").symlink_to(_REPO_ROOT)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "xprof_mcp.server.xprof_mcp_server"],
        env={
            **os.environ,
            "PYTHONPATH": str(pkg_dir),
            "XPROF_LOGDIR": os.path.join(FIXTURES, "logdir"),
            "XLA_JF_DUMP_DIR": os.path.join(FIXTURES, "jf_dump"),
            "TF_CPP_MIN_LOG_LEVEL": "3",
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [t.name for t in (await session.list_tools()).tools]
            for tool in ("check_kernel_profiling", "get_llo_utilization",
                         "list_llo_programs", "get_llo_bundles",
                         "get_custom_call_mlir"):
                assert tool in names

            r = await _call(session, "check_kernel_profiling", {"run": RUN})
            assert r["kernel_profiling_active"] is True

            r = await _call(session, "get_llo_utilization",
                            {"run": RUN, "kernel": "matmul_optimized"})
            mxu = r["/device:TPU:0"]["units"]["MXU"]
            assert 65 < mxu["p50_util_pct"] < 80

            r = await _call(session, "get_llo_static_utilization",
                            {"program": "matmul_optimized.1"})
            hot = r["hot_ranges"][0]

            # the trace<->dump join: hot address range -> exact bundles
            r = await _call(session, "get_llo_bundles",
                            {"program": "matmul_optimized.1",
                             "address_range": "-".join(hot["address_range"]),
                             "limit": 2})
            assert r["total_matched"] == hot["length"]

            r = await _call(session, "get_custom_call_mlir",
                            {"mosaic_dump_dir": os.path.join(FIXTURES, "mosaic_dump"),
                             "max_chars": 300})
            assert r["source"] == "mosaic_dump"
