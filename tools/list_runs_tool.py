"""Tool for listing available XProf runs from the local server."""

import json
import logging

from xprof_mcp.internal import xprof_client


def list_runs() -> str:
    """Lists all profiling runs available on the connected XProf server.

    **START HERE** if you don't know the run name. This replaces the
    internal `find_xprof_session` tool for OSS use.

    The run name returned here is what you pass as the `run` argument to
    all other tools (e.g. `get_overview`, `get_top_hlo_ops`).

    The xprof server must be running locally. Start it with:
      xprof --logdir=<path_to_profiles> --port=8791

    Returns:
        A JSON-formatted dict with the list of run names and server URL.
    """
    client = xprof_client.get_client()
    try:
        runs = client.get_runs()

        return json.dumps(
            {
                "server": client.base_url,
                "logdir": client.logdir or "(not set — set XPROF_LOGDIR for disk-based tools)",
                "runs": runs,
                "count": len(runs),
                "tip": (
                    "Pass a run name to get_overview, get_top_hlo_ops, "
                    "list_hlo_modules, etc."
                ),
            },
            indent=2,
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error listing runs")
        return json.dumps(
            {
                "error": str(e),
                "tip": (
                    f"Make sure the xprof server is running at {client.base_url}. "
                    "Start it with: xprof --logdir=<path> --port=8791"
                ),
            },
            indent=2,
        )
