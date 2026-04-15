"""Tool for listing available XProf runs from the local server."""

import json
import logging

from xprof_mcp.internal import xprof_client


def list_runs(session_path: str = "", run_path: str = "") -> str:
    """Lists all profiling runs available on the connected XProf server.

    **START HERE** if you don't know the run name. This replaces the
    internal `find_xprof_session` tool for OSS use.

    The run name returned here is what you pass as the `run` argument to
    all other tools (e.g. `get_overview`, `get_top_hlo_ops`).

    The xprof server must be running locally. Start it with:
      xprof --logdir=<path_to_profiles> --port=8791

    Or point the MCP server at a specific path by passing URL params:
      session_path: load a single session directory directly.
      run_path:     load all sessions under a parent directory.

    Args:
        session_path: Optional path to a single session directory containing
                      .xplane.pb files (e.g. '/data/my_run'). When set,
                      only that session is shown.
        run_path:     Optional path to a directory containing multiple session
                      directories (e.g. '/data/profiles'). Lists all sessions
                      found underneath.

    Returns:
        A JSON-formatted dict with the list of run names and server URL.
    """
    client = xprof_client.get_client()
    try:
        params: dict = {}
        if session_path:
            params["session_path"] = session_path
        if run_path:
            params["run_path"] = run_path

        import requests  # pylint: disable=g-import-not-at-top
        resp = requests.get(
            f"{client.base_url}/plugins/profile/runs",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        runs = resp.json()

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
