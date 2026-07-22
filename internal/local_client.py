"""In-process xprof client — analyzes profiles without a running xprof server.

Drop-in replacement for OSSXprofClient: same duck-type surface
(get_runs / get_hosts / get_hlo_module_list / fetch / direct-disk reads),
but `fetch()` runs the OSS xprof converters in-process via
`xprof.convert.raw_to_tool_data.xspace_to_tool_data(...)` instead of an
HTTP round-trip to `xprof --logdir=... --port=8791`.

Consequences:
  - No server to start/restart; new profile captures are visible on the
    next call because every call re-resolves paths from the logdir.
  - Requires the `xprof` pip package (converters + pywrap C++ backend)
    and, for GCS logdirs, tensorflow's gfile. Install via
    `pip install xprof-cli[full]`.

Selection between this client and the HTTP client happens in
`xprof_client.get_client()` (XPROF_MODE env var: local | http | auto).
"""

import logging
import os
from typing import Optional

from xprof_mcp.internal import xprof_client as _http_client

logger = logging.getLogger(__name__)

# Tool tags whose converter requires exactly one xplane file.
_SINGLE_HOST_TOOLS = frozenset({"memory_profile"})

# Query-arg names that belong to graph_viewer options (mirrors
# profile_plugin._get_graph_viewer_options).
_GRAPH_VIEWER_OPTION_KEYS = (
    "node_name",
    "module_name",
    "program_id",
    "graph_width",
    "show_metadata",
    "merge_fusion",
    "format",
    "type",
)

import sys

def converters_available() -> bool:
    """True if the xprof pip converters are importable."""
    try:
        from xprof.convert import raw_to_tool_data  # noqa: F401 pylint: disable=g-import-not-at-top,unused-import

        return True
    except Exception as e:  # noqa: BLE001 — any import-time failure means converters are unusable
        print("converters_available: import failed:", e, file=sys.stderr)
        return False


def _convert():
    from xprof.convert import raw_to_tool_data  # pylint: disable=g-import-not-at-top

    return raw_to_tool_data


class LocalXprofClient(_http_client.OSSXprofClient):
    """Reads profile runs from the logdir and converts tool data in-process.

    Inherits the logdir resolution (XPROF_LOGDIR env or procfs detection),
    nested-session-dir mapping, and direct-disk XPlane/HLO-proto readers
    from OSSXprofClient; replaces every HTTP endpoint with a local
    equivalent.
    """

    # ------------------------------------------------------------------
    # Listing endpoints (local equivalents of the HTTP routes)
    # ------------------------------------------------------------------

    def get_runs(self) -> list[str]:
        """Scans the logdir for profile runs.

        Mirrors profile_plugin.generate_runs frontend naming: a session
        directory <logdir>/plugins/profile/<run> is reported as "<run>";
        a nested capture <logdir>/<sub>/plugins/profile/<run> is reported
        as "<sub>/<run>". Nesting is scanned up to 3 levels deep, which
        covers the TensorBoard train/validation layout and per-experiment
        capture trees.
        """
        logdir = self._require_logdir()
        runs: list[str] = []

        def scan(prefix: str, depth: int) -> None:
            base = os.path.join(logdir, prefix) if prefix else logdir
            profile_dir = os.path.join(base, "plugins", "profile")
            if _http_client._gfile_exists(profile_dir):  # pylint: disable=protected-access
                try:
                    for entry in _http_client._gfile_listdir(profile_dir):  # pylint: disable=protected-access
                        entry = entry.rstrip("/")
                        session = os.path.join(profile_dir, entry)
                        if _http_client._gfile_exists(session):  # pylint: disable=protected-access
                            runs.append(
                                os.path.join(prefix, entry) if prefix else entry
                            )
                except OSError:
                    pass
            if depth <= 0:
                return
            try:
                entries = _http_client._gfile_listdir(base)  # pylint: disable=protected-access
            except OSError:
                return
            for entry in entries:
                entry = entry.rstrip("/")
                if entry == "plugins":
                    continue
                sub = os.path.join(base, entry)
                if os.path.isdir(sub) if not _http_client._is_gcs(sub) else True:  # pylint: disable=protected-access
                    scan(os.path.join(prefix, entry) if prefix else entry, depth - 1)

        scan("", 3)
        return sorted(runs)

    def get_run_tools(self, run: str) -> list[str]:
        """Tools available for a run (local mode: static superset)."""
        del run
        # The HTTP route inspects the xspace to enumerate tools; locally the
        # converter simply fails per-tool when data is absent, so report the
        # supported set instead of paying a full xspace parse here.
        return [
            "overview_page",
            "hlo_stats",
            "op_profile",
            "memory_profile",
            "roofline_model",
            "graph_viewer",
            "pod_viewer",
            "megascale_stats",
            "memory_viewer",
            "input_pipeline_analyzer",
            "framework_op_stats",
            "kernel_stats",
            "utilization_viewer",
            "smart_suggestion",
        ]

    def get_hosts(self, run: str, tool: str = "overview_page") -> list[dict]:
        """Returns host metadata by scanning xplane filenames."""
        del tool
        return [{"hostname": h} for h in self.list_xplane_hosts(run)]

    def get_hlo_module_list(self, run: str, host: str = "") -> str:
        """Returns comma-separated HLO module names for a run.

        Mirrors profile_plugin.hlo_module_list_impl: list *.hlo_proto.pb in
        the session dir; when none exist yet, trigger the C++ xspace →
        hlo_proto extraction (writes the .hlo_proto.pb files next to the
        trace) and rescan.
        """
        del host
        session_dir = self.get_session_dir(run)

        def scan() -> list[str]:
            try:
                return sorted(
                    f[: -len(".hlo_proto.pb")]
                    for f in _http_client._gfile_listdir(session_dir)  # pylint: disable=protected-access
                    if f.endswith(".hlo_proto.pb")
                )
            except OSError:
                return []

        modules = scan()
        if not modules:
            xplane_paths = [
                self.get_xplane_file_path(run, h)
                for h in self.list_xplane_hosts(run)
            ]
            if xplane_paths:
                try:
                    _convert().xspace_to_tool_names(xplane_paths)
                except Exception:  # pylint: disable=broad-exception-caught
                    logger.warning(
                        "xspace -> hlo_proto extraction failed for %s",
                        run,
                        exc_info=True,
                    )
                modules = scan()
        return ",".join(modules)

    # ------------------------------------------------------------------
    # Data fetch — the core replacement
    # ------------------------------------------------------------------

    def fetch(
        self,
        tool: str,
        run: str,
        host: str = "",
        timeout: int = 120,
        **kwargs,
    ) -> bytes:
        """Converts tool data in-process (no HTTP).

        Args mirror OSSXprofClient.fetch; `timeout` is ignored. Extra
        kwargs are mapped onto the converter params the same way the xprof
        server's data route maps query args (module_name, program_id,
        graph_viewer options, memory_space, ...).

        Note the graph_viewer quirk inherited from the HTTP protocol: HLO
        tools pass the *module name* via the `host` argument.
        """
        del timeout
        logger.info(
            "local fetch: tool=%s run=%s host=%s extra=%s", tool, run, host, kwargs
        )
        tool = tool[:-1] if tool.endswith("^") else tool

        params: dict = {
            "host": host or None,
            "module_name": kwargs.pop("module_name", None),
            "program_id": kwargs.pop("program_id", None),
            # Never write cached tool-data files next to the trace; the
            # trace directories may be immutable (raw/profiles) and the CLI
            # layer has its own result cache.
            "use_saved_result": kwargs.pop("use_saved_result", False),
            "memory_space": str(kwargs.pop("memory_space", "0")),
            "tqx": kwargs.pop("tqx", None),
        }
        if tool == "graph_viewer":
            options = {
                k: kwargs.pop(k) for k in _GRAPH_VIEWER_OPTION_KEYS if k in kwargs
            }
            # HTTP protocol quirk: module selection arrives via `host`.
            if host and "module_name" not in options:
                options["module_name"] = host
            options.setdefault("graph_width", 3)
            options["show_metadata"] = int(
                str(options.get("show_metadata", "")).lower() in ("1", "true")
            )
            options["merge_fusion"] = int(
                str(options.get("merge_fusion", "")).lower() in ("1", "true")
            )
            params["graph_viewer_options"] = options
            params["host"] = None
        if tool == "memory_viewer" and kwargs.pop(
            "view_memory_allocation_timeline", False
        ):
            params["view_memory_allocation_timeline"] = True
        params.update(kwargs)  # forward anything else verbatim

        xspace_paths = self._select_xspace_paths(tool, run, host)
        params["hosts"] = [
            os.path.basename(p)[: -len(".xplane.pb")] for p in xspace_paths
        ]

        data, _content_type = _convert().xspace_to_tool_data(
            xspace_paths, tool, params
        )
        if isinstance(data, bytes):
            return data
        return str(data).encode("utf-8")

    def _select_xspace_paths(self, tool: str, run: str, host: str) -> list[str]:
        """Mirrors profile_plugin._get_valid_hosts path selection."""
        hosts = self.list_xplane_hosts(run)
        if not hosts:
            session_dir = self.get_session_dir(run)
            raise FileNotFoundError(
                f"No .xplane.pb files found for run {run!r} in {session_dir}"
            )
        if host and host != "ALL_HOSTS":
            if host in hosts:
                return [self.get_xplane_file_path(run, host)]
            # graph_viewer abuses `host` for the module name; other tools may
            # get a stale hostname — fall through to all hosts rather than
            # failing hard, matching the server's lenient default path.
            if tool != "graph_viewer":
                raise FileNotFoundError(
                    f"No xplane file for host {host!r} in run {run!r}; "
                    f"available: {hosts}"
                )
        if tool in _SINGLE_HOST_TOOLS:
            return [self.get_xplane_file_path(run, hosts[0])]
        return [self.get_xplane_file_path(run, h) for h in hosts]


# ---------------------------------------------------------------------------
# Factory helper used by xprof_client.get_client()
# ---------------------------------------------------------------------------


def try_create() -> Optional[LocalXprofClient]:
    """Returns a LocalXprofClient if converters + logdir are available."""
    if not converters_available():
        return None
    client = LocalXprofClient()
    if not client.logdir:
        return None
    return client
