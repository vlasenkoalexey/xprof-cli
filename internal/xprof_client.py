"""OSS XProf HTTP client.

Connects to a locally running `xprof` server (default: http://localhost:8791).

Configuration via environment variables:
  XPROF_URL    - Base URL of the xprof server (default: http://localhost:8791)
  XPROF_LOGDIR - Path to the logdir used when starting the xprof server.
                 Required for tools that read raw .xplane.pb / .hlo_proto.pb
                 files directly from disk.
"""

import logging
import os
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# GCS-aware file helpers (use tf.io.gfile when logdir is a GCS path)
# ---------------------------------------------------------------------------

def _is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def _gfile_exists(path: str) -> bool:
    if _is_gcs(path):
        try:
            import tensorflow as tf  # pylint: disable=g-import-not-at-top
            return tf.io.gfile.exists(path)
        except ImportError:
            return False
    return os.path.exists(path)


def _gfile_listdir(path: str) -> list[str]:
    if _is_gcs(path):
        try:
            import tensorflow as tf  # pylint: disable=g-import-not-at-top
            return tf.io.gfile.listdir(path)
        except ImportError:
            return []
    return os.listdir(path)


def _gfile_read(path: str) -> bytes:
    if _is_gcs(path):
        import tensorflow as tf  # pylint: disable=g-import-not-at-top
        with tf.io.gfile.GFile(path, "rb") as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()

_DEFAULT_XPROF_URL = "http://localhost:8791"
_INSTANCE: Optional["OSSXprofClient"] = None


class OSSXprofClient:
    """HTTP client for the OSS XProf server."""

    def __init__(self, base_url: str = "", logdir: str = ""):
        self._base_url = (
            base_url or os.environ.get("XPROF_URL", _DEFAULT_XPROF_URL)
        ).rstrip("/")
        self._logdir = logdir or os.environ.get("XPROF_LOGDIR", "")
        self._session = requests.Session()
        self._session.headers.update({"Accept": "*/*"})

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def logdir(self) -> str:
        return self._logdir

    # ------------------------------------------------------------------
    # Listing endpoints
    # ------------------------------------------------------------------

    def get_runs(self) -> list[str]:
        """Returns a sorted list of available profiling run names."""
        resp = self._session.get(
            f"{self._base_url}/data/plugin/profile/runs", timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def get_run_tools(self, run: str) -> list[str]:
        """Returns available tools for a given run."""
        resp = self._session.get(
            f"{self._base_url}/data/plugin/profile/run_tools",
            params={"run": run},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_hosts(self, run: str, tool: str = "overview_page") -> list[dict]:
        """Returns host metadata for a run and tool.

        Returns a list of dicts with key 'hostname'.
        """
        resp = self._session.get(
            f"{self._base_url}/data/plugin/profile/hosts",
            params={"run": run, "tag": tool},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_hlo_module_list(self, run: str, host: str = "") -> str:
        """Returns a comma-separated string of HLO module names."""
        params: dict = {"run": run}
        if host:
            params["host"] = host
        resp = self._session.get(
            f"{self._base_url}/data/plugin/profile/module_list",
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.text.strip()

    # ------------------------------------------------------------------
    # Data fetch
    # ------------------------------------------------------------------

    def fetch(
        self,
        tool: str,
        run: str,
        host: str = "",
        timeout: int = 120,
        **kwargs,
    ) -> bytes:
        """Fetches raw tool data bytes from the xprof server.

        Args:
            tool:    Tool name, e.g. 'overview_page', 'hlo_stats',
                     'graph_viewer', 'memory_profile'.
            run:     Run / session name (directory name under plugins/profile/).
            host:    Host name or 'ALL_HOSTS'. For tools that support all hosts
                     leave empty to let the server choose.
            timeout: Request timeout in seconds.
            **kwargs: Additional query params forwarded to the server
                      (e.g. type='long_txt' for graph_viewer).
        """
        params: dict = {"run": run, "tag": tool}
        if host:
            params["host"] = host
        params.update(kwargs)
        logging.info("xprof fetch: tool=%s run=%s host=%s extra=%s", tool, run, host, kwargs)
        resp = self._session.get(
            f"{self._base_url}/data/plugin/profile/data",
            params=params,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.content

    def fetch_json(self, tool: str, run: str, host: str = "", **kwargs) -> bytes:
        """Fetches tool data, raising if the response is not JSON-like."""
        return self.fetch(tool, run, host, **kwargs)

    # ------------------------------------------------------------------
    # Direct disk access (requires XPROF_LOGDIR)
    # ------------------------------------------------------------------

    def _require_logdir(self) -> str:
        if not self._logdir:
            raise RuntimeError(
                "XPROF_LOGDIR environment variable is not set. "
                "Set it to the logdir you passed to `xprof --logdir=...` "
                "to enable tools that read raw .xplane.pb files."
            )
        return self._logdir

    def get_session_dir(self, run: str) -> str:
        """Returns the on-disk session directory for a run."""
        logdir = self._require_logdir()
        return os.path.join(logdir, "plugins", "profile", run)

    def get_xplane_file_path(self, run: str, host: str) -> str:
        """Returns path to <host>.xplane.pb for the given run."""
        return os.path.join(self.get_session_dir(run), f"{host}.xplane.pb")

    def get_hlo_proto_file_path(self, run: str, module_name: str) -> str:
        """Returns path to <module_name>.hlo_proto.pb for the given run."""
        return os.path.join(self.get_session_dir(run), f"{module_name}.hlo_proto.pb")

    def read_xplane_bytes(self, run: str, host: str) -> bytes:
        """Reads and returns raw bytes from <host>.xplane.pb."""
        path = self.get_xplane_file_path(run, host)
        if not _gfile_exists(path):
            session_dir = self.get_session_dir(run)
            try:
                available = _gfile_listdir(session_dir)
            except Exception:  # pylint: disable=broad-exception-caught
                available = []
            raise FileNotFoundError(
                f"XPlane file not found: {path}\n"
                f"Available files: {available}"
            )
        return _gfile_read(path)

    def read_hlo_proto_bytes(self, run: str, module_name: str) -> bytes:
        """Reads and returns raw bytes from <module_name>.hlo_proto.pb."""
        path = self.get_hlo_proto_file_path(run, module_name)
        if not _gfile_exists(path):
            raise FileNotFoundError(f"HLO proto file not found: {path}")
        return _gfile_read(path)

    def list_xplane_hosts(self, run: str) -> list[str]:
        """Lists hosts by scanning xplane.pb files in the session directory."""
        session_dir = self.get_session_dir(run)
        hosts = []
        try:
            for fname in _gfile_listdir(session_dir):
                if fname.endswith(".xplane.pb"):
                    host = fname[: -len(".xplane.pb")]
                    hosts.append(host)
        except (FileNotFoundError, Exception):  # pylint: disable=broad-exception-caught
            pass
        return sorted(hosts)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

def get_client() -> OSSXprofClient:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = OSSXprofClient()
    return _INSTANCE


def set_client(client: Optional[OSSXprofClient]) -> None:
    """Override the global client (useful for testing)."""
    global _INSTANCE
    _INSTANCE = client
