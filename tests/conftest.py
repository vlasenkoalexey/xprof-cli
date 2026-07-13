"""Test bootstrap: make the repo importable as `xprof_mcp` regardless of the
checkout directory name (the GitHub checkout dir is `xprof-mcp`, hyphenated)."""

import importlib.util
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "xprof_mcp" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "xprof_mcp",
        os.path.join(_REPO_ROOT, "__init__.py"),
        submodule_search_locations=[_REPO_ROOT],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["xprof_mcp"] = mod
    spec.loader.exec_module(mod)

FIXTURES = os.path.join(_REPO_ROOT, "tests", "fixtures")
FIXTURE_LOGDIR = os.path.join(FIXTURES, "logdir")
FIXTURE_RUN = "testrun"

try:
    import pytest

    @pytest.fixture
    def local_mode(monkeypatch):
        """Fresh in-process client bound to the fixture logdir."""
        from xprof_mcp.internal import xprof_client

        monkeypatch.setenv("XPROF_MODE", "local")
        monkeypatch.setenv("XPROF_LOGDIR", FIXTURE_LOGDIR)
        xprof_client.set_client(None)
        yield
        xprof_client.set_client(None)

except ImportError:  # conftest is also imported by non-pytest tooling
    pass
