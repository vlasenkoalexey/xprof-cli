"""Unit tests for the CLI's SQLite result cache (no TF/xprof needed)."""

import json
import time

import pytest

from xprof_mcp.cli import cache as cache_mod


@pytest.fixture
def cache(tmp_path):
    return cache_mod.Cache(directory=tmp_path)


def test_set_get_roundtrip(cache):
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}


def test_miss_returns_default(cache):
    assert cache.get("nope") is cache_mod.Cache.UNKNOWN
    assert cache.get("nope", default=None) is None


def test_expiry(cache):
    cache.set("k", "v", expire=0.05)
    assert cache.get("k") == "v"
    time.sleep(0.1)
    assert cache.get("k") is cache_mod.Cache.UNKNOWN


def test_salt_changes_key():
    k1 = cache_mod.make_key("tool", ("run",), {}, salt="mtime-1")
    k2 = cache_mod.make_key("tool", ("run",), {}, salt="mtime-2")
    assert k1 != k2


def test_result_is_error_detection():
    assert cache_mod.result_is_error(json.dumps({"error": "boom"}))
    assert not cache_mod.result_is_error(json.dumps({"data": 1}))
    assert not cache_mod.result_is_error("plain text result")
    assert not cache_mod.result_is_error(None)


def test_call_cached_hits_second_time(cache, monkeypatch):
    monkeypatch.setattr(cache_mod, "_GLOBAL", cache)
    calls = []

    def fn(run):
        calls.append(run)
        return json.dumps({"data": run})

    r1 = cache_mod.call_cached("t", fn, ("x",), {})
    r2 = cache_mod.call_cached("t", fn, ("x",), {})
    assert r1 == r2
    assert calls == ["x"], "second call must be served from cache"


def test_call_cached_never_stores_errors(cache, monkeypatch):
    """A transient failure must not poison the TTL window."""
    monkeypatch.setattr(cache_mod, "_GLOBAL", cache)
    results = [json.dumps({"error": "logdir missing"}), json.dumps({"ok": 1})]
    calls = []

    def fn(run):
        calls.append(run)
        return results[len(calls) - 1]

    r1 = cache_mod.call_cached("t", fn, ("x",), {})
    assert cache_mod.result_is_error(r1)
    r2 = cache_mod.call_cached("t", fn, ("x",), {})
    assert r2 == json.dumps({"ok": 1})
    assert len(calls) == 2, "error result must not have been cached"


def test_call_cached_bypass(cache, monkeypatch):
    monkeypatch.setattr(cache_mod, "_GLOBAL", cache)
    calls = []

    def fn():
        calls.append(1)
        return "r"

    cache_mod.call_cached("t", fn, (), {})
    cache_mod.call_cached("t", fn, (), {}, bypass=True)
    assert len(calls) == 2


def test_null_cache_fallback(monkeypatch):
    """sqlite3 unavailable -> NullCache, calls still work uncached."""
    monkeypatch.setattr(cache_mod, "sqlite3", None)
    monkeypatch.setattr(cache_mod, "_GLOBAL", None)
    c = cache_mod.get_cache()
    assert isinstance(c, cache_mod.NullCache)
    c.set("k", "v")
    assert c.get("k") is cache_mod.NullCache.UNKNOWN


def test_non_serializable_result_not_cached(cache):
    cache.set("k", object())  # silently skipped
    assert cache.get("k") is cache_mod.Cache.UNKNOWN
