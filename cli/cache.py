"""Persistent SQLite result cache for the CLI.

Adapted from openxla/xprof plugin/xprof/cli/internal/decorators.py
(Apache-2.0), with one addition: an optional *salt* in the cache key so
callers can bind an entry to the state of the data on disk (e.g. the
profile session directory's mtime). A re-captured run therefore misses the
cache instead of returning stale results — important for the autoresearch
loop, which reuses run names within the cache TTL.

The CLI runs process-per-call, so an in-memory cache is useless; this is
what keeps repeat invocations (agents re-reading the same overview) fast.
"""

import contextlib
import getpass
import json
import logging
import pathlib
import tempfile
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import sqlite3
except ImportError:  # environment-specific ABI conflicts (e.g. conda
    sqlite3 = None   # libstdc++ vs system) — degrade to no caching.

_UNKNOWN = object()
DEFAULT_EXPIRE_S = 3600.0


def _cache_dir() -> pathlib.Path:
    d = pathlib.Path(tempfile.gettempdir()) / f"xprof_cli_cache_{getpass.getuser()}"
    d.mkdir(mode=0o700, exist_ok=True)
    return d


class Cache:
    """Minimal persistent SQLite-backed cache (JSON values)."""

    UNKNOWN = _UNKNOWN

    def __init__(self, directory: pathlib.Path | None = None):
        self.directory = directory or _cache_dir()
        self.db_path = self.directory / "cache.db"
        self._init_db()

    def _init_db(self) -> None:
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS cache ("
                    " key TEXT PRIMARY KEY, value TEXT, expire REAL)"
                )
                conn.execute(
                    "DELETE FROM cache WHERE expire IS NOT NULL AND expire < ?",
                    (time.time(),),
                )
                conn.commit()
        except sqlite3.Error:
            logger.warning("cache init failed; caching disabled", exc_info=True)

    def get(self, key: str, default: Any = _UNKNOWN) -> Any:
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                row = conn.execute(
                    "SELECT value, expire FROM cache WHERE key = ?", (key,)
                ).fetchone()
            if row is None:
                return default
            value_str, expire = row
            if expire is not None and expire < time.time():
                self.delete(key)
                return default
            return json.loads(value_str)
        except (sqlite3.Error, json.JSONDecodeError, TypeError):
            return default

    def set(self, key: str, value: Any, expire: float | None = None) -> None:
        try:
            val = json.dumps(value)
        except TypeError:
            return  # non-JSON-serializable results are simply not cached
        expire_at = time.time() + expire if expire is not None else None
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expire)"
                    " VALUES (?, ?, ?)",
                    (key, val, expire_at),
                )
                conn.commit()
        except sqlite3.Error:
            pass  # best effort

    def delete(self, key: str) -> None:
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
        except sqlite3.Error:
            pass


class NullCache:
    """No-op cache used when sqlite3 is unavailable in the environment."""

    UNKNOWN = _UNKNOWN

    def get(self, key: str, default: Any = _UNKNOWN) -> Any:  # noqa: D102
        del key
        return default

    def set(self, key: str, value: Any, expire: float | None = None) -> None:  # noqa: D102
        del key, value, expire

    def delete(self, key: str) -> None:  # noqa: D102
        del key


_GLOBAL: "Cache | NullCache | None" = None


def get_cache() -> "Cache | NullCache":
    global _GLOBAL
    if _GLOBAL is None:
        if sqlite3 is None:
            logger.warning("sqlite3 unavailable; CLI result caching disabled")
            _GLOBAL = NullCache()
        else:
            _GLOBAL = Cache()
    return _GLOBAL


def make_key(tool_name: str, args: tuple, kwargs: dict, salt: str = "") -> str:
    """Stable cache key over (tool, args, kwargs, data-state salt)."""
    return json.dumps(
        [tool_name, list(args), sorted(kwargs.items()), salt],
        sort_keys=True,
        default=str,
    )


def result_is_error(result: Any) -> bool:
    """True if a tool's string result is a top-level JSON error object."""
    if not isinstance(result, str) or not result.lstrip().startswith("{"):
        return False
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict) and "error" in parsed


def call_cached(
    tool_name: str,
    fn,
    args: tuple,
    kwargs: dict,
    *,
    salt: str = "",
    bypass: bool = False,
    expire: float = DEFAULT_EXPIRE_S,
):
    """Runs fn(*args, **kwargs) through the cache.

    Error results (JSON {"error": ...} bodies) are returned but never
    stored — a transient failure (missing logdir, race with a capture in
    progress) must not poison subsequent calls for the TTL duration.
    """
    key = make_key(tool_name, args, kwargs, salt)
    cache = get_cache()
    if not bypass:
        hit = cache.get(key)
        if hit is not Cache.UNKNOWN:
            logger.debug("cache hit: %s", tool_name)
            return hit
    result = fn(*args, **kwargs)
    if not result_is_error(result):
        cache.set(key, result, expire=expire)
    return result
