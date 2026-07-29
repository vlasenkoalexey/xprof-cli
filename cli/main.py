"""xprof-cli — CLI frontend over the shared tool registry.

Every tool the MCP server exposes is available as a subcommand:

    xprof-cli list_runs --logdir=/tmp/profiles
    xprof-cli get_overview --logdir=/tmp/profiles --run=myrun
    xprof-cli get_llo_utilization --logdir=... --run=... --kernel=my_kernel
    xprof-cli get_hlo_dump --dump_dir=/tmp/hlo --module_pattern=train_step

Conventions:
  - Runs fully in-process by default (XPROF_MODE=local): no xprof server
    needed; new captures are picked up on every invocation.
  - --logdir may be passed to any command (or set XPROF_LOGDIR).
  - Results are cached in a per-user SQLite cache (1h TTL) keyed on the
    arguments AND the profile directory's mtime, so re-captured runs are
    never served stale. Pass --bypass_cache=True to force recompute.
  - Output goes to stdout (JSON or markdown-ish text depending on the
    tool); errors exit non-zero with a JSON error object on stderr.
"""

import inspect
import json
import logging
import os
import sys

import fire

# Import order is load-bearing: the cache (sqlite3) must be imported BEFORE
# tool_registry pulls in tensorflow. TF drags in a libstdc++ whose ABI can
# conflict with the one conda's _sqlite3 extension needs; importing sqlite3
# first pins the working resolution (and cache.py degrades to a no-op cache
# if sqlite3 is unavailable anyway).
from xprof_mcp.cli import cache as _cache
from xprof_mcp import tool_registry


def _run_salt(run: str | None, logdir: str | None) -> str:
    """Best-effort data-state salt: mtime of the run's session directory."""
    if not run:
        return ""
    try:
        from xprof_mcp.internal import xprof_client  # pylint: disable=g-import-not-at-top

        client = xprof_client.get_client()
        if logdir:
            client.set_logdir(logdir)
        session_dir = client.get_session_dir(run)
        stat = os.stat(session_dir)
        return f"{os.path.abspath(session_dir)}:{stat.st_mtime_ns}"
    except Exception:  # pylint: disable=broad-exception-caught
        return ""


def _make_command(name: str, fn):
    """Wraps a registry tool as a CLI command: logdir + cache + errors.

    Prints the tool's result to stdout itself (rather than letting fire
    interpret the returned string) and exits non-zero both on exceptions
    and on tools that report failure via a JSON {"error": ...} body.
    """

    # Position of the `run` parameter (if any) so the cache salt works
    # whether fire binds it positionally or as a keyword.
    try:
        _param_names = list(inspect.signature(fn).parameters)
        _run_index = _param_names.index("run") if "run" in _param_names else None
    except (ValueError, TypeError):
        _run_index = None

    # Fire turns `--top_stalls 5` into the int 5, and a run named "12345" into
    # the int 12345 too. The latter must become a string again; the former must
    # NOT — blanket-stringifying every number passed "5" to an `int` parameter
    # and blew up on `runs[:top_stalls]`. Coerce per the tool's own annotation.
    try:
        _annots = {
            name: p.annotation
            for name, p in inspect.signature(fn).parameters.items()
        }
    except (ValueError, TypeError):
        _annots = {}

    def _numeric_param(name: str) -> bool:
        """True when the tool declares this parameter as int/float."""
        a = _annots.get(name, inspect.Parameter.empty)
        if a is inspect.Parameter.empty:
            return False
        if a in (int, float):
            return True
        # tolerate string annotations and unions like `int | None`
        return isinstance(a, str) and a.split("|")[0].strip() in ("int", "float")

    def _coerce(name: str, v):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return v
        if not _numeric_param(name):
            return str(v)
        a = _annots.get(name)
        want_float = a is float or (isinstance(a, str)
                                    and a.split("|")[0].strip() == "float")
        return float(v) if want_float else int(v)

    def command(*args, logdir: str = "", bypass_cache: bool = False, **kwargs):
        args = tuple(
            _coerce(_param_names[i] if i < len(_param_names) else "", a)
            for i, a in enumerate(args)
        )
        kwargs = {k: _coerce(k, v) for k, v in kwargs.items()}
        if logdir:
            os.environ["XPROF_LOGDIR"] = logdir
        try:
            if logdir:
                from xprof_mcp.internal import xprof_client  # pylint: disable=g-import-not-at-top

                xprof_client.get_client().set_logdir(logdir)
            if name in tool_registry.UNCACHED_TOOLS:
                result = fn(*args, **kwargs)
            else:
                run_val = kwargs.get("run")
                if run_val is None and _run_index is not None and len(args) > _run_index:
                    run_val = args[_run_index]
                salt = _run_salt(run_val, logdir)
                result = _cache.call_cached(
                    name, fn, args, kwargs, salt=salt, bypass=bypass_cache
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(
                json.dumps(
                    {"error": f"{type(e).__name__}: {e}", "tool": name},
                    indent=2,
                ),
                file=sys.stderr,
            )
            sys.exit(1)
        if result is not None:
            print(result)
        if _cache.result_is_error(result):
            sys.exit(1)

    # Give fire an accurate signature: the tool's own params plus the two
    # CLI-level keyword-only params. (functools.wraps is NOT used because
    # its __wrapped__ attribute would make fire resolve the original
    # signature and reject --logdir/--bypass_cache.)
    command.__name__ = name
    command.__doc__ = fn.__doc__
    try:
        sig = inspect.signature(fn)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        params.append(
            inspect.Parameter(
                "logdir", inspect.Parameter.KEYWORD_ONLY, default="",
                annotation=str,
            )
        )
        params.append(
            inspect.Parameter(
                "bypass_cache", inspect.Parameter.KEYWORD_ONLY, default=False,
                annotation=bool,
            )
        )
        command.__signature__ = sig.replace(
            parameters=params, return_annotation=None
        )
    except (ValueError, TypeError):
        pass
    return command


def _build_commands() -> dict:
    commands = {
        name: _make_command(name, fn)
        for name, fn in tool_registry.ALL_TOOLS.items()
    }
    return commands


def main() -> None:
    # CLI default: in-process analysis. Explicit XPROF_MODE (e.g. http for a
    # remote xprof server) is respected.
    os.environ.setdefault("XPROF_MODE", "local")
    # Keep C++/TF converter chatter off the CLI's stdout/stderr unless the
    # user opts into verbosity.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(_build_commands(), name="xprof-cli")


if __name__ == "__main__":
    main()
