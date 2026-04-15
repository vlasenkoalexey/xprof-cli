"""Tools for reading XLA HLO dump files produced by XLA_FLAGS.

Enable HLO dumps in JAX/XLA by setting XLA_FLAGS before running your program:

  # Minimal: just before/after optimizations (text)
  XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text"

  # All compiler passes:
  XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text \\
             --xla_dump_hlo_pass_re=.*"

  # Also dump proto format (needed for get_hlo_neighborhood on dump files):
  XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text \\
             --xla_dump_hlo_as_proto --xla_dump_hlo_pass_re=.*"

Files produced in the dump directory:
  module_<N>.<name>.before_optimizations.hlo  — pre-optimization HLO text
  module_<N>.<name>.after_optimizations.hlo   — final compiled HLO text
  module_<N>.<name>.after_pass_<Pass>.hlo     — HLO after each compiler pass
  module_<N>.<name>.hlo.pb                    — binary HloProto (all stages)
  module_<N>.<name>.hlo.pbtxt                 — text HloProto

Configure via environment variable:
  XLA_HLO_DUMP_DIR=/tmp/hlo_dumps
"""

import difflib
import fnmatch
import json
import logging
import os
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Matches: module_<N>.<name>.<stage>.hlo
_STAGE_RE = re.compile(
    r"^(?P<prefix>module_\d+\..+?)"
    r"\.(?P<stage>before_optimizations|after_optimizations|after_pass_[^.]+)"
    r"\.hlo$"
)
# Matches: module_<N>.<name>.hlo.pb  or  module_<N>.<name>.hlo.pbtxt
_PROTO_RE = re.compile(
    r"^(?P<prefix>module_\d+\..+?)\.hlo\.(?P<fmt>pb|pbtxt)$"
)
# Matches: module_<N>.<name>.dot  or  module_<N>.<name>.svg
_VIZ_RE = re.compile(
    r"^(?P<prefix>module_\d+\..+?)\.(?P<fmt>dot|svg)$"
)


def _parse_filename(fname: str) -> Optional[dict]:
    """Returns a dict with keys {prefix, stage, ext} or None if not recognised."""
    m = _STAGE_RE.match(fname)
    if m:
        return {"prefix": m.group("prefix"), "stage": m.group("stage"), "ext": "hlo"}

    m = _PROTO_RE.match(fname)
    if m:
        return {"prefix": m.group("prefix"), "stage": "proto", "ext": m.group("fmt")}

    m = _VIZ_RE.match(fname)
    if m:
        return {"prefix": m.group("prefix"), "stage": "viz", "ext": m.group("fmt")}

    return None


def _get_dump_dir() -> str:
    """Returns XLA_HLO_DUMP_DIR env var, or raises if not set."""
    d = os.environ.get("XLA_HLO_DUMP_DIR", "")
    if not d:
        raise RuntimeError(
            "XLA_HLO_DUMP_DIR environment variable is not set.\n"
            "Set it to the directory used with --xla_dump_to=<dir>, e.g.:\n"
            "  export XLA_HLO_DUMP_DIR=/tmp/hlo_dumps\n"
            "Or pass dump_dir explicitly to the tool."
        )
    return d


def _resolve_dump_dir(dump_dir: str) -> str:
    if dump_dir:
        return dump_dir
    return _get_dump_dir()


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------

def _scan_dump_dir(dump_dir: str) -> dict[str, dict]:
    """Scans dump_dir and returns {prefix -> {stage -> filename}}."""
    modules: dict[str, dict] = {}
    try:
        entries = os.listdir(dump_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"HLO dump directory not found: {dump_dir}")

    for fname in sorted(entries):
        info = _parse_filename(fname)
        if info is None:
            continue
        prefix = info["prefix"]
        stage = info["stage"]
        ext = info["ext"]
        if prefix not in modules:
            modules[prefix] = {}
        key = stage if ext == "hlo" else f"{stage}.{ext}"
        modules[prefix][key] = fname

    return modules


def _match_prefix(modules: dict, pattern: str) -> list[str]:
    """Returns prefixes matching the glob pattern (case-insensitive)."""
    pattern_lower = pattern.lower()
    matched = [
        p for p in modules
        if fnmatch.fnmatch(p.lower(), f"*{pattern_lower}*")
        or fnmatch.fnmatch(p.lower(), pattern_lower)
    ]
    return matched


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def list_hlo_dump_modules(dump_dir: str = "") -> str:
    """Lists HLO modules and their available compilation stages in a dump directory.

    Use this to discover which modules were dumped and which compiler stages
    are available (before_optimizations, after_optimizations, per-pass stages).

    Enable dumps with:
      XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text \\
                 --xla_dump_hlo_pass_re=.*"

    Args:
        dump_dir: Path to the XLA dump directory. Defaults to the
                  XLA_HLO_DUMP_DIR environment variable.

    Returns:
        A JSON-formatted dict mapping each module prefix to its available stages.
    """
    try:
        dump_dir = _resolve_dump_dir(dump_dir)
        modules = _scan_dump_dir(dump_dir)
        if not modules:
            return json.dumps(
                {
                    "error": f"No HLO dump files found in {dump_dir}",
                    "tip": (
                        "Run your program with: "
                        "XLA_FLAGS='--xla_dump_to=/tmp/hlo_dumps "
                        "--xla_dump_hlo_as_text'"
                    ),
                },
                indent=2,
            )

        # Summarise stages for each module
        summary = {}
        for prefix, stages in sorted(modules.items()):
            stage_names = sorted(stages.keys())
            # Separate text stages from proto/viz
            text_stages = [s for s in stage_names if not s.startswith("proto") and not s.startswith("viz")]
            other = [s for s in stage_names if s.startswith("proto") or s.startswith("viz")]
            summary[prefix] = {
                "stages": text_stages,
                "also_available": other,
            }

        return json.dumps(
            {
                "dump_dir": dump_dir,
                "module_count": len(summary),
                "modules": summary,
                "tip": (
                    "Pass a module prefix (or part of one) to get_hlo_dump "
                    "or diff_hlo_stages."
                ),
            },
            indent=2,
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error listing HLO dump modules in %s", dump_dir)
        return json.dumps({"error": str(e)}, indent=2)


def get_hlo_dump(
    module_pattern: str,
    stage: str = "after_optimizations",
    dump_dir: str = "",
    max_lines: int = 2000,
) -> str:
    """Returns the HLO text for a module at a specific compilation stage.

    Reads directly from XLA dump files — no xprof server required.

    Useful for inspecting:
      - `before_optimizations`: what JAX/TF produced before XLA touched it.
      - `after_optimizations`: what XLA actually compiled and ran.
      - `after_pass_<Name>`: intermediate state after a specific compiler pass.

    Args:
        module_pattern: Module name or substring to match (case-insensitive glob).
                        Use `list_hlo_dump_modules` to find available names.
        stage:          Compilation stage to read. One of:
                          'before_optimizations' (default pre-opt HLO),
                          'after_optimizations'  (default, final compiled HLO),
                          'after_pass_<PassName>' (e.g. 'after_pass_HloCSE').
                        Defaults to 'after_optimizations'.
        dump_dir:       Path to the XLA dump directory. Defaults to
                        XLA_HLO_DUMP_DIR env var.
        max_lines:      Maximum lines to return (default 2000, -1 for unlimited).

    Returns:
        The HLO text for the matched module and stage.
    """
    try:
        dump_dir = _resolve_dump_dir(dump_dir)
        modules = _scan_dump_dir(dump_dir)
        if not modules:
            return f"No HLO dump files found in {dump_dir}"

        matched = _match_prefix(modules, module_pattern)
        if not matched:
            available = list(modules.keys())[:10]
            return (
                f"No module matching '{module_pattern}' found in {dump_dir}.\n"
                f"Available (first 10): {available}"
            )
        if len(matched) > 1:
            return (
                f"Pattern '{module_pattern}' matched {len(matched)} modules: "
                f"{matched[:10]}{'...' if len(matched) > 10 else ''}.\n"
                "Refine the pattern to match exactly one module."
            )

        prefix = matched[0]
        stages = modules[prefix]

        if stage not in stages:
            available_stages = sorted(stages.keys())
            return (
                f"Stage '{stage}' not available for module '{prefix}'.\n"
                f"Available stages: {available_stages}"
            )

        fpath = os.path.join(dump_dir, stages[stage])
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        if max_lines > 0:
            lines = text.splitlines()
            if len(lines) > max_lines:
                text = "\n".join(lines[:max_lines])
                text += (
                    f"\n... (truncated after {max_lines} lines, "
                    f"total {len(lines)}. Use max_lines=-1 to see all)"
                )

        return f"# {prefix} — {stage}\n# File: {fpath}\n\n{text}"

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception(
            "Error reading HLO dump for module=%s stage=%s dir=%s",
            module_pattern, stage, dump_dir,
        )
        return f"Error reading HLO dump: {e}"


def diff_hlo_stages(
    module_pattern: str,
    stage_before: str = "before_optimizations",
    stage_after: str = "after_optimizations",
    dump_dir: str = "",
    context_lines: int = 5,
    max_lines: int = 500,
) -> str:
    """Shows a unified diff of HLO text between two compilation stages.

    **Use this to understand what a compiler pass changed.** For example,
    diff `before_optimizations` vs `after_optimizations` to see the full
    effect of XLA's optimizer, or diff two consecutive `after_pass_*` stages
    to isolate what a single pass did.

    Args:
        module_pattern:  Module name or substring to match.
        stage_before:    Earlier stage (default: 'before_optimizations').
        stage_after:     Later stage (default: 'after_optimizations').
        dump_dir:        Path to XLA dump directory. Defaults to
                         XLA_HLO_DUMP_DIR env var.
        context_lines:   Lines of context around each change (default 5).
        max_lines:       Max diff lines to return (default 500, -1 unlimited).

    Returns:
        A unified diff string, or a summary if there are no differences.
    """
    try:
        dump_dir = _resolve_dump_dir(dump_dir)
        modules = _scan_dump_dir(dump_dir)

        matched = _match_prefix(modules, module_pattern)
        if not matched:
            return f"No module matching '{module_pattern}' in {dump_dir}."
        if len(matched) > 1:
            return (
                f"Pattern matched {len(matched)} modules: {matched[:10]}. "
                "Refine the pattern."
            )

        prefix = matched[0]
        stages = modules[prefix]

        for stage in (stage_before, stage_after):
            if stage not in stages:
                return (
                    f"Stage '{stage}' not available for '{prefix}'.\n"
                    f"Available stages: {sorted(stages.keys())}"
                )

        def _read(stage: str) -> list[str]:
            fpath = os.path.join(dump_dir, stages[stage])
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                return f.readlines()

        lines_before = _read(stage_before)
        lines_after = _read(stage_after)

        diff = list(
            difflib.unified_diff(
                lines_before,
                lines_after,
                fromfile=f"{prefix}.{stage_before}.hlo",
                tofile=f"{prefix}.{stage_after}.hlo",
                n=context_lines,
            )
        )

        if not diff:
            return (
                f"No differences between '{stage_before}' and '{stage_after}' "
                f"for module '{prefix}'."
            )

        if max_lines > 0 and len(diff) > max_lines:
            diff = diff[:max_lines]
            diff.append(
                f"\n... (diff truncated after {max_lines} lines. "
                "Use max_lines=-1 to see all)\n"
            )

        header = (
            f"# Diff: {prefix}\n"
            f"# {stage_before}  →  {stage_after}\n"
            f"# {len(lines_before)} lines before, {len(lines_after)} lines after\n\n"
        )
        return header + "".join(diff)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception(
            "Error diffing HLO stages for module=%s dir=%s", module_pattern, dump_dir
        )
        return f"Error diffing HLO stages: {e}"


def get_hlo_dump_neighborhood(
    instruction_name: str,
    module_pattern: str,
    stage: str = "after_optimizations",
    dump_dir: str = "",
    radius: int = 2,
) -> str:
    """Returns the neighborhood of an HLO instruction from a dump file.

    Same BFS neighborhood analysis as `get_hlo_neighborhood`, but reads
    directly from XLA dump files instead of the xprof server.

    Args:
        instruction_name: Name of the instruction (e.g. 'fusion.3').
                          Leading '%' is stripped automatically.
        module_pattern:   Module name or substring to match.
        stage:            Compilation stage to use (default 'after_optimizations').
        dump_dir:         Path to XLA dump directory. Defaults to
                          XLA_HLO_DUMP_DIR env var.
        radius:           BFS hops in each direction (default 2).

    Returns:
        A text listing of the instruction neighborhood.
    """
    # Reuse the text-based neighborhood logic from hlo_tools by fetching the
    # text here and delegating the parsing.
    from xprof_mcp.internal import hlo_tools  # pylint: disable=g-import-not-at-top

    instruction_name = instruction_name.lstrip("%")
    try:
        hlo_text = get_hlo_dump(
            module_pattern, stage=stage, dump_dir=dump_dir, max_lines=-1
        )
        if hlo_text.startswith("Error") or hlo_text.startswith("No module"):
            return hlo_text

        # Inline the neighborhood analysis against the dump text
        import collections  # pylint: disable=g-import-not-at-top

        define_re = re.compile(r"^\s*%(?P<name>[\w.\-]+)\s*=")
        operand_re = re.compile(r"%(?P<op>[\w.\-]+)")

        name_to_line: dict[str, str] = {}
        name_to_operands: dict[str, list[str]] = {}

        for line in hlo_text.splitlines():
            m = define_re.match(line)
            if m:
                name = m.group("name")
                name_to_line[name] = line.strip()
                ops = [om.group("op") for om in operand_re.finditer(line[m.end():])]
                name_to_operands[name] = ops

        if instruction_name not in name_to_line:
            candidates = [
                n for n in name_to_line
                if instruction_name.split(".")[0] in n
            ][:10]
            msg = f"Instruction '%{instruction_name}' not found."
            if candidates:
                msg += f" Similar: {', '.join('%' + c for c in candidates)}"
            return msg

        name_to_users: dict[str, list[str]] = collections.defaultdict(list)
        for name, operands in name_to_operands.items():
            for op in operands:
                name_to_users[op].append(name)

        visited: set[str] = {instruction_name}
        queue: collections.deque = collections.deque([(instruction_name, 0)])
        neighborhood: list[tuple[int, str]] = []

        while queue:
            curr, dist = queue.popleft()
            neighborhood.append((dist, curr))
            if dist < radius:
                for nb in name_to_operands.get(curr, []) + name_to_users.get(curr, []):
                    if nb not in visited and nb in name_to_line:
                        visited.add(nb)
                        queue.append((nb, dist + 1))

        neighborhood.sort(key=lambda x: (x[0], x[1]))

        lines_out = [
            f"Neighborhood of '%{instruction_name}' (radius={radius})",
            f"Stage: {stage}  Module: {module_pattern}",
            "",
        ]
        for dist, name in neighborhood:
            indent = "  " * (dist + 1)
            marker = " [TARGET]" if name == instruction_name else f" [dist={dist}]"
            lines_out.append(f"{indent}{marker} {name_to_line.get(name, '%' + name)}")

        return "\n".join(lines_out)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error in get_hlo_dump_neighborhood")
        return f"Error: {e}"
