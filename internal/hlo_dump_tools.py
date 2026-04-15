"""Tools for reading XLA HLO dump files produced by XLA_FLAGS.

Supports two dump formats:

**JAX / TF format** (--xla_dump_hlo_as_text):
  Files: module_<N>.<name>.before_optimizations.hlo
         module_<N>.<name>.after_optimizations.hlo
         module_<N>.<name>.after_pass_<Pass>.hlo
         module_<N>.<name>.hlo.pb / .hlo.pbtxt

  Enable with:
    XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text"
    XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text \\
               --xla_dump_hlo_pass_re=.*"   # all compiler passes

**PyTorch/XLA format** (torch.compile with openxla backend):
  Files: module_<N>.<name>[.cl_<id>].<seq>.<pipeline>.<after_X>.<before_Y>.txt

  Stage aliases: "before_optimizations" = first seq, "after_optimizations" = last seq.
  Intermediate stages: "pass_0001", "pass_0042", etc. (4-digit sequence number).

  Enable with:
    XLA_FLAGS="--xla_dump_to=/tmp/hlo_dumps --xla_dump_hlo_as_text"
  (PyTorch/XLA uses text HLO with its own naming convention.)

  Recommended: text format (--xla_dump_hlo_as_text) for both JAX and PyTorch/XLA —
  it is human-readable, works with all tools here, and does not require protobuf.

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

# JAX/TF format: module_<N>.<name>.<stage>.hlo
_STAGE_RE = re.compile(
    r"^(?P<prefix>module_\d+\..+?)"
    r"\.(?P<stage>before_optimizations|after_optimizations|after_pass_[^.]+)"
    r"\.hlo$"
)
# JAX/TF proto format: module_<N>.<name>.hlo.pb  or  .hlo.pbtxt
_PROTO_RE = re.compile(
    r"^(?P<prefix>module_\d+\..+?)\.hlo\.(?P<fmt>pb|pbtxt)$"
)
# JAX/TF viz format: module_<N>.<name>.dot  or  .svg
_VIZ_RE = re.compile(
    r"^(?P<prefix>module_\d+\..+?)\.(?P<fmt>dot|svg)$"
)
# PyTorch/XLA format: module_<N>.<name>[.cl_<id>].<seq>.<pipeline>.<after_X>.<before_Y>.txt
# The module name may contain dots (e.g. tt_jit_distributed.barrier_distributed.all_reduce).
# The 4-digit sequence number is the disambiguating token.
_PTXLA_RE = re.compile(
    r"^(?P<prefix>module_\d+\..+?)"   # module_N.name  (lazy — grows until rest matches)
    r"(?:\.cl_\d+)?"                   # optional .cl_NNNNNNNN  (PyTorch/XLA compilation id)
    r"\.(?P<seq>\d{4})"                # .NNNN  sequence / pass number
    r"\.[^.]+\.[^.]+\.[^.]+\.txt$"    # .pipeline.after_X.before_Y.txt
)


def _parse_filename(fname: str) -> Optional[dict]:
    """Returns a dict with keys {prefix, stage, ext, format} or None."""
    m = _STAGE_RE.match(fname)
    if m:
        return {"prefix": m.group("prefix"), "stage": m.group("stage"),
                "ext": "hlo", "format": "jax"}

    m = _PROTO_RE.match(fname)
    if m:
        return {"prefix": m.group("prefix"), "stage": "proto",
                "ext": m.group("fmt"), "format": "jax"}

    m = _VIZ_RE.match(fname)
    if m:
        return {"prefix": m.group("prefix"), "stage": "viz",
                "ext": m.group("fmt"), "format": "jax"}

    m = _PTXLA_RE.match(fname)
    if m:
        return {"prefix": m.group("prefix"), "seq": int(m.group("seq")),
                "ext": "txt", "format": "ptxla"}

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
    """Scans dump_dir and returns {prefix -> {stage -> filename}}.

    For JAX/TF format, stages are named 'before_optimizations',
    'after_optimizations', 'after_pass_<Name>', 'proto', 'proto.pb', etc.

    For PyTorch/XLA format, stages are named:
      'before_optimizations'  — alias for the first (seq=0000) file
      'after_optimizations'   — alias for the last (highest seq) file
      'pass_NNNN'             — specific intermediate stage by sequence number
    """
    modules: dict[str, dict] = {}
    # Accumulate PyTorch/XLA entries by prefix -> {seq -> fname} before finalising
    ptxla_seqs: dict[str, dict[int, str]] = {}

    try:
        entries = os.listdir(dump_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"HLO dump directory not found: {dump_dir}")

    for fname in sorted(entries):
        info = _parse_filename(fname)
        if info is None:
            continue
        prefix = info["prefix"]

        if info["format"] == "ptxla":
            ptxla_seqs.setdefault(prefix, {})[info["seq"]] = fname
        else:
            # JAX/TF format
            stage = info["stage"]
            ext = info["ext"]
            if prefix not in modules:
                modules[prefix] = {"_format": "jax"}
            key = stage if ext == "hlo" else f"{stage}.{ext}"
            modules[prefix][key] = fname

    # Finalise PyTorch/XLA modules: map seq numbers to stage names
    for prefix, seq_map in ptxla_seqs.items():
        if prefix not in modules:
            modules[prefix] = {"_format": "ptxla"}
        else:
            modules[prefix]["_format"] = "ptxla"
        sorted_seqs = sorted(seq_map)
        for seq in sorted_seqs:
            modules[prefix][f"pass_{seq:04d}"] = seq_map[seq]
        # Convenience aliases
        modules[prefix]["before_optimizations"] = seq_map[sorted_seqs[0]]
        modules[prefix]["after_optimizations"] = seq_map[sorted_seqs[-1]]
        modules[prefix]["_ptxla_seq_count"] = len(sorted_seqs)
        modules[prefix]["_ptxla_first_seq"] = sorted_seqs[0]
        modules[prefix]["_ptxla_last_seq"] = sorted_seqs[-1]

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
            fmt = stages.get("_format", "jax")
            # Filter out internal metadata keys
            stage_names = [k for k in sorted(stages.keys()) if not k.startswith("_")]

            if fmt == "ptxla":
                # For PyTorch/XLA, show compact summary (can have 100+ passes)
                n = stages.get("_ptxla_seq_count", 0)
                first = stages.get("_ptxla_first_seq", 0)
                last = stages.get("_ptxla_last_seq", 0)
                summary[prefix] = {
                    "format": "ptxla",
                    "stages": ["before_optimizations", "after_optimizations"],
                    "intermediate_passes": n,
                    "pass_range": f"pass_{first:04d}..pass_{last:04d}",
                    "note": "Use 'before_optimizations'/'after_optimizations' or 'pass_NNNN'",
                }
            else:
                text_stages = [s for s in stage_names if not s.startswith("proto") and not s.startswith("viz")]
                other = [s for s in stage_names if s.startswith("proto") or s.startswith("viz")]
                summary[prefix] = {
                    "format": "jax",
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
      - `before_optimizations`: what JAX/TF/PyTorch produced before XLA optimized it.
      - `after_optimizations`: the final compiled HLO.
      - `after_pass_<Name>`: intermediate state after a specific compiler pass (JAX format).
      - `pass_NNNN`: specific intermediate pass by sequence number (PyTorch/XLA format).

    Args:
        module_pattern: Module name or substring to match (case-insensitive glob).
                        Use `list_hlo_dump_modules` to find available names.
                        For PyTorch/XLA with many instances of the same module name
                        (e.g. jit_splash_fn), include the module number to disambiguate
                        (e.g. 'module_0006' or 'module_0006.jit__my_fwd').
        stage:          Compilation stage to read. Supported values:
                          'before_optimizations'  — first/pre-opt stage (both formats)
                          'after_optimizations'   — last/final stage (both formats)
                          'after_pass_<PassName>' — JAX format only
                          'pass_NNNN'             — PyTorch/XLA format, by sequence number
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
            # For PyTorch/XLA, many instances of the same named module are common.
            # If they all share the same logical name, pick the first and note it.
            unique_names = {re.sub(r'^module_\d+\.', '', p) for p in matched}
            if len(unique_names) == 1:
                # All same logical name — use the first (lowest module number)
                matched = [matched[0]]
            else:
                return (
                    f"Pattern '{module_pattern}' matched {len(matched)} distinct modules: "
                    f"{matched[:10]}{'...' if len(matched) > 10 else ''}.\n"
                    "Refine the pattern to match exactly one module "
                    "(include the module number, e.g. 'module_0006.jit__my_fwd')."
                )

        prefix = matched[0]
        stages = modules[prefix]

        if stage not in stages:
            fmt = stages.get("_format", "jax")
            if fmt == "ptxla":
                available_stages = (
                    ["before_optimizations", "after_optimizations"]
                    + [k for k in sorted(stages) if k.startswith("pass_")]
                )
            else:
                available_stages = [k for k in sorted(stages) if not k.startswith("_")]
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
            unique_names = {re.sub(r'^module_\d+\.', '', p) for p in matched}
            if len(unique_names) == 1:
                matched = [matched[0]]
            else:
                return (
                    f"Pattern matched {len(matched)} distinct modules: {matched[:10]}. "
                    "Refine the pattern."
                )

        prefix = matched[0]
        stages = modules[prefix]

        for stage in (stage_before, stage_after):
            if stage not in stages:
                available = [k for k in sorted(stages) if not k.startswith("_")]
                return (
                    f"Stage '{stage}' not available for '{prefix}'.\n"
                    f"Available stages: {available}"
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
