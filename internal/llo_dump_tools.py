"""Tools for reading LLO (Low-Level Optimizer) dump files from libtpu.

LLO is the TPU backend IR below HLO: the jellyfish compiler lowers each HLO
program (including Pallas/Mosaic custom calls) into VLIW instruction bundles.
libtpu can dump every pass of that pipeline as text:

    LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/jf_dump \\
                      --xla_jf_dump_llo_text=true \\
                      [--xla_jf_dump_llo_static_gaps=true] \\
                      [--xla_jf_dump_llo_pass_label_regex=<re>]"

IMPORTANT: the LLO dumper uses its own `--xla_jf_dump_to` directory flag —
XLA's `--xla_dump_to` does NOT receive LLO dumps (the llo flags silently
write nothing without it). These flags are undocumented (validated on v6e,
2026-07-09) and may drift across libtpu releases.

File naming: `<epoch-ns>-<program>-<NN>-<pass-label>.{txt,html}` where
`<program>` is the LLO program name (the jitted-fn-derived kernel program,
XLA fusions, or `<late-initialization>`/`<late-finalization>`/`TLP`
infrastructure programs) and `<NN>` orders the passes. Useful checkpoints:

  - `final_bundles.txt`                       VLIW bundle listing, HLO-attributed
  - `*hlo-static-per-bundle-utilization.txt`  per-bundle slot matrix vs capacity
  - `schedule-analysis_*.txt`                 bundles per HLO / per opcode
  - `static_gap_analysis_*.txt`               static gap analysis

Configure via environment variable:
  XLA_JF_DUMP_DIR=/tmp/jf_dump
"""

import collections
import json
import logging
import os
import re
import statistics
from typing import Optional

# <epoch-ns>-<program>-<NN>-<pass>.<ext>; program names may contain '.', '-',
# and angle brackets (<late-initialization>), so we anchor on the LAST
# '-<2-3 digit>-' occurrence, which is the pass sequence number.
_FNAME_RE = re.compile(
    r"^(?P<ts>\d{10,})-(?P<rest>.+)\.(?P<ext>txt|html)$"
)
_PASS_SPLIT_RE = re.compile(r"-(\d{2,3})-(?!.*-\d{2,3}-)")

_INFRA_PROGRAMS = ("TLP", "<late-initialization>", "<late-finalization>")

_NO_DIR_MSG = (
    "LLO dump directory not found or empty: {d!r}. Pass dump_dir explicitly "
    "or set XLA_JF_DUMP_DIR. Capture with LIBTPU_INIT_ARGS="
    "\"--xla_jf_dump_to=<dir> --xla_jf_dump_llo_text=true\" "
    "(note: --xla_dump_to alone does NOT produce LLO dumps)."
)


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------

def _resolve_dump_dir(dump_dir: str) -> str:
    d = dump_dir or os.environ.get("XLA_JF_DUMP_DIR", "")
    return os.path.expanduser(d) if d else ""


def _parse_filename(fname: str) -> Optional[dict]:
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    rest = m.group("rest")
    sm = _PASS_SPLIT_RE.search(rest)
    if not sm:
        # e.g. `TLP-hlo.txt` — program with a bare label, no pass number.
        return {"ts": int(m.group("ts")), "program": rest, "pass_num": None,
                "pass_label": "", "ext": m.group("ext")}
    return {
        "ts": int(m.group("ts")),
        "program": rest[: sm.start()],
        "pass_num": int(sm.group(1)),
        "pass_label": rest[sm.end():],
        "ext": m.group("ext"),
    }


def _scan(dump_dir: str) -> dict[str, dict]:
    """Returns {program: {instances: {ts: {pass_label: fname}}, ...}}.

    A program may be compiled several times (several timestamps); each
    timestamp group is one complete pass pipeline.
    """
    programs: dict[str, dict] = collections.defaultdict(
        lambda: {"instances": collections.defaultdict(dict)})
    for fname in os.listdir(dump_dir):
        info = _parse_filename(fname)
        if info is None or info["ext"] != "txt":
            continue
        prog = programs[info["program"]]
        prog["instances"][info["ts"]][info["pass_label"]] = fname
    return dict(programs)


def _classify(program: str) -> str:
    if program in _INFRA_PROGRAMS or program.startswith("TLP"):
        return "infrastructure"
    if "fusion" in program:
        return "fusion"
    return "kernel"


def _pick_instance(prog: dict, instance_ts: Optional[int] = None) -> tuple[int, dict]:
    """Picks a compile-instance (default: the one with the most passes)."""
    instances = prog["instances"]
    if instance_ts is not None:
        if instance_ts not in instances:
            raise KeyError(f"instance ts {instance_ts} not found")
        return instance_ts, instances[instance_ts]
    ts = max(instances, key=lambda t: len(instances[t]))
    return ts, instances[ts]


def _find_pass(passes: dict[str, str], pass_query: str) -> tuple[str, str]:
    """Finds a pass by substring/regex, preferring exact matches."""
    if pass_query in passes:
        return pass_query, passes[pass_query]
    q_re = re.compile(pass_query)
    matches = sorted(label for label in passes if q_re.search(label))
    if not matches:
        raise KeyError(
            f"pass {pass_query!r} not found; available: {sorted(passes)[:20]}")
    return matches[-1], passes[matches[-1]]


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def list_llo_programs(dump_dir: str = "") -> str:
    """Lists LLO programs and pass checkpoints in an --xla_jf_dump_to dir.

    **Start here for LLO dump analysis.** A dump dir holds thousands of
    per-pass files; this returns the program-level map: program name,
    classification (kernel / fusion / infrastructure), compile instances,
    and which analysis checkpoints exist (final_bundles,
    per-bundle-utilization, schedule-analysis, static gaps).

    Args:
        dump_dir: The --xla_jf_dump_to directory. Defaults to $XLA_JF_DUMP_DIR.

    Returns:
        JSON list of programs with their available checkpoints.
    """
    try:
        d = _resolve_dump_dir(dump_dir)
        if not d or not os.path.isdir(d):
            return _NO_DIR_MSG.format(d=d)
        programs = _scan(d)
        if not programs:
            return _NO_DIR_MSG.format(d=d)
        out = []
        for name, prog in sorted(programs.items()):
            ts, passes = _pick_instance(prog)
            out.append({
                "program": name,
                "kind": _classify(name),
                "compile_instances": len(prog["instances"]),
                "primary_instance_ts": ts,
                "num_passes": len(passes),
                "checkpoints": {
                    "final_bundles": any(
                        p == "final_bundles" for p in passes),
                    "per_bundle_utilization": any(
                        "per-bundle-utilization" in p for p in passes),
                    "schedule_analysis": any(
                        p.startswith("schedule-analysis") for p in passes),
                    "static_gap_analysis": any(
                        p.startswith("static_gap_analysis") for p in passes),
                },
            })
        # kernels first — they're what the caller is usually after
        out.sort(key=lambda p: (p["kind"] != "kernel", p["program"]))
        return json.dumps({"dump_dir": d, "programs": out}, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("list_llo_programs failed for %s", dump_dir)
        return f"Error in list_llo_programs: {e}"


def get_llo_schedule_analysis(
    dump_dir: str = "",
    program: str = "",
    pass_label: str = "schedule-analysis_final_bundles",
    top_n: int = 20,
) -> str:
    """Bundle-count attribution: which HLO ops / opcodes own the bundles.

    Parses a `schedule-analysis_*.txt` LLO dump into totals (scheduled /
    empty bundles) and the per-HLO-instruction and per-opcode bundle counts
    — the static "where would the cycles go" table for one LLO program.

    Args:
        dump_dir:   The --xla_jf_dump_to directory (or $XLA_JF_DUMP_DIR).
        program:    LLO program name from list_llo_programs (required).
        pass_label: Which analysis checkpoint (default: final bundles).
        top_n:      Max attribution rows returned per table.

    Returns:
        JSON: {totals, per_hlo: [...], per_opcode: [...]}.
    """
    try:
        d = _resolve_dump_dir(dump_dir)
        if not d or not os.path.isdir(d):
            return _NO_DIR_MSG.format(d=d)
        if not program:
            return "Missing required arg `program` — call list_llo_programs first."
        programs = _scan(d)
        if program not in programs:
            close = [p for p in programs if program in p][:5]
            return f"Program {program!r} not found. Close matches: {close}"
        _, passes = _pick_instance(programs[program])
        label, fname = _find_pass(passes, pass_label)
        text = open(os.path.join(d, fname), encoding="utf-8",
                    errors="replace").read()

        totals: dict = {}
        for key, pat in (
            ("total_bundles", r"total scheduled bundles:\s+(\d+)"),
            ("empty_bundles", r"empty scheduled bundles:\s+(\d+)"),
            ("non_empty_bundles", r"non empty scheduled bundles:\s+(\d+)"),
        ):
            m = re.search(pat, text)
            if m:
                totals[key] = int(m.group(1))

        per_hlo, per_opcode = [], []
        # rows: `\t 124.50 scheduled bundles (92.22%): <attribution>`
        row_re = re.compile(
            r"^\s*(?:\[opcode\]\s*)?([\d.]+)\s+scheduled bundles\s+"
            r"\(\s*([\d.]+)%\):\s+(.*)$")
        for line in text.splitlines():
            m = row_re.match(line)
            if not m:
                continue
            row = {
                "bundles": float(m.group(1)),
                "pct": float(m.group(2)),
                # cap the HLO text — full custom-call configs run to megabytes
                "attribution": m.group(3)[:300],
            }
            (per_opcode if "[opcode]" in line else per_hlo).append(row)

        return json.dumps({
            "program": program, "pass": label, "file": fname,
            "totals": totals,
            "per_hlo": per_hlo[:top_n],
            "per_opcode": per_opcode[:top_n],
        }, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("get_llo_schedule_analysis failed")
        return f"Error in get_llo_schedule_analysis: {e}"


def get_llo_static_utilization(
    dump_dir: str = "",
    program: str = "",
    pass_label: str = "per-bundle-utilization",
    hot_ranges: int = 5,
) -> str:
    """Per-unit static slot occupancy from the per-bundle utilization matrix.

    Parses `*hlo-static-per-bundle-utilization.txt`: a CAPACITY row (issue
    slots per unit) followed by one row per bundle with slots used per unit
    (MXU, XLU, VALU, VPOP, EUP, VLOAD, VLOAD:FILL, VSTORE, VSTORE:SPILL,
    SALU). Returns per-unit occupancy statistics plus the largest contiguous
    bundle ranges dominated by a single unit — static bottleneck candidates
    to inspect with get_llo_bundles.

    Args:
        dump_dir:   The --xla_jf_dump_to directory (or $XLA_JF_DUMP_DIR).
        program:    LLO program name from list_llo_programs (required).
        pass_label: Which utilization checkpoint (`pre-delay` or `final`;
                    default matches either).
        hot_ranges: How many single-unit-dominated bundle ranges to report.

    Returns:
        JSON: {capacity, units: {unit: {occupancy_pct, nonzero_bundle_pct,
        saturated_bundle_pct}}, spills/fills usage, hot_ranges}.
    """
    try:
        d = _resolve_dump_dir(dump_dir)
        if not d or not os.path.isdir(d):
            return _NO_DIR_MSG.format(d=d)
        if not program:
            return "Missing required arg `program` — call list_llo_programs first."
        programs = _scan(d)
        if program not in programs:
            close = [p for p in programs if program in p][:5]
            return f"Program {program!r} not found. Close matches: {close}"
        _, passes = _pick_instance(programs[program])
        label, fname = _find_pass(passes, pass_label)
        lines = open(os.path.join(d, fname), encoding="utf-8",
                     errors="replace").read().splitlines()

        # Header: `== CAPACTIY:` (sic — typo in libtpu), then unit names row,
        # capacities row, `== UTILIZATION:`, then one int row per bundle.
        units: list[str] = []
        capacity: list[int] = []
        rows: list[list[int]] = []
        mode = ""
        for line in lines:
            if re.match(r"==\s*CAPAC", line):
                mode = "capacity"
                continue
            if re.match(r"==\s*UTILIZATION", line):
                mode = "util"
                continue
            s = line.strip()
            if not s:
                continue
            if mode == "capacity":
                if re.match(r"^[A-Z]", s):
                    units = [u.strip() for u in s.split(",")]
                else:
                    capacity = [int(x) for x in s.split()]
            elif mode == "util":
                try:
                    rows.append([int(x) for x in s.split()])
                except ValueError:
                    continue

        if not units or not capacity or not rows:
            return (f"Could not parse utilization matrix from {fname!r} "
                    f"(units={len(units)}, capacity={len(capacity)}, "
                    f"rows={len(rows)}).")

        n = len(rows)
        result_units: dict = {}
        dominant_per_bundle: list[Optional[int]] = []
        for i, unit in enumerate(units):
            col = [r[i] for r in rows if len(r) == len(units)]
            cap = capacity[i] or 1
            occ = [c / cap for c in col]
            result_units[unit] = {
                "capacity": capacity[i],
                "occupancy_pct": round(100 * statistics.mean(occ), 2),
                "nonzero_bundle_pct": round(
                    100 * sum(1 for c in col if c) / max(1, len(col)), 2),
                "saturated_bundle_pct": round(
                    100 * sum(1 for c in col if c >= cap) / max(1, len(col)), 2),
            }
        for r in rows:
            if len(r) != len(units) or not any(r):
                dominant_per_bundle.append(None)
                continue
            dominant_per_bundle.append(max(range(len(units)), key=lambda i: r[i]))

        # largest contiguous runs dominated by one unit
        runs: list[tuple[int, int, int]] = []  # (start, end, unit_idx)
        start = None
        for idx, dom in enumerate(dominant_per_bundle + [None]):
            if start is None or dom != dominant_per_bundle[start]:
                if start is not None and dominant_per_bundle[start] is not None:
                    runs.append((start, idx - 1, dominant_per_bundle[start]))
                start = idx if dom is not None else None
        runs.sort(key=lambda r: r[1] - r[0], reverse=True)
        hot = [{
            "bundle_range": [s, e],
            "address_range": [hex(s), hex(e)],
            "length": e - s + 1,
            "dominant_unit": units[u],
        } for s, e, u in runs[:hot_ranges]]

        return json.dumps({
            "program": program, "pass": label, "file": fname,
            "num_bundles": n,
            "units": result_units,
            "hot_ranges": hot,
            "semantics": "static slot occupancy per bundle (compile-time)",
        }, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("get_llo_static_utilization failed")
        return f"Error in get_llo_static_utilization: {e}"


def get_llo_bundles(
    dump_dir: str = "",
    program: str = "",
    pass_label: str = "final_bundles",
    address_range: str = "",
    grep: str = "",
    limit: int = 100,
) -> str:
    """Windowed access to the VLIW bundle listing of an LLO program.

    Returns bundle lines from `final_bundles.txt` (or another bundle
    checkpoint): scalar/vector/DMA instructions per bundle with HLO
    attribution comments and loop-region markers. ALWAYS narrow with
    `address_range` (accepts the `0x...` addresses from Tensor Core trace
    markers and from get_llo_static_utilization hot_ranges) or `grep` —
    full listings run to thousands of bundles and will be truncated.

    Args:
        dump_dir:      The --xla_jf_dump_to directory (or $XLA_JF_DUMP_DIR).
        program:       LLO program name from list_llo_programs (required).
        pass_label:    Bundle checkpoint (default `final_bundles`).
        address_range: `0x<start>-0x<end>` or single `0x<addr>` bundle address.
        grep:          Regex over bundle text (e.g. `dma\\.hbm_to_vmem`).
        limit:         Max bundles returned (default 100, hard cap 500).

    Returns:
        JSON: {bundles: [{address, text}], truncated, total_matched}.
    """
    try:
        d = _resolve_dump_dir(dump_dir)
        if not d or not os.path.isdir(d):
            return _NO_DIR_MSG.format(d=d)
        if not program:
            return "Missing required arg `program` — call list_llo_programs first."
        limit = min(limit, 500)
        programs = _scan(d)
        if program not in programs:
            close = [p for p in programs if program in p][:5]
            return f"Program {program!r} not found. Close matches: {close}"
        _, passes = _pick_instance(programs[program])
        label, fname = _find_pass(passes, pass_label)

        lo, hi = None, None
        if address_range:
            parts = address_range.replace(" ", "").split("-")
            lo = int(parts[0], 16)
            hi = int(parts[1], 16) if len(parts) > 1 else lo
        g_re = re.compile(grep) if grep else None

        bundles: list[dict] = []
        matched = 0
        current: Optional[dict] = None
        addr_re = re.compile(r"^\s*(0x[0-9a-f]+|\d+)\s*:\s*(.*)$")
        with open(os.path.join(d, fname), encoding="utf-8",
                  errors="replace") as f:
            for line in f:
                m = addr_re.match(line)
                if m:
                    addr_s = m.group(1)
                    addr = int(addr_s, 16) if addr_s.startswith("0x") else int(addr_s)
                    current = {"address": hex(addr), "text": m.group(2).rstrip()}
                    if lo is not None and not (lo <= addr <= hi):
                        current = None
                        continue
                    if g_re and not g_re.search(line):
                        current = None
                        continue
                    matched += 1
                    if len(bundles) < limit:
                        bundles.append(current)
                elif current is not None and line.startswith((" ", "\t")):
                    # continuation lines (multi-line bundle text / hlo comments)
                    if len(current["text"]) < 2000:
                        current["text"] += " " + line.strip()

        return json.dumps({
            "program": program, "pass": label, "file": fname,
            "total_matched": matched,
            "returned": len(bundles),
            "truncated": matched > len(bundles),
            "bundles": bundles,
            "hint": ("" if matched <= limit else
                     "Narrow with address_range / grep — listing truncated."),
        }, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("get_llo_bundles failed")
        return f"Error in get_llo_bundles: {e}"
