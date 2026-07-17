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

# Reference points for the VMEM fit report. The operative limit is set by
# `--xla_tpu_scoped_vmem_limit_kib` and is NOT recorded in the LLO dump, so
# the summary reports headroom against the two campaign reference points.
_VMEM_SCOPED_DEFAULT_BYTES = 32 * 1024 * 1024   # common scoped-vmem default
_VMEM_HW_BYTES = 128 * 1024 * 1024              # v5p/v6e per-core VMEM
_UNRESOLVED_SIZE = 0xFFFFFFFFFFFFFFFF

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


def _parse_util_matrix(
    path: str,
) -> tuple[list[str], list[int], list[list[int]]]:
    """Parses a `*per-bundle-utilization.txt` matrix.

    Header: `== CAPACTIY:` (sic — typo in libtpu), then unit names row,
    capacities row, `== UTILIZATION:`, then one int row per bundle.
    Returns (units, capacity, rows).
    """
    units: list[str] = []
    capacity: list[int] = []
    rows: list[list[int]] = []
    mode = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
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
    return units, capacity, rows


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
        units, capacity, rows = _parse_util_matrix(os.path.join(d, fname))

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


# ---------------------------------------------------------------------------
# Fit summary (composition layer over the tools above)
# ---------------------------------------------------------------------------

# `#allocation3 [shape = 'u8[...]{0}', space=vmem, size = 0x300000, scoped,
#  tag = 'scoped memory for matmul_optimized.1']`
_ALLOC_RE = re.compile(
    r"#allocation(?P<id>\d+)\s*\[shape = '(?P<shape>[^']*)',\s*"
    r"space=(?P<space>\w+),\s*size = (?P<size>0x[0-9a-fA-F]+)"
    r"(?P<attrs>[^\]]*)\]"
)

# MXU mnemonics in the final bundle listing, e.g.
# `vmatmul.mubr.f32.vlgmr.msra.gmra.mxu0`, `vmatpush1.bf16.msra.mxu1`.
_MXU_MNEMONIC_RE = re.compile(
    r"\bv?(?P<op>matmul|matpush(?P<push>[12])|matprep|matpop|matres)"
    r"(?P<mods>(?:\.[a-z0-9]+)*)\b"
)
_MXU_DTYPES = {"bf16", "f32", "f16", "s8", "u8", "s16", "s32", "f8e4m3",
               "f8e5m2", "x8", "hf8", "qf8"}

def _extract_vmem_allocations(path: str) -> dict:
    """Parses `#allocation` header lines of one LLO IR pass file.

    Returns {"allocations": [...], "vmem_total_bytes", "vmem_scoped_bytes",
    "vmem_unresolved_count"} — vmem-space only, deduped by allocation id.
    """
    seen: dict[str, dict] = {}
    unresolved = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "#allocation" not in line or "[shape" not in line:
                continue
            m = _ALLOC_RE.search(line)
            if not m or m.group("space") != "vmem" or m.group("id") in seen:
                continue
            size = int(m.group("size"), 16)
            if size == _UNRESOLVED_SIZE:
                unresolved += 1
                continue
            seen[m.group("id")] = {
                "id": int(m.group("id")),
                "shape": m.group("shape"),
                "size_bytes": size,
                "scoped": "scoped" in m.group("attrs"),
                "tag": (re.search(r"tag = '([^']*)'", m.group("attrs")) or
                        [None, ""])[1],
            }
    allocs = sorted(seen.values(), key=lambda a: -a["size_bytes"])
    return {
        "allocations": allocs,
        "vmem_total_bytes": sum(a["size_bytes"] for a in allocs),
        "vmem_scoped_bytes": sum(
            a["size_bytes"] for a in allocs if a["scoped"]),
        "vmem_unresolved_count": unresolved,
    }


def _find_alloc_pass(d: str, passes: dict[str, str]) -> Optional[str]:
    """Picks the allocation-bearing IR pass file to read (latest preferred).

    Bundle listings also mention `#allocation`, so a candidate only counts
    if its head actually parses as allocation definitions (_ALLOC_RE).
    """
    preferred = ("post-ra", "pre-finalize-registers-and-allocations",
                 "original", "")
    labels = sorted(passes)  # deterministic
    candidates = []
    for want in preferred:
        candidates.extend(l for l in labels if l == want)
        candidates.extend(l for l in labels if want and want in l)
    candidates.extend(labels)
    seen = set()
    for label in candidates:
        if label in seen:
            continue
        seen.add(label)
        path = os.path.join(d, passes[label])
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                head = f.read(65536)
        except OSError:
            continue
        if _ALLOC_RE.search(head):
            return label
    return None


def _scan_mxu_mnemonics(path: str) -> dict:
    """Counts MXU mnemonics + dtype modifiers in a bundle listing."""
    ops: dict[str, int] = collections.defaultdict(int)
    dtypes: dict[str, int] = collections.defaultdict(int)
    push_kinds: set[str] = set()
    mxus: set[str] = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "mat" not in line:
                continue
            for m in _MXU_MNEMONIC_RE.finditer(line):
                op = m.group("op")
                ops[op] += 1
                if m.group("push"):
                    push_kinds.add(m.group("push"))
                mods = set(m.group("mods").strip(".").split("."))
                for dt in mods & _MXU_DTYPES:
                    if op.startswith("matmul") or op.startswith("matpush"):
                        dtypes[dt] += 1
                for unit in mods & {"mxu0", "mxu1", "mxu2", "mxu3"}:
                    mxus.add(unit)
    matmuls = sum(v for k, v in ops.items() if k.startswith("matmul"))
    width = None
    if ops:
        # Heuristic (v5p/v6e): operands are staged with vmatpush1/vmatpush2
        # (two 128-lane halves) when the matmul consumes the full 256-lane
        # tile; a 128-lane (half-width) matmul only ever issues matpush1.
        width = 256 if {"1", "2"} <= push_kinds else 128
    return {
        "matmul_count": matmuls,
        "op_counts": dict(sorted(ops.items())),
        "dtypes": dict(sorted(dtypes.items())),
        "mxus_used": sorted(mxus),
        "inferred_operand_lanes": width,
        "lane_inference": (
            "" if width is None else
            ("matpush1+matpush2 both present -> full 256-lane tile staging"
             if width == 256 else
             "only matpush1 present -> 128-lane (half-width) staging")),
    }




def _pct_of(val: float, ref: float) -> float:
    return round(100.0 * val / ref, 1) if ref else 0.0


def _mib(bytes_val: int) -> float:
    return round(bytes_val / (1024 * 1024), 2)


def _vmem_fit(d: str, passes: dict[str, str]) -> dict:
    """Section 1: VMEM allocation vs the applicable limit + headroom."""
    alloc_label = _find_alloc_pass(d, passes)
    if alloc_label is None:
        return {}
    vmem = _extract_vmem_allocations(os.path.join(d, passes[alloc_label]))
    vmem["from_pass"] = alloc_label or "(unlabeled)"
    total = vmem["vmem_total_bytes"]
    # The operative limit is --xla_tpu_scoped_vmem_limit_kib; it is NOT
    # recorded in the dump, so headroom is reported against both campaign
    # reference points (the false-wall killer: authors assume the 128 MiB
    # hardware VMEM when the 32 MiB scoped default is what actually binds).
    vmem["limit_scoped_default_bytes"] = _VMEM_SCOPED_DEFAULT_BYTES
    vmem["limit_hardware_bytes"] = _VMEM_HW_BYTES
    vmem["used_of_scoped_default_pct"] = _pct_of(
        total, _VMEM_SCOPED_DEFAULT_BYTES)
    vmem["headroom_vs_scoped_default_pct"] = round(
        100.0 - vmem["used_of_scoped_default_pct"], 1)
    vmem["used_of_hardware_pct"] = _pct_of(total, _VMEM_HW_BYTES)
    vmem["headroom_vs_hardware_pct"] = round(
        100.0 - vmem["used_of_hardware_pct"], 1)
    vmem["limit_note"] = (
        "operative limit = --xla_tpu_scoped_vmem_limit_kib (not recorded "
        "in the dump); 32 MiB scoped default unless overridden, 128 MiB "
        "hardware VMEM")
    return vmem


def _spill_metrics(units: list[str], rows: list[list[int]]) -> dict:
    """Section: register spill/fill traffic from the utilization matrix."""
    n_units = len(units)
    idx = {u: i for i, u in enumerate(units)}
    spills = fills = 0
    n = 0
    for r in rows:
        if len(r) != n_units:
            continue
        n += 1
        if "VSTORE:SPILL" in idx:
            spills += r[idx["VSTORE:SPILL"]]
        if "VLOAD:FILL" in idx:
            fills += r[idx["VLOAD:FILL"]]
    rate = round((spills + fills) / n, 2) if n else 0.0
    out = {
        "spills_total": spills,
        "fills_total": fills,
        "bundles": n,
        "spill_fill_per_bundle": rate,
    }
    if rate > 0.5:
        out["flag"] = (
            "REGISTER-PRESSURE: VPU-active% is inflated by spill traffic "
            "— reduce live vectors (less unroll / split accumulator chain)")
    return out


def _bundle_hlo_map(path: str) -> dict[int, str]:
    """One pass over a bundle listing: bundle address -> HLO attribution.

    Only some bundles carry an explicit `hlo:` comment; the rest inherit
    the nearest previous attribution (region-local approximation).
    """
    hlo_re = re.compile(r"hlo: ([^\s*/]+)")
    addr_re = re.compile(r"^\s*(0x[0-9a-f]+|\d+)\s*:")
    out: dict[int, str] = {}
    addr = None
    last = "?"
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = addr_re.match(line)
            if m:
                a = m.group(1)
                addr = int(a, 16) if a.startswith("0x") else int(a)
                out[addr] = last
            if addr is None:
                continue
            h = hlo_re.search(line)
            if h:
                last = h.group(1)
                out[addr] = last
    return out


# Bundle classes. NOTE: "mem" here is vector VMEM<->register traffic
# (VLOAD/VSTORE issue slots); asynchronous DMA engines are not visible in
# the static matrix — a mem-stall run is where the *schedule* has the MXU
# idle while memory traffic (or nothing) issues.
_STALL_CLASSES = ("mem_stall_mxu_idle", "vpu_only_mxu_idle", "salu_bubble",
                  "idle")


def _timeline(units: list[str], rows: list[list[int]],
              hlo_map: dict[int, str], top_stalls: int) -> dict:
    """Section: per-bundle pipeline classification + longest stall runs."""
    n_units = len(units)
    idx = {u: i for i, u in enumerate(units)}

    def slot(r, u):
        return r[idx[u]] if u in idx else 0

    classes: list[str] = []
    for r in rows:
        if len(r) != n_units:
            classes.append("idle")
            continue
        mxu = slot(r, "MXU") > 0
        mem = (slot(r, "VLOAD") + slot(r, "VSTORE") + slot(r, "VLOAD:FILL")
               + slot(r, "VSTORE:SPILL")) > 0
        vpu = (slot(r, "VALU") + slot(r, "VPOP") + slot(r, "EUP")
               + slot(r, "XLU")) > 0
        salu = slot(r, "SALU") > 0
        if mxu and mem:
            classes.append("overlap_mxu_mem")
        elif mxu:
            classes.append("mxu_only")
        elif mem:
            classes.append("mem_stall_mxu_idle")
        elif vpu:
            classes.append("vpu_only_mxu_idle")
        elif salu:
            classes.append("salu_bubble")
        else:
            classes.append("idle")

    n = len(classes) or 1
    counts = collections.Counter(classes)
    pct = {c: _pct_of(counts.get(c, 0), n) for c in (
        "overlap_mxu_mem", "mxu_only", "mem_stall_mxu_idle",
        "vpu_only_mxu_idle", "salu_bubble", "idle")}

    # longest contiguous MXU-idle runs
    runs: list[dict] = []
    start = None
    for i, c in enumerate(classes + ["mxu_only"]):
        stalled = c in _STALL_CLASSES
        if stalled and start is None:
            start = i
        elif not stalled and start is not None:
            seg = classes[start:i]
            hlos = collections.Counter(
                hlo_map.get(b, "?") for b in range(start, i))
            runs.append({
                "bundle_range": [hex(start), hex(i - 1)],
                "length": i - start,
                "dominant_class": collections.Counter(seg).most_common(1)[0][0],
                "hlo": hlos.most_common(1)[0][0],
            })
            start = None
    runs.sort(key=lambda r: -r["length"])
    return {
        "bundle_class_pct": pct,
        "mxu_idle_stall_pct": round(
            sum(pct[c] for c in _STALL_CLASSES), 1),
        "top_stall_runs": runs[:top_stalls],
        "semantics": ("static issue-slot co-residency per bundle; async "
                      "DMA engines are not visible — 'mem' means "
                      "VLOAD/VSTORE slots"),
    }


def _recommendations(vmem: dict, mxu: dict, spill: dict, tl: dict) -> list[dict]:
    """Ranked (lever, expected ceiling) list — the feedback-loop payload."""
    recs: list[dict] = []

    def ceiling(stall_pct: float) -> str:
        if stall_pct >= 99:
            return "unbounded"
        return f"<={1.0 / (1.0 - stall_pct / 100.0):.2f}x"

    pct = tl.get("bundle_class_pct", {})
    headroom = vmem.get("headroom_vs_scoped_default_pct")

    if spill.get("spill_fill_per_bundle", 0) > 0.5:
        recs.append({
            "kind": "STRUCTURAL",
            "lever": ("restructure the live-register set: less unroll / "
                      "split accumulator chain (spill traffic is not a "
                      "tunable)"),
            "ceiling_estimate": "structural — spills inflate VPU-active% "
                                "and steal VSTORE/VLOAD slots",
            "basis": (f"{spill['spills_total']} spills + "
                      f"{spill['fills_total']} fills over "
                      f"{spill['bundles']} bundles "
                      f"({spill['spill_fill_per_bundle']}/bundle)"),
        })

    mem_stall = pct.get("mem_stall_mxu_idle", 0)
    if mem_stall >= 5:
        lever = ("overlap memory traffic with MXU: deepen multi-buffering "
                 "/ prefetch")
        if headroom is not None and headroom >= 25:
            lever += (f" (VMEM headroom {headroom}% of scoped default "
                      "allows deeper buffering)")
        recs.append({
            "kind": "TUNE", "lever": lever,
            "ceiling_estimate": ceiling(mem_stall),
            "basis": f"eliminate {mem_stall}% MXU-idle memory-stall bundles",
        })

    salu = pct.get("salu_bubble", 0)
    if salu >= 5:
        recs.append({
            "kind": "TUNE",
            "lever": ("hoist scalar address computation / batch DMA "
                      "descriptors out of the inner loop"),
            "ceiling_estimate": ceiling(salu),
            "basis": f"eliminate {salu}% scalar-only bubbles",
        })

    vpu = pct.get("vpu_only_mxu_idle", 0)
    if vpu >= 5 and mxu.get("matmul_count"):
        recs.append({
            "kind": "TUNE",
            "lever": ("schedule vector-only prologue/epilogue work under "
                      "the matmul pipeline"),
            "ceiling_estimate": ceiling(vpu),
            "basis": f"{vpu}% bundles are VPU-only with MXU idle",
        })

    idle = pct.get("idle", 0)
    if idle >= 5:
        recs.append({
            "kind": "TUNE",
            "lever": "scheduler bubbles: restructure dependency chains / "
                     "adjust unroll",
            "ceiling_estimate": ceiling(idle),
            "basis": f"{idle}% fully-idle bundles",
        })

    if mxu.get("matmul_count") and mxu.get("inferred_operand_lanes") == 128:
        recs.append({
            "kind": "STRUCTURAL",
            "lever": "feed full 256-lane MXU tiles (operand staging is "
                     "half-width)",
            "ceiling_estimate": "<=2.0x",
            "basis": "only matpush1 observed — 128-lane staging",
        })

    def rank(r: dict):
        # spill-pressure headline first, then numeric ceilings descending.
        if r["kind"] == "STRUCTURAL" and "spill" in r["basis"]:
            return (0, 0.0)
        m = re.search(r"([\d.]+)x", r["ceiling_estimate"])
        return (1, -float(m.group(1)) if m else 0.0)

    recs.sort(key=rank)
    return recs


def _fit_data(d: str, program: str, top_stalls: int) -> dict:
    """Computes the full fit-summary payload for one dump dir."""
    programs = _scan(d)
    if not programs:
        raise FileNotFoundError(_NO_DIR_MSG.format(d=d))
    if not program:
        kernels = {n: p for n, p in programs.items()
                   if _classify(n) == "kernel"}
        pool = kernels or {n: p for n, p in programs.items()
                           if _classify(n) == "fusion"} or programs
        program = max(pool, key=lambda n: len(_pick_instance(pool[n])[1]))
    if program not in programs:
        close = [p for p in programs if program in p][:5]
        raise KeyError(f"Program {program!r} not found. Close matches: {close}")
    _, passes = _pick_instance(programs[program])

    data: dict = {"dump_dir": d, "program": program,
                  "kind": _classify(program)}
    data["other_programs"] = sorted(
        n for n in programs
        if n != program and _classify(n) != "infrastructure")

    data["vmem"] = _vmem_fit(d, passes)
    mxu: dict = {}
    hlo_map: dict[int, str] = {}
    try:
        _, bundles_fname = _find_pass(passes, "final_bundles")
        bundles_path = os.path.join(d, bundles_fname)
        mxu = _scan_mxu_mnemonics(bundles_path)
        hlo_map = _bundle_hlo_map(bundles_path)
    except KeyError:
        pass
    data["mxu"] = mxu

    units_stats: dict = {}
    tl: dict = {}
    spill: dict = {}
    try:
        _, util_fname = _find_pass(passes, "per-bundle-utilization")
        units, capacity, rows = _parse_util_matrix(
            os.path.join(d, util_fname))
        if units and capacity and rows:
            cap = {u: capacity[i] or 1 for i, u in enumerate(units)}
            for i, u in enumerate(units):
                col = [r[i] for r in rows if len(r) == len(units)]
                units_stats[u] = {
                    "capacity": cap[u],
                    "occupancy_pct": round(
                        100 * statistics.mean(c / cap[u] for c in col), 2)
                    if col else 0.0,
                    "nonzero_bundle_pct": round(
                        100 * sum(1 for c in col if c) / max(1, len(col)), 2),
                }
            spill = _spill_metrics(units, rows)
            tl = _timeline(units, rows, hlo_map, top_stalls)
    except KeyError:
        pass
    data["units"] = units_stats
    data["spills"] = spill
    data["timeline"] = tl

    sched = json.loads(get_llo_schedule_analysis(d, program, top_n=3))
    data["totals"] = sched.get("totals", {}) if isinstance(sched, dict) else {}

    recs = _recommendations(data["vmem"], mxu, spill, tl)
    data["recommendations"] = recs

    # -- verdict (human line + machine-readable class) ----------------------
    occ = {u: v.get("occupancy_pct", 0.0) for u, v in units_stats.items()
           if ":" not in u}
    verdict = "no signal (no utilization matrix in dump)"
    if occ:
        dominant = max(occ, key=occ.get)
        verdict = {
            "MXU": "compute-bound (MXU)",
            "VLOAD": "DMA/memory-bound (vector loads dominate)",
            "VSTORE": "DMA/memory-bound (vector stores dominate)",
            "SALU": "scalar-serialized (SALU dominates)",
            "VALU": "VPU-serialized (vector ALU dominates, MXU starved)",
            "VPOP": "VPU-serialized (vector pop dominates)",
            "EUP": "VPU-serialized (transcendentals dominate)",
            "XLU": "XLU-serialized (transposes/permutes dominate)",
        }.get(dominant, f"{dominant}-dominated")
        verdict += f" — {dominant} {occ[dominant]}% static occupancy"
    if data["vmem"].get("used_of_scoped_default_pct", 0) >= 90.0:
        verdict = (f"VMEM-capped "
                   f"({data['vmem']['used_of_scoped_default_pct']}% of the "
                   f"32 MiB scoped default) — then " + verdict)
    data["verdict"] = verdict
    if recs:
        data["verdict_class"] = {
            "class": recs[0]["kind"],
            "top_lever": recs[0]["lever"],
            "ceiling_estimate": recs[0]["ceiling_estimate"],
        }
    else:
        data["verdict_class"] = {
            "class": "AT-CEILING",
            "top_lever": None,
            "ceiling_estimate": "~1.0x (no stall lever above threshold)",
        }
    data["caveat"] = (
        "static compile-time analysis (issue slots + allocation sizes), "
        "not measured time; verify with get_llo_utilization on a "
        "kernel-profiling trace")
    return data


_DIFF_SCALARS = (
    ("vmem_total_mib", lambda x: _mib(x["vmem"].get("vmem_total_bytes", 0))),
    ("spills", lambda x: x["spills"].get("spills_total", 0)),
    ("fills", lambda x: x["spills"].get("fills_total", 0)),
    ("spill_fill_per_bundle",
     lambda x: x["spills"].get("spill_fill_per_bundle", 0)),
    ("mxu_occupancy_pct",
     lambda x: x["units"].get("MXU", {}).get("occupancy_pct", 0)),
    ("mxu_idle_stall_pct",
     lambda x: x["timeline"].get("mxu_idle_stall_pct", 0)),
    ("total_bundles", lambda x: x["totals"].get("total_bundles", 0)),
)


def _fit_digest(data: dict, top_stalls: int) -> list[str]:
    """Renders the payload as a <=30-line markdown digest."""
    vmem, mxu = data["vmem"], data["mxu"]
    units, spill, tl = data["units"], data["spills"], data["timeline"]
    lines = [f"# LLO fit summary — {data['program']} ({data['kind']})"]

    if vmem:
        lines.append(
            f"VMEM: {_mib(vmem['vmem_total_bytes'])} MiB allocated "
            f"({len(vmem['allocations'])} allocs, pass {vmem['from_pass']})"
            f" = {vmem['used_of_scoped_default_pct']}% of 32 MiB scoped "
            f"default ({vmem['headroom_vs_scoped_default_pct']}% headroom)"
            f", {vmem['used_of_hardware_pct']}% of 128 MiB HW "
            f"({vmem['headroom_vs_hardware_pct']}% headroom)")
        lines.append("  top allocs: " + ", ".join(
            f"{_mib(a['size_bytes'])} MiB {a['tag'] or a['shape']}"
            for a in vmem["allocations"][:3]))
    else:
        lines.append("VMEM: n/a (no allocation-bearing IR pass in dump — "
                     "capture without pass-label filtering)")

    if mxu.get("matmul_count"):
        lines.append(
            f"MXU: {mxu['matmul_count']} matmuls on "
            f"{'+'.join(mxu['mxus_used']) or 'mxu?'}, dtypes "
            f"{'/'.join(mxu['dtypes']) or 'n/a'}, operand width "
            f"~{mxu['inferred_operand_lanes']} lanes "
            f"({mxu['lane_inference']})")
    else:
        lines.append("MXU: no matmul mnemonics in final bundles")

    if units:
        lines.append("Unit split (static occupancy): " + "  ".join(
            f"{u}={units[u]['occupancy_pct']}%" for u in
            ("MXU", "VALU", "EUP", "VLOAD", "VSTORE", "SALU") if u in units))

    if spill:
        s = (f"Spills: {spill['spills_total']} spills + "
             f"{spill['fills_total']} fills over {spill['bundles']} bundles "
             f"({spill['spill_fill_per_bundle']}/bundle)")
        lines.append(s)
        if spill.get("flag"):
            lines.append(f"  {spill['flag']}")

    if tl:
        p = tl["bundle_class_pct"]
        lines.append(
            f"Timeline: overlap(MXU+mem)={p['overlap_mxu_mem']}%  "
            f"MXU-only={p['mxu_only']}%  mem-stall(MXU idle)="
            f"{p['mem_stall_mxu_idle']}%  VPU-only={p['vpu_only_mxu_idle']}%"
            f"  SALU-bubble={p['salu_bubble']}%  idle={p['idle']}%")
        if tl["top_stall_runs"]:
            lines.append(f"Top stall runs (max {top_stalls}):")
            for r in tl["top_stall_runs"]:
                lines.append(
                    f"  - bundles {r['bundle_range'][0]}-"
                    f"{r['bundle_range'][1]} (len {r['length']}): "
                    f"{r['dominant_class']} — hlo: {r['hlo']}")

    recs = data["recommendations"]
    if recs:
        lines.append("Recommendations (ranked):")
        for i, r in enumerate(recs[:4], 1):
            lines.append(f"  {i}. [{r['kind']}] {r['lever']} -> "
                         f"{r['ceiling_estimate']} ({r['basis']})")

    vc = data["verdict_class"]
    lines.append(f"Verdict: {data['verdict']} [class={vc['class']} "
                 f"ceiling={vc['ceiling_estimate']}]")
    lines.append(f"Caveat: {data['caveat']}.")
    if data["other_programs"]:
        o = data["other_programs"]
        lines.append("Other programs in dump: " + ", ".join(o[:4])
                     + (" ..." if len(o) > 4 else ""))
    if "diff" in data:
        diff = data["diff"]
        if "deltas" in diff:
            deltas = ", ".join(
                f"{k} {v['before']} -> {v['after']}"
                for k, v in diff["deltas"].items() if v["changed"])
            lines.append(
                f"Diff vs {diff['other_program']} in "
                f"{diff['other_dump_dir']}: "
                + (deltas or "no scalar changes"))
        else:
            lines.append(f"Diff vs {diff['other_dump_dir']}: "
                         f"{diff['error']}")
    return lines[:30]


def get_llo_fit_summary(
    dump_dir: str = "",
    program: str = "",
    top_stalls: int = 5,
    diff_dump_dir: str = "",
) -> str:
    """Kernel fit digest of an `--xla_jf_dump_to` LLO dump (~30 lines).

    The composition layer over the other get_llo_* tools, designed to be
    pasted into a kernel author's context:

      1. per-program VMEM allocation vs the applicable limit (32 MiB
         scoped default unless overridden / 128 MiB hardware) + headroom;
      2. MXU matmul width observed (128 vs 256 lanes) + dtypes;
      3. per-unit static occupancy split (MXU/VPU/EUP/loads/stores/SALU);
      4. register spill/fill totals + per-bundle rate (REGISTER-PRESSURE
         flag above 0.5/bundle) — invisible to xprof/HLO;
      5. pipeline timeline classification (overlap(MXU+mem) / MXU-only /
         mem-stall / VPU-only / SALU-bubble / idle) + longest stall runs
         with HLO attribution;
      6. ranked recommendations with ceiling estimates and a
         machine-readable verdict_class {class: STRUCTURAL|TUNE|AT-CEILING,
         top_lever, ceiling_estimate} for worker loops to branch on.

    CAVEAT — everything here is STATIC (compile-time issue slots and
    allocation sizes), not measured hardware time. Use the trace-side
    get_llo_utilization / get_kernel_stage_breakdown for measured
    alignment, and never rank candidates by these numbers alone.

    Args:
        dump_dir:      The --xla_jf_dump_to directory. Defaults to
                       $XLA_JF_DUMP_DIR.
        program:       LLO program name (from list_llo_programs). Default:
                       the kernel-classified program with the most passes.
        top_stalls:    Max stall-run rows reported.
        diff_dump_dir: Optional second dump dir; appends a scalar delta
                       report (spills / stalls / alloc / util) — the
                       iteration feedback loop between two candidates.

    Returns:
        Markdown digest (<= 30 lines) followed by a fenced full-JSON block.
    """
    try:
        d = _resolve_dump_dir(dump_dir)
        if not d or not os.path.isdir(d):
            return _NO_DIR_MSG.format(d=d)
        try:
            data = _fit_data(d, program, top_stalls)
        except FileNotFoundError as e:
            return str(e)
        except KeyError as e:
            return str(e.args[0])

        if diff_dump_dir:
            d2 = os.path.expanduser(diff_dump_dir)
            try:
                try:
                    # same program when the other dump has it ...
                    other = _fit_data(d2, data["program"], top_stalls)
                except KeyError:
                    # ... else the other dump's own default program
                    other = _fit_data(d2, "", top_stalls)
                deltas = {}
                for name, fn in _DIFF_SCALARS:
                    before, after = fn(other), fn(data)
                    deltas[name] = {"before": before, "after": after,
                                    "changed": before != after}
                data["diff"] = {
                    "other_dump_dir": d2,
                    "other_program": other["program"],
                    "direction": "before=diff_dump_dir, after=dump_dir",
                    "deltas": deltas,
                }
            except Exception as e:  # pylint: disable=broad-exception-caught
                data["diff"] = {"other_dump_dir": d2,
                                "error": f"{type(e).__name__}: {e}"}

        lines = _fit_digest(data, top_stalls)
        return ("\n".join(lines) + "\n\n```json\n"
                + json.dumps(data, indent=2) + "\n```")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("get_llo_fit_summary failed for %s", dump_dir)
        return f"Error in get_llo_fit_summary: {e}"
