"""Mosaic MLIR extraction for Pallas custom calls — kernel-level noop audit.

Answers "what did the compiler actually lower for this Pallas kernel" —
verifying tiling / fusion / DMA structure matches the optimization plan
(the kernel-level analog of the HLO silent-noop audit). Two sources:

1. **Mosaic dump dir** (`--xla_mosaic_dump_to=<dir>` in LIBTPU_INIT_ARGS):
   textual MLIR per Mosaic pass, files named
   `<epoch-ns>-<NNNN>-mosaic-dump-<kernel>-<pass>.txt`. Richest source;
   needs the extra flag at capture time.

2. **HLO dump dir** (the standard `--xla_dump_to`): the custom-call op's
   `backend_config` carries `custom_call_config.body` — base64-encoded MLIR
   *bytecode* (not text). Full text rendering would need mlir tooling, so
   this path returns a structural summary (op mnemonics + named scopes
   recovered from the bytecode's string table). Works with captures the
   optimization loop already does.
"""

import base64
import collections
import json
import logging
import os
import re
from typing import Optional

_MOSAIC_FNAME_RE = re.compile(
    r"^(?P<ts>\d{10,})-(?P<seq>\d{4})-mosaic-dump-(?P<name>.+?)-(?P<pass>[\w\-]+)\.txt$"
)

# tpu-dialect / vector / dma mnemonics worth counting in a structural summary.
_MLIR_OPS_OF_INTEREST = (
    "tpu.matmul", "tpu.enqueue_dma", "tpu.wait_dma", "tpu.wait_dma2",
    "tpu.memref_slice", "tpu.vector_store", "vector.load", "vector.shape_cast",
    "tpu.sem_alloc", "tpu.region", "tpu.trace_start", "tpu.trace_stop",
    "scf.for", "scf.if", "arith.addf", "arith.mulf",
)


def _summarize_mlir_text(text: str) -> dict:
    counts = {op: text.count(op) for op in _MLIR_OPS_OF_INTEREST}
    return {op: c for op, c in counts.items() if c}


def _summarize_bytecode(raw: bytes) -> dict:
    """Structural summary of MLIR bytecode via its embedded string table."""
    strings = re.findall(rb"[\x20-\x7e]{4,}", raw)
    decoded = [s.decode("ascii", errors="replace") for s in strings]
    op_counts: dict[str, int] = collections.Counter()
    scopes: set[str] = set()
    for s in decoded:
        for op in _MLIR_OPS_OF_INTEREST:
            if s == op or s.startswith(op):
                op_counts[op] += 1
        # named scopes / stage labels look like `acc/dot_general`, `ep_copy_in/...`
        if re.fullmatch(r"[\w.]+/[\w./]+", s) and len(s) < 80:
            scopes.add(s)
    version = next((s for s in decoded if s.startswith("MLIR")), "")
    return {
        "mlir_version": version,
        "op_counts": dict(op_counts),
        "named_scopes": sorted(scopes)[:40],
    }


def get_custom_call_mlir(
    kernel: str = "",
    mosaic_dump_dir: str = "",
    hlo_dump_dir: str = "",
    pass_label: str = "",
    max_chars: int = 20000,
) -> str:
    """Returns the lowered Mosaic MLIR for a Pallas custom call.

    The kernel-level silent-noop audit: verify the compiler actually emitted
    the tiling / DMA / matmul structure the optimization plan intended.
    Prefers `mosaic_dump_dir` (textual MLIR per pass, from
    `--xla_mosaic_dump_to`); falls back to decoding the base64
    `custom_call_config.body` (MLIR bytecode) from an HLO dump, returning a
    structural summary (op counts + named scopes) instead of full text.

    Args:
        kernel:          Kernel name filter (matched against dump filenames /
                         HLO custom-call op names). Empty = first found.
        mosaic_dump_dir: `--xla_mosaic_dump_to` directory (textual MLIR).
        hlo_dump_dir:    `--xla_dump_to` directory (bytecode fallback);
                         defaults to $XLA_HLO_DUMP_DIR if neither dir given.
        pass_label:      Mosaic pass to return (default: last pass — the
                         closest to what LLO consumes).
        max_chars:       Cap on returned MLIR text (default 20000).

    Returns:
        JSON with either {source: "mosaic_dump", pass, mlir_text (capped),
        op_summary} or {source: "hlo_backend_config", bytecode_summary}.
    """
    try:
        if mosaic_dump_dir and os.path.isdir(os.path.expanduser(mosaic_dump_dir)):
            return _from_mosaic_dump(
                os.path.expanduser(mosaic_dump_dir), kernel, pass_label, max_chars)

        d = hlo_dump_dir or os.environ.get("XLA_HLO_DUMP_DIR", "")
        d = os.path.expanduser(d) if d else ""
        if d and os.path.isdir(d):
            return _from_hlo_dump(d, kernel)

        return (
            "No dump directory available. Provide `mosaic_dump_dir` (capture "
            "with LIBTPU_INIT_ARGS=\"--xla_mosaic_dump_to=<dir>\" — textual "
            "MLIR) or `hlo_dump_dir` / $XLA_HLO_DUMP_DIR (XLA_FLAGS="
            "\"--xla_dump_to=<dir>\" — bytecode summary)."
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("get_custom_call_mlir failed")
        return f"Error in get_custom_call_mlir: {e}"


def _from_mosaic_dump(
    d: str, kernel: str, pass_label: str, max_chars: int
) -> str:
    groups: dict[tuple[int, str], dict[int, tuple[str, str]]] = (
        collections.defaultdict(dict))
    for fname in os.listdir(d):
        m = _MOSAIC_FNAME_RE.match(fname)
        if not m:
            continue
        if kernel and kernel not in m.group("name") and kernel not in fname:
            continue
        groups[(int(m.group("ts")), m.group("name"))][int(m.group("seq"))] = (
            m.group("pass"), fname)
    if not groups:
        return (f"No mosaic dump files matched kernel={kernel!r} in {d!r}. "
                "Files are named <ts>-<NNNN>-mosaic-dump-<kernel>-<pass>.txt.")

    # pick the compile instance with the most passes
    key = max(groups, key=lambda k: len(groups[k]))
    passes = groups[key]
    if pass_label:
        candidates = {seq: pf for seq, pf in passes.items()
                      if re.search(pass_label, pf[0])}
        if not candidates:
            return (f"Pass {pass_label!r} not found; available: "
                    f"{sorted(p for p, _ in passes.values())}")
        seq = max(candidates)
        label, fname = candidates[seq]
    else:
        seq = max(passes)
        label, fname = passes[seq]

    text = open(os.path.join(d, fname), encoding="utf-8",
                errors="replace").read()
    return json.dumps({
        "source": "mosaic_dump",
        "kernel": key[1],
        "pass": label,
        "pass_seq": seq,
        "available_passes": [p for _, (p, _f) in sorted(passes.items())],
        "op_summary": _summarize_mlir_text(text),
        "mlir_chars": len(text),
        "mlir_text": text[:max_chars],
        "truncated": len(text) > max_chars,
    }, indent=2)


def _from_hlo_dump(d: str, kernel: str) -> str:
    body_re = re.compile(r'"body"\s*:\s*"([A-Za-z0-9+/=]+)"')
    # prefer post-optimization HLO text files
    candidates = sorted(
        f for f in os.listdir(d)
        if "after_optimizations" in f and f.endswith((".txt", ".hlo")))
    candidates += sorted(
        f for f in os.listdir(d) if f.endswith((".txt", ".hlo"))
        and f not in candidates)
    for fname in candidates:
        path = os.path.join(d, fname)
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except (IsADirectoryError, OSError):
            continue
        if "custom-call" not in text or "custom_call_config" not in text:
            continue
        for line in text.splitlines():
            if "custom_call_config" not in line:
                continue
            if kernel and kernel not in line:
                continue
            m = body_re.search(line)
            if not m:
                continue
            op_m = re.search(r"%([\w.\-]+)\s*=[^=]*custom-call", line)
            raw = base64.b64decode(m.group(1))
            return json.dumps({
                "source": "hlo_backend_config",
                "hlo_file": fname,
                "kernel": op_m.group(1) if op_m else "",
                "body_format": "mlir_bytecode",
                "body_bytes": len(raw),
                "bytecode_summary": _summarize_bytecode(raw),
                "note": ("body is MLIR bytecode; for full textual MLIR "
                         "re-capture with --xla_mosaic_dump_to and pass "
                         "mosaic_dump_dir."),
            }, indent=2)
    return (f"No custom-call with custom_call_config found in {d!r}"
            + (f" matching kernel={kernel!r}" if kernel else "")
            + ". Is this an --xla_dump_to dir with HLO text dumps?")
