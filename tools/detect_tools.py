"""Automated bottleneck detectors.

detect_unfused_reshapes is adapted from openxla/xprof
plugin/xprof/cli/tools/detect_unfused_reshapes_tool.py (Apache-2.0),
rewired onto this repo's tool functions. It automates the "materialized
broadcast / standalone formatting op" gotcha: a reshape/transpose/copy
that escaped fusion and feeds a compute op forces an explicit HBM
intermediate — restructure so it lands inside the consumer's fusion.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

_FORMATTING_CATEGORIES = ("data formatting", "copy", "reshape", "transpose")
_FORMATTING_NAME_KEYS = ("reshape", "transpose", "copy")
_COMPUTE_OPS = ("dot", "einsum", "custom-call", "fusion")


def detect_unfused_reshapes(run: str, limit: int = 50) -> str:
    """Detects unfused reshape/transpose/copy ops causing HBM materialization.

    Scans the top ops by bytes accessed for standalone formatting ops
    (not inside a fusion) whose direct consumer is a compute op — the
    signature of an explicit HBM intermediate the compiler failed to fold.

    Args:
        run: Profile run name (see list_runs).
        limit: How many top-by-bytes ops to analyze.

    Returns:
        JSON: {bottlenecks_found, inefficient_ops (each with a
        recommendation), message}.
    """
    from xprof_mcp.internal import hlo_tools  # pylint: disable=g-import-not-at-top
    from xprof_mcp.tools import get_top_hlo_ops_tool  # pylint: disable=g-import-not-at-top

    try:
        ops_data = json.loads(
            get_top_hlo_ops_tool.get_top_hlo_ops(run, limit=limit)
        )
        if "error" in ops_data:
            return json.dumps(
                {"error": ops_data["error"], "tool": "detect_unfused_reshapes"},
                indent=2,
            )
        top_by_bytes = ops_data.get("top_by_bytes_accessed", [])

        candidates = []
        for op in top_by_bytes:
            category = str(op.get("category", "")).lower()
            name = str(op.get("name", "")).lower()
            if any(c in category for c in _FORMATTING_CATEGORIES) or any(
                k in name for k in _FORMATTING_NAME_KEYS
            ):
                candidates.append(op)

        if not candidates:
            return json.dumps(
                {
                    "bottlenecks_found": False,
                    "message": "No formatting operations found.",
                    "inefficient_ops": [],
                },
                indent=2,
            )

        inefficient_ops = []
        module_names_cache = None

        for candidate in candidates:
            raw_name = str(candidate.get("name", ""))
            instr_name = (
                raw_name.split("/")[-1]
                .split(" and its ")[0]
                .replace("%", "")
                .strip()
            )
            parts = raw_name.split("/")
            mod_name = parts[1] if len(parts) > 1 else None

            neighborhood = ""
            found = False
            if mod_name:
                neighborhood = hlo_tools.get_hlo_neighborhood(
                    run,
                    instruction_name=instr_name,
                    radius=2,
                    module_name=mod_name,
                )
                found = "not found" not in neighborhood.lower()

            if not found:
                if module_names_cache is None:
                    modules_str = hlo_tools.list_hlo_modules(run)
                    module_names_cache = re.findall(
                        r"^\s*\d+\.\s+([A-Za-z0-9._-]+)\(",
                        modules_str,
                        re.MULTILINE,
                    )
                for name_only in module_names_cache:
                    neighborhood = hlo_tools.get_hlo_neighborhood(
                        run,
                        instruction_name=instr_name,
                        radius=2,
                        module_name=name_only,
                    )
                    if "not found" not in neighborhood.lower():
                        found = True
                        break

            if not found:
                continue

            is_standalone = False
            feeds_compute = False
            compute_target = None
            for line in neighborhood.splitlines():
                line_lower = line.lower()
                if f"%{instr_name} = " in line_lower:
                    comp_context = line_lower.split(f"%{instr_name} = ")[0]
                    if "[" in comp_context and "]" in comp_context:
                        comp_name = (
                            comp_context.split("[")[-1].split("]")[0].strip()
                        )
                        if not any(
                            k in comp_name
                            for k in ("fused_computation", "fusion")
                        ):
                            is_standalone = True
                    elif not any(
                        k in line_lower for k in ("fused_computation", "fusion")
                    ):
                        is_standalone = True
                elif "[dist=1]" in line_lower and f"%{instr_name}" in line_lower:
                    target = next(
                        (op for op in _COMPUTE_OPS if op in line_lower), None
                    )
                    if target:
                        feeds_compute = True
                        compute_target = target
                        break

            if is_standalone and feeds_compute:
                candidate["hbm_materialization_overhead"] = True
                candidate["downstream_compute"] = compute_target
                candidate["recommendation"] = (
                    f"Standalone formatting op '{instr_name}' feeds compute op"
                    f" '{compute_target}'. This forces an explicit HBM"
                    " intermediate. Restructure so it folds into the consumer"
                    " (e.g. einsum / fused form)."
                )
                inefficient_ops.append(candidate)

        message = (
            f"Detected {len(inefficient_ops)} standalone formatting operations"
            " causing HBM materialization overhead."
            if inefficient_ops
            else "No unfused reshape bottlenecks detected."
        )
        return json.dumps(
            {
                "bottlenecks_found": bool(inefficient_ops),
                "inefficient_ops": inefficient_ops,
                "message": message,
            },
            indent=2,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("detect_unfused_reshapes failed", exc_info=True)
        return json.dumps(
            {"error": f"{type(e).__name__}: {e}",
             "tool": "detect_unfused_reshapes"},
            indent=2,
        )
