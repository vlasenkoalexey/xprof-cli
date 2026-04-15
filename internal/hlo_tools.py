"""HLO-related tools for OSS XProf MCP.

These tools use the xprof HTTP server's `graph_viewer` and `module_list`
endpoints to inspect HLO modules — no local protobuf parsing required.

The `graph_viewer` endpoint (with `type=long_txt`) returns a full human-
readable HLO text representation identical to what you'd get from
`xla_client.HloModule.to_string()`.
"""

import collections
import logging
import re
from typing import Optional

from xprof_mcp.internal import xprof_client


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NO_HLO_DATA_MSG = (
    "No HLO graph data available for this run.\n"
    "The profile was captured without HLO export. To get HLO data:\n"
    "  1. XLA dump files (no re-profile needed):\n"
    "       XLA_FLAGS='--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text'\n"
    "       Then use list_hlo_dump_modules / get_hlo_dump tools.\n"
    "  2. Profile with HLO export enabled (PyTorch/XLA):\n"
    "       xp.trace(..., options={'xplane_options': {'save_hlo': True}})"
)


def _fetch_hlo_text(
    run: str,
    module_name: Optional[str],
    *,
    print_metadata: bool = False,
) -> str:
    """Fetches the full HLO text for a module via the graph_viewer endpoint."""
    import requests as _requests  # pylint: disable=g-import-not-at-top
    client = xprof_client.get_client()
    kwargs: dict = {"type": "long_txt"}
    if module_name:
        kwargs["host"] = module_name
    if print_metadata:
        kwargs["show_metadata"] = "true"
    try:
        data = client.fetch("graph_viewer", run, **kwargs)
    except _requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 500:
            body = e.response.text.strip()
            if "hlo proto" in body.lower() or "can not load" in body.lower():
                raise RuntimeError(_NO_HLO_DATA_MSG) from None
        raise
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def list_hlo_modules(run: str) -> str:
    """Lists all HLO modules available in the profiling run.

    **Use this first** to discover which compiled programs are in the profile.

    Args:
        run: The run name (session directory name). Use `list_runs` to discover
             available runs.

    Returns:
        A human-readable list of module names.
    """
    client = xprof_client.get_client()
    try:
        modules_str = client.get_hlo_module_list(run)
        if not modules_str:
            return f"No HLO modules found for run '{run}'."
        modules = [m.strip() for m in modules_str.split(",") if m.strip()]
        lines = [f"Found {len(modules)} HLO module(s) in run '{run}':"]
        for i, name in enumerate(modules):
            lines.append(f"  {i}. {name}")
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error listing HLO modules for run %s", run)
        return f"Error listing HLO modules: {e}"


def get_hlo_module_content(
    run: str,
    module_name: Optional[str] = None,
    fmt: str = "text",
    max_lines: int = 2000,
    *,
    print_metadata: bool = False,
) -> str:
    """Returns the full HLO module content (instruction graph) as text.

    **Use this** after `list_hlo_modules` to inspect the compiled HLO for a
    specific module. This is the primary tool for detailed code review.

    Args:
        run:            The run name.
        module_name:    Name of the module (from `list_hlo_modules`). If
                        omitted, returns the first (or only) module.
        fmt:            Output format. Only 'text' is supported.
        max_lines:      Maximum number of lines to return (default 2000).
                        Set to -1 for unlimited.
        print_metadata: Include op metadata (source location) in output.

    Returns:
        The full HLO text representation for the selected module.
    """
    if fmt != "text":
        return f"Unsupported format '{fmt}'. Only 'text' is supported."
    try:
        text = _fetch_hlo_text(run, module_name, print_metadata=print_metadata)
        if not text.strip():
            # Try listing modules to give a helpful error
            client = xprof_client.get_client()
            modules_str = client.get_hlo_module_list(run)
            msg = f"No HLO content returned for run='{run}'"
            if module_name:
                msg += f", module='{module_name}'"
            if modules_str:
                msg += f". Available modules: {modules_str}"
            return msg

        if max_lines > 0:
            lines = text.splitlines()
            if len(lines) > max_lines:
                truncated = "\n".join(lines[:max_lines])
                return (
                    truncated
                    + f"\n... (truncated after {max_lines} lines,"
                    f" total {len(lines)}. Use max_lines=-1 to see all)"
                )
        return text

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error fetching HLO module content for run %s", run)
        return f"Error fetching HLO module content: {e}"


def get_hlo_neighborhood(
    run: str,
    instruction_name: str,
    radius: int = 2,
    module_name: Optional[str] = None,
    *,
    print_metadata: bool = False,
) -> str:
    """Returns the neighborhood of a specific HLO instruction.

    **Use this to root-cause slow ops.** Inspects the immediate producers and
    consumers of an instruction to find blocking fusions, bitcasts, or copies.

    The neighborhood is extracted by:
    1. Fetching the full HLO text from the server.
    2. Building a lightweight adjacency map from the text using regex.
    3. Performing a BFS traversal up to `radius` hops away.

    Args:
        run:              The run name.
        instruction_name: Name of the target instruction (e.g. 'fusion.3').
                          Leading '%' is stripped automatically.
        radius:           Number of hops to traverse in each direction (default 2).
        module_name:      Module to search in. Defaults to the first module.
        print_metadata:   Include op metadata in output.

    Returns:
        A text listing of the instruction neighborhood.
    """
    # Strip leading '%' if present
    instruction_name = instruction_name.lstrip("%")

    try:
        full_text = _fetch_hlo_text(run, module_name, print_metadata=print_metadata)
        if not full_text.strip():
            return f"No HLO text returned for run='{run}', module='{module_name}'."

        # ------------------------------------------------------------------
        # Parse the HLO text into a lightweight graph
        # name -> (definition_line, set_of_operand_names)
        # ------------------------------------------------------------------
        # Match lines like:  %foo = f32[...] op(%bar, %baz)
        define_re = re.compile(r"^\s*%(?P<name>[\w.\-]+)\s*=")
        operand_re = re.compile(r"%(?P<op>[\w.\-]+)")

        name_to_line: dict[str, str] = {}
        name_to_operands: dict[str, list[str]] = {}

        for line in full_text.splitlines():
            m = define_re.match(line)
            if m:
                name = m.group("name")
                name_to_line[name] = line.strip()
                ops = [
                    om.group("op")
                    for om in operand_re.finditer(line[m.end():])
                ]
                name_to_operands[name] = ops

        if instruction_name not in name_to_line:
            # Suggest nearby names
            candidates = [
                n for n in name_to_line
                if instruction_name.split(".")[0] in n
            ][:10]
            msg = f"Instruction '%{instruction_name}' not found in HLO text."
            if candidates:
                msg += f" Similar names: {', '.join('%' + c for c in candidates)}"
            return msg

        # Build reverse map: name -> list of users
        name_to_users: dict[str, list[str]] = collections.defaultdict(list)
        for name, operands in name_to_operands.items():
            for op in operands:
                name_to_users[op].append(name)

        # ------------------------------------------------------------------
        # BFS
        # ------------------------------------------------------------------
        visited: set[str] = {instruction_name}
        queue: collections.deque = collections.deque([(instruction_name, 0)])
        neighborhood: list[tuple[int, str]] = []

        while queue:
            curr, dist = queue.popleft()
            neighborhood.append((dist, curr))
            if dist < radius:
                for neighbor in name_to_operands.get(curr, []) + name_to_users.get(curr, []):
                    if neighbor not in visited and neighbor in name_to_line:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        neighborhood.sort(key=lambda x: (x[0], x[1]))

        # ------------------------------------------------------------------
        # Format output
        # ------------------------------------------------------------------
        lines_out = [
            f"Neighborhood of '%{instruction_name}' (radius={radius}, run='{run}'):",
            "",
        ]
        for dist, name in neighborhood:
            indent = "  " * (dist + 1)
            marker = " [TARGET]" if name == instruction_name else f" [dist={dist}]"
            definition = name_to_line.get(name, f"%{name} = (definition not found)")
            lines_out.append(f"{indent}{marker} {definition}")

        return "\n".join(lines_out)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Error computing HLO neighborhood for run %s", run)
        return f"Error computing HLO neighborhood: {e}"
