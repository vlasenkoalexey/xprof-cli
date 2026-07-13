"""xprof-cli entry point.

CLI frontend over the same tool functions the MCP server registers.
Populated in the cli-frontend branch; this stub keeps the console script
importable from the first packaging commit.
"""

import sys


def main() -> None:
    print(
        "xprof-cli: CLI frontend under construction on this branch. "
        "Use the MCP server (`xprof-mcp`) meanwhile.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
