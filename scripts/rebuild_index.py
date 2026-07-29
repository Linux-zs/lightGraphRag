"""Compatibility wrapper for a workspace-scoped LightRAG rebuild."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from src.app.cli import main as cli_main

    sys.argv.insert(1, "rebuild")
    cli_main()


if __name__ == "__main__":
    main()
