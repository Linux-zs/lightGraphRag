"""Compatibility wrapper for the LightRAG CLI ingest command."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so "src" imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    from src.app.cli import main as cli_main

    sys.argv.insert(1, "ingest")
    cli_main()


if __name__ == "__main__":
    main()
