"""Rebuild index script: clear all data and rebuild from scratch."""

from __future__ import annotations

from loguru import logger

from src.config_loader import get_config
from src.vector_store.indexer import Indexer


def main() -> None:
    """Clear existing data and rebuild the index from scratch."""
    config = get_config()

    logger.info("=== Rebuilding TDX Knowledge Base Index ===")

    indexer = Indexer(config)
    stats = indexer.rebuild()

    print("\n=== Rebuild Complete ===")
    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Stored: {stats['stored_count']}")


if __name__ == "__main__":
    main()
