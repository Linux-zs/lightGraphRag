"""One-click knowledge base initialization script."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so "src" imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.config_loader import get_config
from src.graph_manager.builder import GraphBuilder
from src.vector_store.indexer import Indexer


def main() -> None:
    """Initialize the knowledge base: ingest documents + build graph."""
    config = get_config()

    logger.info("=== TDX Knowledge Base Initialization ===")

    # Step 1: Ingest documents and build vector index
    logger.info("Step 1: Ingesting documents...")
    indexer = Indexer(config)
    stats = indexer.ingest()
    logger.info(f"Ingestion complete: {stats}")

    # Step 2: Build knowledge graph (skeleton only, no LLM enrichment in init)
    logger.info("Step 2: Building knowledge graph skeleton...")
    builder = GraphBuilder(config)
    graph, nodes = builder.build_skeleton()
    logger.info(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Step 3: Verify
    logger.info("Step 3: Verifying...")
    count = indexer.store.count()
    logger.info(f"ChromaDB collection has {count} chunks")

    print("\n=== Initialization Complete ===")
    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Stored: {stats['stored_count']}")
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print("\nReady to use! Run:")
    print("  python -m src.app.cli ui          # Start Streamlit UI")
    print("  python -m src.app.cli search 'query'  # Search via CLI")


if __name__ == "__main__":
    main()
