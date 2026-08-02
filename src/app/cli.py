"""Command-line tools for the LightRAG knowledge-base workbench."""

from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from loguru import logger

from src.config_loader import get_config
from src.doc_processor.loader import DocumentLoader
from src.lightrag_service import DEFAULT_WORKSPACE, LightRAGService, sanitize_workspace


async def _ingest_docs(
    docs_dir: str | Path,
    workspace: str,
    service: LightRAGService | None = None,
) -> dict:
    config = get_config()
    loader = DocumentLoader()
    workspace = sanitize_workspace(workspace)
    service = service or LightRAGService(config, workspace=workspace)
    source_dir = Path(docs_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {source_dir}")
    data_dir = Path(config.get("paths", {}).get("data_dir", "./data"))
    upload_dir = data_dir / "uploads" / workspace
    raw_text_dir = data_dir / "upload_text" / workspace
    upload_dir.mkdir(parents=True, exist_ok=True)
    raw_text_dir.mkdir(parents=True, exist_ok=True)

    docs = loader.load_all(docs_dir)
    indexed = 0
    failed = 0
    for doc in docs:
        try:
            source_path = Path(doc.file_path)
            destination = upload_dir / source_path.name
            if source_path.resolve() != destination.resolve():
                shutil.copy2(source_path, destination)
            managed_doc = loader.load_document(destination)
            registered = service.register_upload(managed_doc)
            raw_text_path = raw_text_dir / f"{registered['doc_id']}.txt"
            raw_text_path.write_text(managed_doc.raw_text, encoding="utf-8")
            managed_doc.metadata["lightrag_doc_id"] = registered["doc_id"]
            managed_doc.metadata["raw_text_path"] = str(raw_text_path)
            service.register_upload(managed_doc)
            await service.index_document(managed_doc)
            indexed += 1
            logger.info("Indexed {} into {}", managed_doc.file_name, workspace)
        except Exception:
            failed += 1
            logger.exception("Failed to index {}", doc.file_name)
    return {
        "workspace": workspace,
        "documents": len(docs),
        "indexed": indexed,
        "failed": failed,
    }


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents and build index."""
    config = get_config()
    docs_dir = args.docs_dir or config.get("paths", {}).get("docs_dir")

    logger.info("Starting LightRAG document ingestion...")
    stats = asyncio.run(_ingest_docs(docs_dir, args.workspace))
    print(f"导入完成: {stats['documents']} 文档, {stats['indexed']} 已索引, {stats['failed']} 失败")


def cmd_rebuild(args: argparse.Namespace) -> None:
    """Clear and rebuild the index."""
    config = get_config()
    docs_dir = args.docs_dir or config.get("paths", {}).get("docs_dir")
    workspace = sanitize_workspace(args.workspace)

    async def rebuild() -> dict:
        service = LightRAGService(config, workspace=workspace)
        await service.clear_workspace()
        stats = await _ingest_docs(docs_dir, workspace, service)
        replay = await service.replay_graph_audit()
        return {**stats, "graph_replay": replay}

    logger.info("Rebuilding LightRAG workspace {}...", workspace)
    stats = asyncio.run(rebuild())
    print(f"重建完成: {stats['documents']} 文档, {stats['indexed']} 已索引, {stats['failed']} 失败")


def cmd_search(args: argparse.Namespace) -> None:
    """Search the knowledge base."""
    config = get_config()
    workspace = sanitize_workspace(args.workspace)
    service = LightRAGService(config, workspace=workspace)

    question = args.query
    top_k = args.top_k or 40

    logger.info("Searching workspace {}: '{}'", workspace, question)
    answer = asyncio.run(service.query(question, top_k=top_k))

    print("\n" + "=" * 60)
    print(f"问题: {question}")
    print("=" * 60)
    print(answer.content)
    print("=" * 60)

    if answer.citations:
        print("\n来源引用:")
        for citation in answer.citations:
            print(f"  [{citation['index']}] {citation['doc_name']} - 分块 #{citation.get('chunk_index', 0)}")
            print(f"      {citation.get('excerpt', '')[:80]}...")

def cmd_server(args: argparse.Namespace) -> None:
    """Start the FastAPI workbench server."""
    import uvicorn
    logger.info("Starting Knowledge Base Workbench API on port {}...", args.port)
    logger.info("Frontend: http://localhost:5173")
    logger.info("API docs: http://localhost:{}/docs", args.port)
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LightRAG Knowledge Base CLI",
        prog="python -m src.app.cli",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents and build index")
    ingest_parser.add_argument("--docs-dir", help="Override docs directory path")
    ingest_parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Target knowledge base")

    # rebuild
    rebuild_parser = subparsers.add_parser("rebuild", help="Clear and rebuild index")
    rebuild_parser.add_argument("--docs-dir", help="Override docs directory path")
    rebuild_parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Target knowledge base")

    # search
    search_parser = subparsers.add_parser("search", help="Search the knowledge base")
    search_parser.add_argument("query", help="Search query string")
    search_parser.add_argument("--top-k", type=int, help="Number of results")
    search_parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Target knowledge base")

    # server
    server_parser = subparsers.add_parser("server", help="Start FastAPI workbench server")
    server_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    server_parser.add_argument("--port", type=int, default=8101, help="Server port")
    server_parser.add_argument("--no-reload", dest="reload", action="store_false",
                                default=True, help="Disable auto-reload")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "rebuild":
        cmd_rebuild(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "server":
        cmd_server(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
