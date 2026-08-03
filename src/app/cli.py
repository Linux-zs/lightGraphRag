"""Command-line tools for the LightRAG knowledge-base workbench."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import shutil
import sys
from pathlib import Path

from loguru import logger

from src.config_loader import get_config
from src.doc_processor.loader import DocumentLoader
from src.lightrag_service import DEFAULT_WORKSPACE, LightRAGService, sanitize_workspace
from src.runtime_lock import RuntimeLock

DOCUMENT_MAX_BYTES = int(
    os.environ.get("LIGHTGRAPHRAG_DOCUMENT_UPLOAD_MAX_BYTES", str(50 * 1024 * 1024))
)
PARSED_TEXT_MAX_CHARS = int(
    os.environ.get("LIGHTGRAPHRAG_PARSED_TEXT_MAX_CHARS", "5000000")
)


def _preflight_documents(docs_dir: str | Path) -> list:
    source_dir = Path(docs_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {source_dir}")
    loader = DocumentLoader()
    files = loader.scan_directory(source_dir)
    if not files:
        raise ValueError(f"No supported documents found in: {source_dir}")
    by_name: dict[str, list[Path]] = {}
    for path in files:
        if path.stat().st_size > DOCUMENT_MAX_BYTES:
            raise ValueError(
                f"Document exceeds the {DOCUMENT_MAX_BYTES} byte limit: {path}"
            )
        by_name.setdefault(path.name.casefold(), []).append(path)
    conflicts = [paths for paths in by_name.values() if len(paths) > 1]
    if conflicts:
        details = "; ".join(
            ", ".join(str(path) for path in paths)
            for paths in conflicts
        )
        raise ValueError(f"Duplicate document basenames are not supported: {details}")
    documents = [loader.load_document(path) for path in files]
    empty = [doc.file_path for doc in documents if not doc.raw_text.strip()]
    if empty:
        raise ValueError(f"Parsed document text is empty: {', '.join(empty)}")
    oversized_text = [
        doc.file_path
        for doc in documents
        if len(doc.raw_text) > PARSED_TEXT_MAX_CHARS
    ]
    if oversized_text:
        raise ValueError(
            "Parsed document text exceeds the configured character limit: "
            + ", ".join(oversized_text)
        )
    return documents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preflight_rebuild_sources(
    docs_dir: str | Path,
    workspace: str,
) -> list:
    documents = _preflight_documents(docs_dir)
    config = get_config()
    data_dir = Path(config.get("paths", {}).get("data_dir", "./data"))
    upload_dir = data_dir / "uploads" / workspace
    service = LightRAGService(config, workspace=workspace)
    manifest = service._load_manifest()
    registered = {
        str(item.get("doc_name") or ""): item
        for item in manifest.get("documents", {}).values()
        if isinstance(item, dict) and item.get("doc_name")
    }
    source_by_name = {doc.file_name: Path(doc.file_path) for doc in documents}
    if set(source_by_name) != set(registered):
        missing = sorted(set(registered) - set(source_by_name))
        extra = sorted(set(source_by_name) - set(registered))
        details = []
        if missing:
            details.append(f"source missing: {', '.join(missing)}")
        if extra:
            details.append(f"not ingested: {', '.join(extra)}")
        raise ValueError(
            "Rebuild source set differs from the current workspace; ingest changes first ("
            + "; ".join(details)
            + ")"
        )
    changed = []
    for name, source_path in source_by_name.items():
        managed_path = upload_dir / name
        if not managed_path.is_file() or _sha256(source_path) != _sha256(managed_path):
            changed.append(name)
    if changed:
        raise ValueError(
            "Rebuild sources differ from the managed uploads; ingest changes first: "
            + ", ".join(changed)
        )
    return documents


async def _ingest_docs(
    docs_dir: str | Path,
    workspace: str,
    service: LightRAGService | None = None,
) -> dict:
    config = get_config()
    loader = DocumentLoader()
    workspace = sanitize_workspace(workspace)
    service = service or LightRAGService(config, workspace=workspace)
    docs = _preflight_documents(docs_dir)
    data_dir = Path(config.get("paths", {}).get("data_dir", "./data"))
    upload_dir = data_dir / "uploads" / workspace
    raw_text_dir = data_dir / "upload_text" / workspace
    upload_dir.mkdir(parents=True, exist_ok=True)
    raw_text_dir.mkdir(parents=True, exist_ok=True)

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
    if stats["failed"]:
        raise RuntimeError(f"{stats['failed']} documents failed to index")


def cmd_rebuild(args: argparse.Namespace) -> None:
    """Clear and rebuild the index."""
    config = get_config()
    workspace = sanitize_workspace(args.workspace)
    data_dir = Path(config.get("paths", {}).get("data_dir", "./data"))
    docs_dir = args.docs_dir or (data_dir / "uploads" / workspace)
    _preflight_rebuild_sources(docs_dir, workspace)

    async def rebuild() -> dict:
        from src.api.server import (
            RebuildIndexRequest,
            _index_tasks,
            _start_workspace_rebuild,
        )

        task, _result = await _start_workspace_rebuild(
            RebuildIndexRequest(workspace=workspace),
            reason="CLI rebuild",
        )
        while task.get("status") not in {"succeeded", "failed", "partial", "cancelled"}:
            await asyncio.sleep(0.2)
            task = _index_tasks[task["task_id"]]
        return task

    logger.info("Rebuilding LightRAG workspace {}...", workspace)
    stats = asyncio.run(rebuild())
    print(stats.get("message", "重建完成"))
    if stats.get("status") != "succeeded":
        raise RuntimeError(stats.get("message") or "Rebuild failed")


def cmd_search(args: argparse.Namespace) -> None:
    """Search the knowledge base."""
    config = get_config()
    workspace = sanitize_workspace(args.workspace)
    service = LightRAGService(config, workspace=workspace)

    question = args.query
    top_k = args.top_k or 40

    logger.info("Searching workspace {}: '{}'", workspace, question)
    async def search():
        try:
            return await service.query(question, top_k=top_k)
        finally:
            await service.finalize()

    answer = asyncio.run(search())

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

    try:
        if args.command == "server":
            cmd_server(args)
            return
        if args.command:
            data_dir = Path(get_config().get("paths", {}).get("data_dir", "./data"))
            with RuntimeLock(data_dir / ".runtime.lock"):
                if args.command == "ingest":
                    cmd_ingest(args)
                elif args.command == "rebuild":
                    cmd_rebuild(args)
                elif args.command == "search":
                    cmd_search(args)
            return
        parser.print_help()
    except Exception as exc:
        logger.error("Command failed: {}", exc)
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
