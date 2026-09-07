import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src import lightrag_service as module
from src.doc_processor.parsers.base_parser import Document
from src.lightrag_service import LightRAGDocStatus, LightRAGService


@pytest.fixture
def versioned_service(tmp_path, monkeypatch):
    service = LightRAGService({"paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "rag")}}, workspace="review")
    doc = Document(doc_id="source", file_name="probe.txt", file_path="probe.txt", file_type="txt", raw_text="new content", metadata={"lightrag_doc_id": "doc_old"})
    service._save_manifest({"documents": {"doc_old": {
        "doc_id": "doc_old", "doc_name": doc.file_name, "indexed": True,
        "active_index_doc_id": "doc_old", "active_index_status": "processed", "status": "processed",
    }}})
    indexed = {"doc_old"}
    rag = SimpleNamespace(addon_params={}, doc_status=object(), adelete_by_doc_id=AsyncMock(return_value=SimpleNamespace(status="success")))
    service._rag = rag
    monkeypatch.setattr(service, "assert_embedding_compatible", lambda: None)
    monkeypatch.setattr(service, "cleanup_interrupted_index_docs", AsyncMock(return_value=[]))
    monkeypatch.setattr(service, "_runtime_models", lambda: {"embedding": {}, "kg": {"model": "fake"}})
    monkeypatch.setattr(service, "record_embedding_signature", lambda *a, **kw: None)
    monkeypatch.setattr(service, "graph_extraction_guidance", lambda **kw: "")
    monkeypatch.setattr(service, "graph_governance_summary", lambda: {})
    monkeypatch.setattr(service, "_temporary_index_llm_and_kg_filter", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(service, "get_doc_status", AsyncMock(return_value=LightRAGDocStatus(doc_id="new", status="processed")))

    async def insert(_rag, _doc, index_id, **kwargs):
        indexed.add(index_id)
        return "track"

    async def discard(index_id):
        indexed.discard(index_id)

    monkeypatch.setattr(service, "_insert_document_text", insert)
    monkeypatch.setattr(service, "discard_lightrag_doc", discard)
    monkeypatch.setattr(module, "install_stage_timing", lambda rag: SimpleNamespace(on_update=None, scope=lambda: nullcontext(), to_stages=lambda: {}))
    return service, rag, doc, indexed


def test_cancel_after_old_version_deleted_keeps_published_version(versioned_service):
    service, rag, doc, indexed = versioned_service

    async def retire(index_id):
        indexed.discard(index_id)
        raise asyncio.CancelledError()

    rag.adelete_by_doc_id = retire
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service.index_document(doc))
    saved = service._load_manifest()["documents"]["doc_old"]
    assert saved["active_index_doc_id"] in indexed
    assert saved["active_index_doc_id"] != "doc_old"
    assert saved["last_index_attempt_status"] == "succeeded"
    assert saved["retired_index_doc_ids"] == ["doc_old"]


def test_missing_sdk_status_never_publishes_a_skipped_insertion(versioned_service):
    service, rag, doc, indexed = versioned_service
    service.get_doc_status.return_value = None
    with pytest.raises(RuntimeError, match="previous index was preserved"):
        asyncio.run(service.index_document(doc))
    assert indexed == {"doc_old"}
    assert service._load_manifest()["documents"]["doc_old"]["active_index_doc_id"] == "doc_old"
    rag.adelete_by_doc_id.assert_not_awaited()


@pytest.mark.parametrize("method", ["query", "stream_query", "preview_context", "text_recall"])
def test_retirement_failure_blocks_stale_queries_and_recovers(versioned_service, method):
    service, rag, doc, _ = versioned_service
    rag.adelete_by_doc_id = AsyncMock(return_value=SimpleNamespace(status="failed", message="disk unavailable"))
    asyncio.run(service.index_document(doc))
    rag.aquery_llm = AsyncMock(return_value={"llm_response": {"content": "current"}})
    rag.chunks_vdb = SimpleNamespace(query=AsyncMock(return_value=[]))
    service._runtime_models = lambda: {"rerank": {"enabled": False}}

    with pytest.raises(RuntimeError, match="INDEX_CLEANUP_PENDING"):
        asyncio.run(getattr(service, method)("question"))
    rag.aquery_llm.assert_not_awaited()
    rag.chunks_vdb.query.assert_not_awaited()

    rag.adelete_by_doc_id.return_value = SimpleNamespace(status="not_found")
    # Preview exercises the shared recovery gate without a streaming mock.
    asyncio.run(service.preview_context("question"))
    assert not service._load_manifest()["documents"]["doc_old"]["retired_index_doc_ids"]
    rag.aquery_llm.assert_awaited_once()


def test_backfill_is_owned_by_active_version(versioned_service, monkeypatch):
    service, rag, _, _ = versioned_service
    manifest = service._load_manifest()
    manifest["documents"]["doc_old"].update(active_index_doc_id="doc_old-v2", chunks_list=["chunk-v2"])
    service._save_manifest(manifest)
    rag.text_chunks = SimpleNamespace(get_by_ids=AsyncMock(return_value=[{"content": "Atlas reads Cedar", "full_doc_id": "doc_old-v2"}]))
    rag._process_extract_entities = AsyncMock(return_value=[({"Atlas": [], "Cedar": []}, {("Atlas", "Cedar"): []})])
    rag._build_global_config = lambda: {}
    rag._insert_done_with_cleanup = AsyncMock()
    for name in ["chunk_entity_relation_graph", "entities_vdb", "relationships_vdb", "full_entities", "full_relations", "llm_response_cache", "entity_chunks", "relation_chunks"]:
        setattr(rag, name, object())
    merger = AsyncMock()
    monkeypatch.setattr(module, "merge_nodes_and_edges", merger)
    asyncio.run(service.backfill_document_graph("probe.txt"))
    assert merger.call_args.kwargs["doc_id"] == "doc_old-v2"
