import asyncio
import json
from contextlib import nullcontext
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from src.api import server
from src import lightrag_service
from src.lightrag_service import LightRAGDocStatus, LightRAGService


def _service(tmp_path, workspace="workspace_a"):
    return LightRAGService(
        config={
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "lightrag_dir": str(tmp_path / "lightrag"),
            },
            "lightrag": {"workspace": "default"},
        },
        workspace=workspace,
    )


def _manifest_item(indexed=False):
    return {
        "documents": {
            "doc_123": {
                "doc_id": "doc_123",
                "doc_name": "example.txt",
                "file_type": "txt",
                "file_path": "example.txt",
                "indexed": indexed,
                "status": "uploaded",
            }
        }
    }


def test_prepare_embedding_text_removes_non_bmp_chars(tmp_path):
    service = _service(tmp_path)

    safe = service._prepare_embedding_text("# Title 📖\x00 body | TCP/IP (config)", 700)

    assert "📖" not in safe
    assert "\x00" not in safe
    assert "|" not in safe
    assert "/" not in safe
    assert "(" not in safe
    assert safe == "# Title body TCP IP config"
    assert service._prepare_embedding_text("📖\x00", 700) == "empty document chunk"


def test_embedding_fallback_splits_batch_and_shortens_bad_text(tmp_path, monkeypatch):
    service = _service(tmp_path)
    calls: list[list[str]] = []

    async def fake_embed(texts, **_kwargs):
        calls.append(list(texts))
        if len(texts) > 1:
            raise RuntimeError("Error code: 400 - {'code': 20015, 'message': 'The parameter is invalid.'}")
        if len(texts[0]) > 120:
            raise RuntimeError("Error code: 400 - {'code': 20015, 'message': 'The parameter is invalid.'}")
        return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr(lightrag_service.openai_embed, "func", fake_embed)

    result = asyncio.run(
        service._embed_texts_with_fallback(
            ["normal chunk", "x" * 480],
            embed_model="embed",
            base_url="https://example.test/v1",
            api_key="key",
            max_tokens=480,
        )
    )

    assert result.shape == (2, 1024)
    assert ["normal chunk", "x" * 480] in calls
    assert ["normal chunk"] in calls
    assert any(len(batch) == 1 and len(batch[0]) <= 120 for batch in calls)


def test_delete_uses_actual_lightrag_status_even_if_manifest_is_not_indexed(tmp_path):
    service = _service(tmp_path)
    service._save_manifest(_manifest_item(indexed=False))
    fake_rag = SimpleNamespace(
        adelete_by_doc_id=AsyncMock(return_value=SimpleNamespace(status="success", message="ok"))
    )
    service._rag = fake_rag
    service.get_doc_status = AsyncMock(
        return_value=LightRAGDocStatus(doc_id="doc_123", status="processed")
    )

    result = asyncio.run(service.delete_document("example.txt"))

    fake_rag.adelete_by_doc_id.assert_awaited_once_with("doc_123")
    assert result["doc_id"] == "doc_123"
    assert service._load_manifest()["documents"] == {}


def test_delete_failure_keeps_manifest_and_source_identity(tmp_path):
    service = _service(tmp_path)
    original = _manifest_item(indexed=True)
    service._save_manifest(original)
    service._rag = SimpleNamespace(
        adelete_by_doc_id=AsyncMock(return_value=SimpleNamespace(status="failed", message="busy"))
    )
    service.get_doc_status = AsyncMock(
        return_value=LightRAGDocStatus(doc_id="doc_123", status="processed")
    )

    with pytest.raises(RuntimeError, match="busy"):
        asyncio.run(service.delete_document("example.txt"))

    assert service._load_manifest()["documents"] == service._validate_manifest(
        original, source=service.manifest_path
    )["documents"]


def test_clear_workspace_can_preserve_manifest_for_rebuild(tmp_path):
    service = _service(tmp_path)
    manifest = _manifest_item(indexed=True)
    service._save_manifest(manifest)
    service.workspace_dir.mkdir(parents=True)
    (service.workspace_dir / "index.json").write_text("{}", encoding="utf-8")

    result = asyncio.run(service.clear_workspace(preserve_manifest=True))

    assert result["preserved_manifest"] is True
    assert service._load_manifest()["documents"] == service._validate_manifest(
        manifest, source=service.manifest_path
    )["documents"]
    assert not (service.workspace_dir / "index.json").exists()


def test_former_default_workspace_manifest_is_migrated_to_its_own_workspace(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "lightrag_manifest.json"
    source.write_text(
        '{"documents":{"doc_1":{"doc_name":"legacy.txt"}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        server,
        "get_config",
        lambda: {
            "paths": {"data_dir": str(data_dir)},
            "lightrag": {"workspace": "default"},
        },
    )

    result = server._migrate_legacy_default_workspace()

    target = data_dir / "lightrag_manifests" / "tdx_default.json"
    assert result["migrated"] is True
    assert not source.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["documents"]["doc_1"][
        "doc_name"
    ] == "legacy.txt"
    assert server._migrate_legacy_default_workspace()["reason"] == "already_migrated"


def test_graph_audit_replay_is_oldest_first_and_does_not_append_audit(tmp_path):
    service = _service(tmp_path)
    config = service.load_graph_governance()
    config["audit_log"] = [
        {
            "id": "newer",
            "action": "edit_entity",
            "payload": {"entity_name": "A", "updated_data": {"description": "new"}},
        },
        {
            "id": "older",
            "action": "create_entity",
            "payload": {"entity_name": "A", "entity_data": {"description": "old"}},
        },
    ]
    service.save_graph_governance(config)
    calls = []

    async def create_entity(**kwargs):
        calls.append(("create", kwargs["entity_name"]))

    async def edit_entity(**kwargs):
        calls.append(("edit", kwargs["entity_name"]))

    service._rag = SimpleNamespace(
        acreate_entity=create_entity,
        aedit_entity=edit_entity,
    )

    result = asyncio.run(service.replay_graph_audit())

    assert calls == [("create", "A"), ("edit", "A")]
    assert result["applied"] == 2
    assert len(service.load_graph_governance()["audit_log"]) == 2


def test_graph_backfill_reuses_existing_chunks_without_chunk_vector_upsert(tmp_path, monkeypatch):
    service = _service(tmp_path)
    manifest = _manifest_item(indexed=True)
    manifest["documents"]["doc_123"]["chunks_list"] = ["chunk-a"]
    manifest["documents"]["doc_123"]["chunk_count"] = 1
    service._save_manifest(manifest)

    async def get_by_ids(ids):
        assert ids == ["chunk-a"]
        return [{"content": "A depends on B", "chunk_order_index": 0}]

    async def extract(chunks, _status, _lock):
        assert list(chunks) == ["chunk-a"]
        return [
            (
                {"A": [{"entity_name": "A"}], "B": [{"entity_name": "B"}]},
                {("A", "B"): [{"description": "depends"}]},
            )
        ]

    fake_rag = SimpleNamespace(
        addon_params={},
        text_chunks=SimpleNamespace(get_by_ids=get_by_ids),
        _process_extract_entities=extract,
        chunk_entity_relation_graph=object(),
        entities_vdb=object(),
        relationships_vdb=object(),
        full_entities=object(),
        full_relations=object(),
        llm_response_cache=object(),
        entity_chunks=object(),
        relation_chunks=object(),
        _build_global_config=lambda: {},
        _insert_done_with_cleanup=AsyncMock(),
        _discard_pending_index_ops=AsyncMock(),
    )
    service._rag = fake_rag
    monkeypatch.setattr(service, "assert_embedding_compatible", lambda: None)
    monkeypatch.setattr(service, "_runtime_models", lambda: {"kg": {"model": "kg-model"}})
    monkeypatch.setattr(
        service,
        "_temporary_index_llm_and_kg_filter",
        lambda *_args, **_kwargs: nullcontext(),
    )
    merge = AsyncMock()
    monkeypatch.setattr(lightrag_service, "merge_nodes_and_edges", merge)

    result = asyncio.run(
        service.backfill_document_graph(
            "example.txt",
            kg_max_entities=8,
            kg_max_records=16,
        )
    )

    merge.assert_awaited_once()
    fake_rag._insert_done_with_cleanup.assert_awaited_once()
    assert result["kg_status"] == "complete"
    assert result["kg_entity_count"] == 2
    assert result["kg_relation_count"] == 1
    saved = service._load_manifest()["documents"]["doc_123"]
    assert saved["indexed"] is True
    assert saved["kg_extraction_limits"]["max_entities_per_chunk"] == 8


def test_temporary_kg_limits_are_restored_after_index_operation(tmp_path, monkeypatch):
    service = _service(tmp_path)

    class FakeRag:
        llm_model_func = None
        llm_model_name = "chat"
        llm_model_kwargs = {}
        entity_extract_max_entities = 24
        entity_extract_max_records = 48
        _role_llm_states = {}

        async def _process_extract_entities(self, *_args, **_kwargs):
            return []

    rag = FakeRag()
    monkeypatch.setattr(service, "_runtime_models", lambda: {"kg": {"model": "kg"}})
    monkeypatch.setattr(service, "_make_kg_llm_func", lambda: AsyncMock())
    monkeypatch.setattr(service, "_llm_kwargs", lambda _role: {})

    with service._temporary_index_llm_and_kg_filter(
        rag,
        skip_kg=False,
        max_entities=8,
        max_records=16,
    ):
        assert rag.entity_extract_max_entities == 8
        assert rag.entity_extract_max_records == 16

    assert rag.entity_extract_max_entities == 24
    assert rag.entity_extract_max_records == 48


def test_custom_graph_import_persists_package_and_removes_source_from_vector_recall(
    tmp_path,
    monkeypatch,
):
    service = _service(tmp_path)
    fake_rag = SimpleNamespace(
        ainsert_custom_kg=AsyncMock(),
        chunks_vdb=SimpleNamespace(delete=AsyncMock()),
        _insert_done=AsyncMock(),
    )
    service._rag = fake_rag
    monkeypatch.setattr(service, "append_graph_audit", lambda *_args, **_kwargs: {})

    result = asyncio.run(
        service.import_custom_graph(
            file_name="relations.txt",
            source_text="A depends on B",
            entities=[
                {"entity_name": "A", "entity_type": "system", "description": "A"},
                {"entity_name": "B", "entity_type": "system", "description": "B"},
            ],
            relationships=[
                {
                    "src_id": "A",
                    "tgt_id": "B",
                    "relation_type": "depends",
                    "description": "A depends on B",
                }
            ],
        )
    )

    fake_rag.ainsert_custom_kg.assert_awaited_once()
    fake_rag.chunks_vdb.delete.assert_awaited_once_with([result["source_chunk_id"]])
    assert service._graph_import_path(result["import_id"]).exists()
    assert service.list_graph_imports()[0]["relationship_count"] == 1


def test_workspace_queries_are_blocked_during_rebuild(monkeypatch):
    monkeypatch.setattr(
        server,
        "_index_tasks",
        {
            "task_active": {
                "task_id": "task_active",
                "workspace": "workspace_a",
                "kind": "rebuild",
                "status": "running",
                "message": "rebuilding",
            }
        },
    )

    with pytest.raises(HTTPException) as exc:
        server._ensure_workspace_available("workspace_a")

    assert exc.value.status_code == 409
    server._ensure_workspace_available("workspace_b")


def test_index_tasks_are_persisted_and_reloadable(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks"
    monkeypatch.setattr(server, "INDEX_TASKS_DIR", task_dir)
    monkeypatch.setattr(server, "_index_tasks", {})

    task = asyncio.run(
        server._create_index_task(
            "batch",
            ["a.txt"],
            "workspace_a",
            {"chunk_size": 256, "chunk_overlap": 20, "separators": ["\n"]},
        )
    )
    asyncio.run(server._update_index_task(task["task_id"], status="running", current=1))

    persisted = server._load_persisted_index_tasks()[task["task_id"]]
    assert persisted["status"] == "running"
    assert persisted["current"] == 1
    assert persisted["request"]["chunk_size"] == 256


def test_interrupted_task_is_resumed_with_original_parameters(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks"
    monkeypatch.setattr(server, "INDEX_TASKS_DIR", task_dir)
    task = {
        "task_id": "abcdef123456",
        "kind": "batch",
        "workspace": "workspace_a",
        "status": "running",
        "phase": "indexing",
        "doc_names": ["a.txt"],
        "total": 1,
        "current": 0,
        "progress": 0,
        "message": "interrupted",
        "results": [],
        "errors": [],
        "cancel_requested": False,
        "request": {
            "chunk_size": 333,
            "chunk_overlap": 22,
            "separators": ["\n"],
        },
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    monkeypatch.setattr(server, "_index_tasks", {task["task_id"]: task})
    resumed = []

    async def fake_run(task_id, req):
        resumed.append((task_id, req))

    monkeypatch.setattr(server, "_run_index_task", fake_run)

    async def run():
        await server._resume_persisted_index_tasks()
        await asyncio.sleep(0)

    asyncio.run(run())

    assert resumed[0][0] == task["task_id"]
    assert resumed[0][1].chunk_size == 333
    assert resumed[0][1].chunk_overlap == 22


def test_invalid_persisted_task_is_failed_without_aborting_other_recovery(
    tmp_path,
    monkeypatch,
):
    task_dir = tmp_path / "tasks"
    monkeypatch.setattr(server, "INDEX_TASKS_DIR", task_dir)
    bad_task = {
        "task_id": "111111111111",
        "kind": "batch",
        "workspace": "workspace_a",
        "status": "running",
        "phase": "indexing",
        "doc_names": ["../escape.txt"],
        "total": 1,
        "current": 0,
        "progress": 0,
        "message": "interrupted",
        "results": [],
        "errors": [],
        "cancel_requested": False,
        "request": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    monkeypatch.setattr(server, "_index_tasks", {bad_task["task_id"]: bad_task})

    asyncio.run(server._resume_persisted_index_tasks())

    assert server._index_tasks[bad_task["task_id"]]["status"] == "failed"
    assert "恢复任务失败" in server._index_tasks[bad_task["task_id"]]["message"]


def test_workspace_rag_locks_are_isolated(monkeypatch):
    monkeypatch.setattr(server, "_workspace_rag_locks", {})

    first = server._get_workspace_rag_lock("workspace_a")
    same = server._get_workspace_rag_lock("workspace_a")
    other = server._get_workspace_rag_lock("workspace_b")

    assert first is same
    assert first is not other


def test_embedding_signature_blocks_incompatible_existing_index(tmp_path, monkeypatch):
    service = _service(tmp_path)
    service._save_manifest(_manifest_item(indexed=True))
    service.record_embedding_signature(
        {
            "base_url": "https://provider-a.example/v1",
            "model": "embed-a",
            "embed_dim": 1024,
        }
    )
    monkeypatch.setattr(
        service,
        "_runtime_models",
        lambda: {
            "embedding": {
                "base_url": "https://provider-b.example/v1",
                "model": "embed-b",
                "embed_dim": 768,
            }
        },
    )

    compatibility = service.embedding_compatibility()

    assert compatibility["compatible"] is False
    with pytest.raises(RuntimeError, match="重建索引"):
        service.assert_embedding_compatible()


def test_clear_workspace_removes_embedding_signature(tmp_path):
    service = _service(tmp_path)
    service._save_manifest(_manifest_item(indexed=True))
    service.record_embedding_signature(
        {
            "base_url": "https://provider.example/v1",
            "model": "embed-a",
            "embed_dim": 1024,
        }
    )

    asyncio.run(service.clear_workspace())

    assert not service.embedding_meta_path.exists()


def test_delete_endpoint_starts_workspace_cleanup_when_graph_residuals_remain(
    tmp_path,
    monkeypatch,
):
    class FakeService:
        async def delete_document(self, _doc_name):
            return {"doc_id": "doc_123", "doc_name": "example.txt"}

        def find_graph_references(self, **_kwargs):
            return {
                "checked": True,
                "has_residuals": True,
                "node_count": 1,
                "edge_count": 0,
                "nodes": [],
                "edges": [],
            }

    cleanup_calls = []

    async def fake_start(req, **kwargs):
        cleanup_calls.append((req.workspace, kwargs))
        return (
            {
                "task_id": "abcdef123456",
                "kind": "rebuild",
                "workspace": req.workspace,
                "status": "queued",
            },
            {},
        )

    monkeypatch.setattr(server, "get_lightrag_service", lambda _workspace: FakeService())
    monkeypatch.setattr(server, "_ensure_workspace_available", lambda _workspace: None)
    monkeypatch.setattr(server, "_start_workspace_rebuild", fake_start)
    monkeypatch.setattr(server, "_workspace_rag_locks", {})
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(server, "RAW_TEXT_DIR", tmp_path / "raw")

    result = asyncio.run(server.delete_document("example.txt", "workspace_a"))

    assert result["deleted"] == 1
    assert result["cleanup_task"]["task_id"] == "abcdef123456"
    assert cleanup_calls[0][0] == "workspace_a"
    assert cleanup_calls[0][1]["allow_empty"] is True


def test_raw_text_update_marks_index_stale_and_persists_sidecar(tmp_path, monkeypatch):
    source = tmp_path / "uploads" / "workspace_a" / "example.txt"
    source.parent.mkdir(parents=True)
    source.write_text("old text", encoding="utf-8")
    document = SimpleNamespace(
        file_name="example.txt",
        file_type="txt",
        raw_text="old text",
        metadata={},
    )

    class FakeService:
        def register_upload(self, doc):
            doc.metadata["lightrag_doc_id"] = "doc_123"
            return {"doc_id": "doc_123", "index_stale": True}

    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(server, "RAW_TEXT_DIR", tmp_path / "raw")
    monkeypatch.setattr(server, "_uploaded_files", {})
    monkeypatch.setattr(server, "_chunk_cache", {})
    monkeypatch.setattr(server, "_workspace_rag_locks", {})
    monkeypatch.setattr(server, "_ensure_workspace_available", lambda _workspace: None)
    monkeypatch.setattr(server, "_load_doc_for_index", lambda *_args: document)
    monkeypatch.setattr(server, "get_lightrag_service", lambda _workspace: FakeService())

    result = asyncio.run(
        server.update_document_raw_text(
            "example.txt",
            server.RawTextUpdateRequest(raw_text="new managed text"),
            "workspace_a",
        )
    )

    raw_path = server._resolve_raw_text_path("workspace_a", "doc_123")
    assert raw_path.read_text(encoding="utf-8") == "new managed text"
    assert result["index_invalidated"] is False
    assert result["index_stale"] is True


def test_reupload_changed_file_keeps_old_index_until_replacement(
    tmp_path,
    monkeypatch,
):
    upload_root = tmp_path / "uploads"
    destination = upload_root / "workspace_a" / "example.txt"
    destination.parent.mkdir(parents=True)
    destination.write_text("old indexed text", encoding="utf-8")
    class FakeService:
        def _load_manifest(self):
            return {
                "documents": {
                    "doc_123": {
                        "doc_id": "doc_123",
                        "doc_name": "example.txt",
                        "indexed": True,
                    }
                }
            }

        def register_upload(self, doc):
            doc.metadata["lightrag_doc_id"] = "doc_123"
            return {"doc_id": "doc_123", "index_stale": True}

    monkeypatch.setattr(server, "UPLOAD_DIR", upload_root)
    monkeypatch.setattr(server, "RAW_TEXT_DIR", tmp_path / "raw")
    monkeypatch.setattr(server, "_uploaded_files", {})
    monkeypatch.setattr(server, "_chunk_cache", {})
    monkeypatch.setattr(server, "_workspace_rag_locks", {})
    monkeypatch.setattr(server, "_ensure_workspace_available", lambda _workspace: None)
    monkeypatch.setattr(server, "get_lightrag_service", lambda _workspace: FakeService())
    upload = UploadFile(
        filename="example.txt",
        file=BytesIO(b"new indexed text replacement"),
    )

    result = asyncio.run(server.upload_document(upload, "workspace_a"))

    assert destination.read_text(encoding="utf-8") == "new indexed text replacement"
    assert result["index_invalidated"] is False
    assert result["index_stale"] is True


def test_empty_cleanup_rebuild_can_run_while_caller_holds_workspace_lock(
    tmp_path,
    monkeypatch,
):
    class FakeService:
        async def replay_graph_audit(self):
            return {"applied": 0, "skipped": 0, "errors": []}

        async def finalize(self):
            return None

        workspace_dir = tmp_path / "shadow"
        working_dir = tmp_path / "shadow" / "lightrag"

    monkeypatch.setattr(server, "INDEX_TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(server, "_index_tasks", {})
    monkeypatch.setattr(server, "_workspace_rag_locks", {})
    monkeypatch.setattr(server, "_workspace_doc_names_for_rebuild", lambda _workspace: [])
    monkeypatch.setattr(server, "get_lightrag_service", lambda _workspace: FakeService())
    monkeypatch.setattr(server, "_clear_workspace_cache", lambda _workspace: None)
    async def prepare(task, _source):
        return FakeService()
    async def commit(_task, _shadow):
        return None
    monkeypatch.setattr(server, "_prepare_shadow_rebuild", prepare)
    monkeypatch.setattr(server, "_commit_shadow_rebuild", commit)

    async def run():
        lock = server._get_workspace_rag_lock("workspace_a")
        async with lock:
            return await asyncio.wait_for(
                server._start_workspace_rebuild(
                    server.RebuildIndexRequest(workspace="workspace_a"),
                    reason="test cleanup",
                    allow_empty=True,
                    workspace_lock_held=True,
                ),
                timeout=1,
            )

    task, _ = asyncio.run(run())

    assert task["status"] == "succeeded"
