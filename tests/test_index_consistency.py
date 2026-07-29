import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from src.api import server
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

    assert service._load_manifest() == original


def test_clear_workspace_can_preserve_manifest_for_rebuild(tmp_path):
    service = _service(tmp_path)
    manifest = _manifest_item(indexed=True)
    service._save_manifest(manifest)
    service.workspace_dir.mkdir(parents=True)
    (service.workspace_dir / "index.json").write_text("{}", encoding="utf-8")

    result = asyncio.run(service.clear_workspace(preserve_manifest=True))

    assert result["preserved_manifest"] is True
    assert service._load_manifest() == manifest
    assert not (service.workspace_dir / "index.json").exists()


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


def test_raw_text_update_invalidates_index_and_persists_sidecar(tmp_path, monkeypatch):
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
        async def invalidate_document(self, _doc_name):
            return {"doc_id": "doc_123", "doc_name": "example.txt"}

        def register_upload(self, doc):
            assert doc.metadata["lightrag_doc_id"] == "doc_123"
            return {"doc_id": "doc_123"}

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
    assert result["index_invalidated"] is True


def test_reupload_changed_file_invalidates_old_index_before_replacement(
    tmp_path,
    monkeypatch,
):
    upload_root = tmp_path / "uploads"
    destination = upload_root / "workspace_a" / "example.txt"
    destination.parent.mkdir(parents=True)
    destination.write_text("old indexed text", encoding="utf-8")
    invalidated = []

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

        async def invalidate_document(self, doc_name):
            invalidated.append(doc_name)
            return {"doc_id": "doc_123", "doc_name": doc_name}

        def register_upload(self, doc):
            doc.metadata["lightrag_doc_id"] = "doc_123"
            return {"doc_id": "doc_123"}

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

    assert invalidated == ["example.txt"]
    assert destination.read_text(encoding="utf-8") == "new indexed text replacement"
    assert result["index_invalidated"] is True


def test_empty_cleanup_rebuild_can_run_while_caller_holds_workspace_lock(
    tmp_path,
    monkeypatch,
):
    class FakeService:
        async def clear_workspace(self, **_kwargs):
            return {"removed_workspace": True}

        async def replay_graph_audit(self):
            return {"applied": 0, "skipped": 0, "errors": []}

    monkeypatch.setattr(server, "INDEX_TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(server, "_index_tasks", {})
    monkeypatch.setattr(server, "_workspace_rag_locks", {})
    monkeypatch.setattr(server, "_workspace_doc_names_for_rebuild", lambda _workspace: [])
    monkeypatch.setattr(server, "get_lightrag_service", lambda _workspace: FakeService())
    monkeypatch.setattr(server, "reset_lightrag_service", lambda _workspace: None)
    monkeypatch.setattr(server, "_clear_workspace_cache", lambda _workspace: None)

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
