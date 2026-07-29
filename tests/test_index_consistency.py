import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

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
