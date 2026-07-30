import asyncio

import pytest

from src.api import server
from src.doc_processor.parsers.base_parser import Document
from src.lightrag_service import LightRAGService


def _document(path, text="content"):
    return Document(
        doc_id="parser-id",
        file_name=path.name,
        file_path=str(path),
        file_type=path.suffix.lstrip("."),
        raw_text=text,
        metadata={},
    )


def test_same_file_name_resolves_to_different_workspace_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")

    first = server._resolve_upload_path("same.txt", "workspace_a", create_dir=True)
    second = server._resolve_upload_path("same.txt", "workspace_b", create_dir=True)

    assert first != second
    assert first.parent.name == "workspace_a"
    assert second.parent.name == "workspace_b"


def test_legacy_upload_is_copied_without_removing_original(tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    legacy = upload_root / "legacy.txt"
    legacy.write_text("legacy content", encoding="utf-8")
    monkeypatch.setattr(server, "UPLOAD_DIR", upload_root)

    migrated = server._resolve_upload_path(
        "legacy.txt",
        "workspace_a",
        migrate_legacy=True,
    )

    assert migrated.read_text(encoding="utf-8") == "legacy content"
    assert legacy.exists()


def test_clear_workspace_sources_does_not_touch_other_workspace(tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    raw_root = tmp_path / "raw"
    monkeypatch.setattr(server, "UPLOAD_DIR", upload_root)
    monkeypatch.setattr(server, "RAW_TEXT_DIR", raw_root)
    monkeypatch.setattr(server, "_uploaded_files", {})
    monkeypatch.setattr(server, "_chunk_cache", {})

    source_a = server._resolve_upload_path("same.txt", "workspace_a", create_dir=True)
    source_b = server._resolve_upload_path("same.txt", "workspace_b", create_dir=True)
    source_a.write_text("A", encoding="utf-8")
    source_b.write_text("B", encoding="utf-8")
    raw_a = server._resolve_raw_text_path("workspace_a", "doc_a", create_dir=True)
    raw_b = server._resolve_raw_text_path("workspace_b", "doc_b", create_dir=True)
    raw_a.write_text("A", encoding="utf-8")
    raw_b.write_text("B", encoding="utf-8")

    removed = server._remove_workspace_sources("workspace_a")

    assert removed == 2
    assert not source_a.exists()
    assert not raw_a.exists()
    assert source_b.read_text(encoding="utf-8") == "B"
    assert raw_b.read_text(encoding="utf-8") == "B"


def test_manifest_reuses_doc_id_when_workspace_path_changes(tmp_path):
    config = {
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "lightrag_dir": str(tmp_path / "lightrag"),
        },
        "lightrag": {"workspace": "default"},
    }
    service = LightRAGService(config=config, workspace="workspace_a")
    first = _document(tmp_path / "legacy" / "same.txt")
    second = _document(tmp_path / "uploads" / "workspace_a" / "same.txt")

    first_item = service.register_upload(first)
    second_item = service.register_upload(second)

    assert second_item["doc_id"] == first_item["doc_id"]
    assert len(service._load_manifest()["documents"]) == 1


def test_chat_sessions_are_filtered_and_rejected_across_workspaces(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(server, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(server, "_session_locks", {})

    created_a = asyncio.run(
        server.create_chat_session(server.ChatSessionCreateRequest(workspace="workspace_a"))
    )
    created_b = asyncio.run(
        server.create_chat_session(server.ChatSessionCreateRequest(workspace="workspace_b"))
    )

    listed_a = asyncio.run(server.list_chat_sessions("workspace_a"))
    listed_b = asyncio.run(server.list_chat_sessions("workspace_b"))
    assert [item.id for item in listed_a] == [created_a["id"]]
    assert [item.id for item in listed_b] == [created_b["id"]]

    with pytest.raises(ValueError, match="different workspace"):
        server._ensure_session(created_a["id"], "question", "workspace_b")


def test_workspace_metadata_cleanup_removes_only_target_workspace(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    sessions_dir = data_dir / "sessions"
    settings_dir = data_dir / "workspace_settings"
    index_tasks_dir = data_dir / "index_tasks"
    sessions_dir.mkdir(parents=True)
    settings_dir.mkdir(parents=True)
    index_tasks_dir.mkdir(parents=True)

    monkeypatch.setattr(server, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(server, "WORKSPACE_SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(server, "INDEX_TASKS_DIR", index_tasks_dir)
    monkeypatch.setattr(server, "_session_locks", {})

    service = LightRAGService(
        config={
            "paths": {
                "data_dir": str(data_dir),
                "lightrag_dir": str(data_dir / "lightrag"),
            },
            "lightrag": {"workspace": "tdx_default"},
        },
        workspace="workspace_a",
    )

    server._save_workspace_settings("workspace_a", {"answer_system_prompt": "A"})
    server._save_workspace_settings("workspace_b", {"answer_system_prompt": "B"})
    created_a = asyncio.run(server.create_chat_session(server.ChatSessionCreateRequest(workspace="workspace_a")))
    created_b = asyncio.run(server.create_chat_session(server.ChatSessionCreateRequest(workspace="workspace_b")))
    service.graph_governance_path.parent.mkdir(parents=True, exist_ok=True)
    service.graph_governance_path.write_text("{}", encoding="utf-8")
    service.graph_reference_dir.mkdir(parents=True, exist_ok=True)
    (service.graph_reference_dir / "ref.txt").write_text("ref", encoding="utf-8")
    task_path = index_tasks_dir / "abc123abc123.json"
    task_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        server,
        "_index_tasks",
        {
            "abc123abc123": {"task_id": "abc123abc123", "workspace": "workspace_a"},
            "def456def456": {"task_id": "def456def456", "workspace": "workspace_b"},
        },
    )

    result = asyncio.run(server._remove_workspace_metadata("workspace_a", service))

    assert result["removed_settings"] == 1
    assert result["removed_graph_config"] == 1
    assert result["removed_graph_reference_files"] == 1
    assert result["removed_sessions"] == 1
    assert result["removed_index_tasks"] == 1
    assert not server._workspace_settings_path("workspace_a").exists()
    assert server._workspace_settings_path("workspace_b").exists()
    assert not server._session_path(created_a["id"]).exists()
    assert server._session_path(created_b["id"]).exists()
    assert "abc123abc123" not in server._index_tasks
    assert "def456def456" in server._index_tasks
