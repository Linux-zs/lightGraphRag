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
