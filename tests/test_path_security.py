import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api import server


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "../outside.txt",
        r"..\outside.txt",
        "/tmp/outside.txt",
        r"C:\temp\outside.txt",
        "nested/file.txt",
        "nested\\file.txt",
        "bad\x00name.txt",
    ],
)
def test_safe_leaf_name_rejects_path_components(value):
    with pytest.raises(ValueError):
        server._safe_leaf_name(value)


def test_resolve_upload_path_stays_under_upload_root(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(server, "UPLOAD_DIR", upload_dir)

    resolved = server._resolve_upload_path("文档.txt")

    assert resolved == (upload_dir / "文档.txt").resolve()
    assert resolved.parent == upload_dir.resolve()


@pytest.mark.parametrize("session_id", ["../secrets", r"..\model_profiles", "abc", "g" * 12])
def test_session_path_rejects_non_generated_ids(session_id):
    with pytest.raises(ValueError):
        server._session_path(session_id)


def test_chat_request_rejects_invalid_session_id():
    with pytest.raises(ValidationError):
        server.ChatSendRequest(session_id=r"..\model_profiles", message="test")


def test_index_endpoint_rejects_traversal_before_disk_access():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            server.index_document(
                server.IndexRequest(
                    workspace="tdx_default",
                    file_name=r"..\outside.txt",
                )
            )
        )

    assert exc.value.status_code == 400


def test_delete_session_rejects_traversal_without_deleting(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    outside = tmp_path / "model_profiles.json"
    outside.write_text('{"keep": true}', encoding="utf-8")
    monkeypatch.setattr(server, "SESSIONS_DIR", sessions_dir)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.delete_chat_session(r"..\model_profiles"))

    assert exc.value.status_code == 400
    assert outside.exists()
