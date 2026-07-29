from pathlib import Path

from src.api import server


class FakeService:
    def __init__(self, manifest):
        self._manifest = manifest

    def _load_manifest(self):
        return self._manifest


def test_rebuild_uses_only_workspace_manifest_documents(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "current.txt").write_text("current", encoding="utf-8")
    (upload_dir / "other.txt").write_text("other", encoding="utf-8")

    manifest = {
        "documents": {
            "doc_current": {
                "doc_name": "current.txt",
                "updated_at": "2026-07-29T10:00:00+08:00",
            }
        }
    }

    monkeypatch.setattr(server, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(server, "get_lightrag_service", lambda workspace: FakeService(manifest))

    assert server._workspace_doc_names_for_rebuild("workspace_a") == ["current.txt"]


def test_rebuild_skips_manifest_documents_missing_from_upload_dir(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    manifest = {
        "documents": {
            "doc_missing": {
                "doc_name": "missing.txt",
                "updated_at": "2026-07-29T10:00:00+08:00",
            }
        }
    }

    monkeypatch.setattr(server, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(server, "get_lightrag_service", lambda workspace: FakeService(manifest))

    assert server._workspace_doc_names_for_rebuild("workspace_a") == []
