import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.api import server
from src.lightrag_service import LightRAGService


def test_legacy_uploads_are_split_by_workspace_without_data_loss(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    raw_dir = tmp_path / "upload_text"
    upload_dir.mkdir()
    raw_dir.mkdir()
    shared = upload_dir / "shared.txt"
    shared.write_text("shared content", encoding="utf-8")
    orphan = upload_dir / "orphan.txt"
    orphan.write_text("orphan content", encoding="utf-8")

    manifests = {
        "alpha": {
            "documents": {
                "doc_alpha": {"doc_name": "shared.txt", "file_path": "data/uploads/shared.txt"}
            }
        },
        "beta": {
            "documents": {
                "doc_beta": {"doc_name": "shared.txt", "file_path": "data/uploads/shared.txt"}
            }
        },
    }

    class FakeService:
        def __init__(self, workspace):
            self.workspace = workspace

        def _load_manifest(self):
            return manifests[self.workspace]

        def _save_manifest(self, manifest):
            manifests[self.workspace] = manifest

    monkeypatch.setattr(server, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(server, "RAW_TEXT_DIR", raw_dir)
    monkeypatch.setattr(server, "_discover_workspaces", lambda: ["alpha", "beta"])
    monkeypatch.setattr(server, "get_lightrag_service", lambda workspace: FakeService(workspace))

    result = server._migrate_legacy_source_layout()

    assert result == {"migrated": 2, "unassigned": 1}
    assert (upload_dir / "alpha" / "shared.txt").read_text(encoding="utf-8") == "shared content"
    assert (upload_dir / "beta" / "shared.txt").read_text(encoding="utf-8") == "shared content"
    assert not shared.exists()
    assert (upload_dir / "_legacy_unassigned" / "orphan.txt").read_text(encoding="utf-8") == "orphan content"
    assert manifests["alpha"]["documents"]["doc_alpha"]["file_path"].endswith(
        str(Path("alpha") / "shared.txt")
    )


def test_old_session_inherits_current_chat_defaults(monkeypatch):
    monkeypatch.setattr(server, "get_config", lambda: {})
    monkeypatch.setattr(
        server,
        "get_bindings",
        lambda _config: {"chat": {"profile_id": "profile-a", "model": "model-a"}},
    )
    monkeypatch.setattr(
        server,
        "get_runtime_model_config",
        lambda _config: {
            "chat": {
                "model": "model-a",
                "temperature": 0.4,
                "top_p": 0.8,
                "max_tokens": 2048,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
            }
        },
    )

    settings = server._session_chat_settings({"title": "legacy", "messages": []})

    assert settings.answer_profile_id == "profile-a"
    assert settings.answer_model == "model-a"
    assert settings.temperature == 0.4
    assert settings.mode == "mix"


def test_session_title_uses_model_summary(monkeypatch):
    calls = {}

    class FakeBackend:
        def __init__(self, config):
            calls["config"] = config

        async def chat(self, **kwargs):
            calls["request"] = kwargs
            return SimpleNamespace(content="  蝴蝶效应风险分析  ")

        async def close(self):
            calls["closed"] = True

    monkeypatch.setattr(server, "SiliconFlowBackend", FakeBackend)
    monkeypatch.setattr(
        server,
        "_answer_runtime",
        lambda _settings: {
            "base_url": "http://127.0.0.1:9999/v1",
            "api_key": "",
            "model": "answer-model",
            "timeout": 12,
        },
    )

    title = asyncio.run(
        server._generate_session_title(
            "C医院储备的药物是否存在风险？",
            "存在供应链中断风险。",
            server.ChatSettings(answer_profile_id="profile-a", answer_model="answer-model"),
        )
    )

    assert title == "蝴蝶效应风险分析"
    assert calls["config"]["chat_model"] == "answer-model"
    assert calls["closed"] is True


def test_text_recall_returns_vector_and_rerank_positions():
    class FakeVectorStore:
        cosine_better_than_threshold = 0.2

        async def query(self, _query, top_k):
            assert top_k == 2
            return [
                {"id": "a", "content": "alpha", "file_path": "a.txt", "distance": 0.91},
                {"id": "b", "content": "beta", "file_path": "b.txt", "distance": 0.83},
            ]

    class FakeService:
        workspace = "workspace-a"

        def assert_embedding_compatible(self):
            return None

        async def get_rag(self):
            return SimpleNamespace(chunks_vdb=FakeVectorStore())

        def _runtime_models(self):
            return {"rerank": {"enabled": True, "api_key": "secret"}}

        def _make_rerank_func(self):
            async def rerank(_query, _documents, top_n):
                assert top_n == 2
                return [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.40},
                ]

            return rerank

    result = asyncio.run(
        LightRAGService.text_recall(
            FakeService(),
            "query",
            top_k=2,
            enable_rerank=True,
        )
    )

    assert result["rerank_applied"] is True
    assert [item["chunk_id"] for item in result["vector_hits"]] == ["a", "b"]
    assert [item["chunk_id"] for item in result["rerank_hits"]] == ["b", "a"]
    assert result["rerank_hits"][0]["vector_rank"] == 2
    assert result["rerank_hits"][0]["rerank_rank"] == 1
