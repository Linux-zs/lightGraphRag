import asyncio

import httpx

from src.api import server


async def _request(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def test_health_contract(monkeypatch):
    monkeypatch.setattr(server, "_index_tasks", {})
    response = asyncio.run(_request("GET", "/api/health"))

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_sessions_are_scoped_by_workspace_over_http(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(server, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(server, "_session_locks", {})
    monkeypatch.setattr(server, "_index_tasks", {})

    async def exercise():
        transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            created = await client.post(
                "/api/chat/sessions",
                json={"workspace": "workspace_a"},
            )
            own_list = await client.get(
                "/api/chat/sessions",
                params={"workspace": "workspace_a"},
            )
            other_list = await client.get(
                "/api/chat/sessions",
                params={"workspace": "workspace_b"},
            )
            return created, own_list, other_list

    created, own_list, other_list = asyncio.run(exercise())

    assert created.status_code == 200
    assert [item["id"] for item in own_list.json()] == [created.json()["id"]]
    assert other_list.json() == []


def test_recall_returns_conflict_for_incompatible_embedding(monkeypatch):
    class IncompatibleService:
        def assert_embedding_compatible(self):
            raise RuntimeError("当前嵌入模型与该知识库已有索引不兼容，请先重建索引。")

    monkeypatch.setattr(server, "_index_tasks", {})
    monkeypatch.setattr(
        server,
        "get_lightrag_service",
        lambda _workspace: IncompatibleService(),
    )

    response = asyncio.run(
        _request(
            "POST",
            "/api/recall/test",
            json={"workspace": "workspace_a", "query": "question"},
        )
    )

    assert response.status_code == 409
    assert "重建索引" in response.json()["detail"]


def test_public_index_task_includes_default_stage_timings():
    public = server._public_index_task(
        {
            "task_id": "abc123abc123",
            "status": "running",
            "cancel_requested": False,
        }
    )

    assert "cancel_requested" not in public
    assert public["current_stage"] == ""
    assert public["current_stage_started_at"] == ""
    assert public["stage_timings"] == {
        "parse": 0.0,
        "chunk_vector": 0.0,
        "kg": 0.0,
        "merge": 0.0,
    }


def test_index_stage_progression_is_monotonic():
    assert server._should_advance_index_stage("", "chunk_vector") is True
    assert server._should_advance_index_stage("chunk_vector", "kg") is True
    assert server._should_advance_index_stage("kg", "merge") is True
    assert server._should_advance_index_stage("kg", "chunk_vector") is False
    assert server._should_advance_index_stage("chunk_vector", "chunk_vector") is False


def test_system_logs_returns_filtered_tail(tmp_path, monkeypatch):
    log_path = tmp_path / "app.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-01-01 00:00:00 | INFO | startup ok",
                "2026-01-01 00:00:01 | WARNING | answer_quality_check action=retry",
                "2026-01-01 00:00:02 | ERROR | model failed",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "APP_LOG_PATH", log_path)

    response = asyncio.run(
        _request(
            "GET",
            "/api/system/logs",
            params={"level": "WARNING", "contains": "answer_quality", "limit": 10},
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is True
    assert body["total_matched"] == 1
    assert body["items"][0]["level"] == "WARNING"
    assert "answer_quality_check" in body["items"][0]["text"]
