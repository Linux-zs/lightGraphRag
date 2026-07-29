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
