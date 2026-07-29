import asyncio

from src.api import server


def test_answer_stream_forwards_provider_tokens_and_configured_parameters(monkeypatch):
    calls = {}

    class FakeBackend:
        def __init__(self, config):
            calls["backend_config"] = config

        async def chat_stream(self, **kwargs):
            calls["generation"] = kwargs
            yield "第一段"
            await asyncio.sleep(0)
            yield "第二段"

        async def close(self):
            calls["closed"] = True

    runtime = {
        "chat": {
            "base_url": "http://127.0.0.1:9999/v1",
            "api_key": "",
            "model": "chat-model",
            "timeout": 12,
            "temperature": 1.1,
            "top_p": 0.72,
            "max_tokens": 1234,
            "frequency_penalty": 0.4,
            "presence_penalty": -0.2,
        }
    }
    monkeypatch.setattr(server, "SiliconFlowBackend", FakeBackend)
    monkeypatch.setattr(server, "get_runtime_model_config", lambda _cfg: runtime)
    monkeypatch.setattr(server, "get_config", lambda: {})
    monkeypatch.setattr(
        server,
        "_load_workspace_settings",
        lambda _workspace: {"answer_system_prompt": "system"},
    )

    async def collect():
        return [
            token
            async for token in server._stream_answer_text(
                "问题",
                [{"index": 1, "doc_name": "a.txt", "chunk_index": 0, "excerpt": "依据"}],
                [],
                "workspace_a",
            )
        ]

    assert asyncio.run(collect()) == ["第一段", "第二段"]
    assert calls["generation"]["temperature"] == 1.1
    assert calls["generation"]["top_p"] == 0.72
    assert calls["generation"]["max_tokens"] == 1234
    assert calls["generation"]["frequency_penalty"] == 0.4
    assert calls["generation"]["presence_penalty"] == -0.2
    assert calls["closed"] is True


def test_answer_stream_falls_back_if_provider_fails_before_first_token(monkeypatch):
    class FakeBackend:
        def __init__(self, _config):
            pass

        async def chat_stream(self, **_kwargs):
            if False:
                yield ""
            raise RuntimeError("provider unavailable")

        async def close(self):
            pass

    monkeypatch.setattr(server, "SiliconFlowBackend", FakeBackend)
    monkeypatch.setattr(
        server,
        "get_runtime_model_config",
        lambda _cfg: {
            "chat": {
                "base_url": "http://127.0.0.1/v1",
                "api_key": "",
                "model": "chat-model",
            }
        },
    )
    monkeypatch.setattr(server, "get_config", lambda: {})

    async def collect():
        return "".join(
            [
                token
                async for token in server._stream_answer_text(
                    "question",
                    [{
                        "index": 1,
                        "doc_name": "a.txt",
                        "chunk_index": 0,
                        "excerpt": "reference fact provides enough detail for a deterministic fallback.",
                    }],
                    [],
                    "workspace_a",
                )
            ]
        )

    answer = asyncio.run(collect())
    assert "reference fact provides enough detail" in answer
    assert "[1]" in answer
