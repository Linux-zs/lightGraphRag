import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from src.lightrag_service import LightRAGService
from src.kg_request_policy import complete_kg


@pytest.mark.parametrize("case,expected", [("server", 2), ("empty", 2), ("auth", 1), ("success", 1)])
def test_real_sdk_request_budget(monkeypatch, case, expected):
    import lightrag.llm.openai as sdk

    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if case in {"server", "auth"}:
            return httpx.Response(500 if case == "server" else 401, json={"error": {"message": "synthetic"}})
        return httpx.Response(200, json={
            "id": "probe", "object": "chat.completion", "created": 0, "model": "probe",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "" if case == "empty" else '{"entities":[],"relationships":[]}'}, "finish_reason": "stop"}],
        })

    def factory(**kwargs):
        return AsyncOpenAI(api_key="synthetic", base_url="https://synthetic.invalid/v1",
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
                           max_retries=kwargs.get("client_configs", {}).get("max_retries", 2))

    monkeypatch.setattr(sdk, "create_openai_async_client", factory)
    service = LightRAGService.__new__(LightRAGService)
    service.config = {"lightrag": {"kg_llm_retry_delay": 0}}
    service.workspace = "test"
    service._runtime_models = lambda: {"kg": {"model": "probe", "base_url": "https://synthetic.invalid/v1", "api_key": "synthetic", "timeout": 1}}

    async def run():
        call = service._make_kg_llm_func()
        if case == "success":
            assert await call("synthetic", response_format={"type": "json_object"}) == '{"entities":[],"relationships":[]}'
        else:
            with pytest.raises(Exception):
                await call("synthetic", response_format={"type": "json_object"})

    asyncio.run(run())
    assert len(requests) == expected
    if case == "empty":
        assert "response_format" in requests[0]
        assert "response_format" not in requests[1]


def policy(complete, **kwargs):
    return complete_kg(
        complete, failure_kind=LightRAGService._kg_failure_kind,
        workspace="test", settings={"kg_llm_retry_delay": 0}, model="probe",
        prompt="private text", api_key="private key", timeout=1, **kwargs,
    )


def test_transient_failure_recovers_without_changing_payload():
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise http_error(503)
        return "valid extraction"

    assert asyncio.run(policy(complete)) == "valid extraction"
    assert calls[0] == calls[1]
    assert calls[0]["openai_client_configs"]["max_retries"] == 0


def http_error(status, headers=None):
    from openai import APIStatusError
    response = httpx.Response(status, headers=headers, request=httpx.Request("POST", "https://synthetic.invalid"))
    return APIStatusError("synthetic", response=response, body=None)


def test_retry_after_does_not_exceed_total_budget():
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs)
        raise http_error(429, {"retry-after": "120"})

    with pytest.raises(Exception, match="synthetic"):
        asyncio.run(policy(complete))
    assert len(calls) == 1


def test_cancelled_call_is_not_retried_and_is_drained():
    async def run():
        started, drained = asyncio.Event(), asyncio.Event()
        calls = []

        async def complete(**kwargs):
            calls.append(kwargs)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()

        task = asyncio.create_task(policy(complete))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert drained.is_set()
        assert len(calls) == 1

    asyncio.run(run())


def test_telemetry_does_not_include_prompt_or_credentials():
    from loguru import logger
    records = []
    sink = logger.add(lambda message: records.append(str(message)), format="{message}")

    async def complete(**kwargs):
        return "private response"

    try:
        asyncio.run(policy(complete, base_url="https://user:password@synthetic.invalid/v1?key=secret"))
    finally:
        logger.remove(sink)
    combined = "\n".join(records)
    assert "KG API started" in combined and "elapsed_seconds=" in combined
    for sensitive in ["private text", "private key", "private response", "password", "key=secret"]:
        assert sensitive not in combined


def test_total_timeout_drains_request_without_restarting_it():
    async def run():
        drained = asyncio.Event()
        calls = []

        async def complete(**kwargs):
            calls.append(kwargs)
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()

        with pytest.raises(TimeoutError):
            await complete_kg(
                complete, failure_kind=LightRAGService._kg_failure_kind,
                workspace="test", settings={"kg_llm_retry_delay": 0},
                model="probe", prompt="synthetic", timeout=0.02,
            )
        assert drained.is_set()
        assert len(calls) == 1

    asyncio.run(run())


def test_chat_does_not_use_kg_retry_policy(monkeypatch):
    from src import lightrag_service
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs)
        return "answer"

    monkeypatch.setattr(lightrag_service, "openai_complete_if_cache", complete)
    service = LightRAGService.__new__(LightRAGService)
    service._runtime_models = lambda: {"chat": {"model": "probe", "base_url": "https://synthetic.invalid", "api_key": "synthetic"}}
    assert asyncio.run(service._make_llm_func()("question")) == "answer"
    assert "openai_client_configs" not in calls[0]
