import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.evidence_budget import ContextBudgetExceeded, check_context_budget


def test_complete_request_and_output_are_counted_without_mutation(monkeypatch):
    import src.evidence_budget as budget
    monkeypatch.setattr(budget, 'evidence_tokenizer', lambda: None)
    messages = [{'role': 'system', 'content': '规则'},
                {'role': 'user', 'content': 'history'},
                {'role': 'user', 'content': 'question\nsource\ngraph'}]
    original = [dict(message) for message in messages]
    count = check_context_budget(messages, 10000, 100)
    assert count == sum(len(value.encode('utf-8')) for message in messages for value in message.values()) + 304
    assert check_context_budget(messages, count + 100, 100) == count
    with pytest.raises(ContextBudgetExceeded):
        check_context_budget(messages, count + 99, 100)
    assert messages == original


@pytest.mark.parametrize('stream', [False, True])
def test_over_budget_request_never_calls_provider_or_citation_fallback(monkeypatch, stream):
    from src.api import server
    backend = SimpleNamespace(chat=AsyncMock(), chat_stream=Mock(), close=AsyncMock())
    monkeypatch.setattr(server, 'SiliconFlowBackend', lambda _: backend)
    monkeypatch.setattr(server, '_answer_runtime', lambda _: {'base_url': '', 'api_key': '', 'model': 'fake'})
    monkeypatch.setattr(server, '_load_workspace_settings', lambda _: {'answer_system_prompt': 'long ' * 3000})
    fallback = Mock(side_effect=AssertionError('must not disguise budget error as an answer'))
    monkeypatch.setattr(server, '_fallback_answer_from_citations', fallback)
    settings = server.ChatSettings(context_window=1024, max_tokens=64)
    args = ('question', [{'index': 1, 'answer_content': 'source'}], [])

    async def run():
        if stream:
            return ''.join([part async for part in server._stream_answer_text(*args, settings=settings)])
        return await server._generate_answer_text(*args, settings=settings)

    assert '上下文超出当前预算' in asyncio.run(run())
    backend.chat.assert_not_called()
    backend.chat_stream.assert_not_called()
    backend.close.assert_awaited_once()
    fallback.assert_not_called()


def test_context_budget_round_trips_in_session_settings(monkeypatch):
    from src.api import server
    monkeypatch.setattr(server, '_default_chat_settings', lambda: server.ChatSettings())
    settings = server._session_chat_settings({'settings': {'context_window': 16384}})
    assert settings.model_dump()['context_window'] == 16384


def test_retry_instructions_are_checked_before_second_provider_call(monkeypatch):
    import src.evidence_budget as budget
    from src.api import server
    monkeypatch.setattr(budget, 'evidence_tokenizer', lambda: None)
    backend = SimpleNamespace(chat=AsyncMock(return_value=SimpleNamespace(content='bad')), close=AsyncMock())
    monkeypatch.setattr(server, 'SiliconFlowBackend', lambda _: backend)
    monkeypatch.setattr(server, '_answer_runtime', lambda _: {'base_url': '', 'api_key': '', 'model': 'fake'})
    monkeypatch.setattr(server, '_build_answer_messages', lambda *args: [{'role': 'user', 'content': 'x' * 684}])
    monkeypatch.setattr(server, '_generated_answer_quality_issues', lambda _: ['incomplete'])
    monkeypatch.setattr(server, '_salvage_generated_answer', lambda _: ('', ['incomplete']))
    settings = server.ChatSettings(context_window=1024, max_tokens=64)
    answer = asyncio.run(server._generate_answer_text('q', [{'index': 1}], [], settings=settings))
    assert '上下文超出当前预算' in answer
    backend.chat.assert_awaited_once()
    backend.close.assert_awaited_once()
