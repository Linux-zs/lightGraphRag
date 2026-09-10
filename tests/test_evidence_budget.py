import pytest

from src.evidence_budget import allocate_evidence_budget


def test_unavailable_tokenizer_uses_bounded_unicode_safe_byte_fallback(monkeypatch):
    import src.evidence_budget as budget
    monkeypatch.setattr(budget, 'evidence_tokenizer', lambda: None)
    sources = ['中文🙂' * 30, 'short']
    result = budget.bound_evidence_tokens(sources, 30)
    assert sum(len(text.encode('utf-8')) for text in result) <= 30
    assert all(source.startswith(text) for source, text in zip(sources, result))
    assert result[1] == 'short'


def test_tokenizer_initialization_failure_is_cached(monkeypatch):
    import tiktoken
    import src.evidence_budget as budget
    from unittest.mock import Mock
    failure = Mock(side_effect=OSError('offline'))
    monkeypatch.setattr(tiktoken, 'get_encoding', failure)
    budget.evidence_tokenizer.cache_clear()
    try:
        assert budget.evidence_tokenizer() is None
        assert budget.evidence_tokenizer() is None
        failure.assert_called_once()
    finally:
        budget.evidence_tokenizer.cache_clear()


def test_token_budget_preserves_unicode_and_treats_special_spellings_as_data():
    from src.evidence_budget import bound_evidence_tokens, evidence_tokenizer
    sources = ['中文配置🙂' * 100, '<|endoftext|> sample']
    result = bound_evidence_tokens(sources, budget=35)
    tokenizer = evidence_tokenizer()
    assert all(source.startswith(text) for source, text in zip(sources, result))
    assert all('\ufffd' not in text for text in result)
    assert sum(len(tokenizer.encode(text, disallowed_special=())) for text in result) <= 35


def test_short_sources_release_unused_budget_to_long_sources():
    assert allocate_evidence_budget([10, 1000, 10], 120) == [10, 100, 10]
    assert allocate_evidence_budget([0, 0, 100], 50) == [0, 0, 50]


@pytest.mark.parametrize('lengths,budget', [([], 0), ([10, 20], 50), ([100] * 100, 7), ([100, 200, 300], 200)])
def test_allocations_use_available_budget_without_exceeding_content(lengths, budget):
    allocation = allocate_evidence_budget(lengths, budget)
    assert sum(allocation) == min(sum(lengths), budget)
    assert all(0 <= used <= available for used, available in zip(allocation, lengths))


def test_long_answer_source_is_not_truncated_when_short_sources_leave_room(monkeypatch):
    from src.api import server
    monkeypatch.setattr(server, '_load_workspace_settings', lambda _: {})
    sources = [{'index': 1, 'answer_content': 'background ' * 1200 + '末尾关键结论'},
               {'index': 2, 'answer_content': '短证据'}]
    prompt = server._build_answer_messages('问题', sources, [])[-1]['content']
    assert '末尾关键结论' in prompt
    assert '短证据' in prompt
