import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from lightrag.llm.openai import InvalidResponseError
from lightrag.prompt import PROMPTS

from src.kg_request_policy import complete_kg
from src.kg_response_validation import ExtractionCacheView, validate_extraction_response
from src.lightrag_service import LightRAGService


@pytest.mark.parametrize("text", [
    '{"entities":[],"relationships":[]}',
    '```json\n{"entities":[],"relationships":[]}\n```',
    '{"entities":[{"name":"2024","type":"year","description":"A valid year"}],"relationships":[]}',
])
def test_valid_empty_and_numeric_entities_are_accepted(text):
    asyncio.run(validate_extraction_response(text, "json"))


@pytest.mark.parametrize("text", ['{}', 'not JSON', '{"entities":"none"}', '{"entities":[{"name":"missing type and description"}]}'])
def test_malformed_or_unparseable_records_are_not_successful_empty(text):
    with pytest.raises(InvalidResponseError):
        asyncio.run(validate_extraction_response(text, "json"))


def test_invalid_shape_uses_existing_bounded_retry_and_never_leaks_internal_argument():
    calls = []

    async def complete(**kwargs):
        calls.append(kwargs)
        return '{}' if len(calls) == 1 else '{"entities":[],"relationships":[]}'

    result = asyncio.run(complete_kg(
        complete, failure_kind=LightRAGService._kg_failure_kind, workspace="test",
        settings={"kg_llm_retry_delay": 0}, timeout=1, model="mock",
        response_format={"type": "json_object"}, _kg_extraction_format="json",
    ))
    assert result == '{"entities":[],"relationships":[]}'
    assert len(calls) == 2
    assert all("_kg_extraction_format" not in call for call in calls)


def test_invalid_cached_extraction_is_ignored_but_summary_is_untouched():
    async def run():
        storage = SimpleNamespace(get_by_id=AsyncMock(return_value={"return": "{}"}))
        cache = ExtractionCacheView(storage, "json")
        assert await cache.get_by_id("default:extract:old") is None
        assert await cache.get_by_id("default:summary:old") == {"return": "{}"}
        storage.get_by_id.return_value = {"return": '{"entities":[],"relationships":[]}'}
        assert await cache.get_by_id("default:extract:empty") is not None
    asyncio.run(run())


def test_text_extraction_distinguishes_completion_marker_from_invalid_text():
    asyncio.run(validate_extraction_response(PROMPTS["DEFAULT_COMPLETION_DELIMITER"], "text"))
    with pytest.raises(InvalidResponseError):
        asyncio.run(validate_extraction_response("nothing structured here", "text"))


def test_zero_parsed_records_are_reported_as_no_entities(tmp_path):
    service = LightRAGService({"paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "rag")}}, workspace="empty")

    async def run():
        stats = {"kept": 1, "total": 1, "skipped": 0}
        await service._extract_entities_with_recovery(AsyncMock(return_value=[({}, {})]), {"chunk": {}}, (), {}, stats)
        assert service._kg_status_for_success(skip_kg=False) == "no_entities"
        assert stats["entity_count"] == stats["relation_count"] == 0
        assert stats["empty_chunks"] == ["chunk"]
    asyncio.run(run())
