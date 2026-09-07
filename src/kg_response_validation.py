"""Validate empty extraction results using the same parsers as the SDK."""

import json_repair
from lightrag.llm.openai import InvalidResponseError
from lightrag.operate import _process_extraction_result, _process_json_extraction_result
from lightrag.prompt import PROMPTS
from lightrag.utils import remove_think_tags
from loguru import logger


async def validate_extraction_response(response: str, format: str) -> None:
    if not isinstance(response, str) or not response.strip():
        raise InvalidResponseError("KG extraction response is empty")
    text = remove_think_tags(response).strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if format == "json":
        parsed = json_repair.loads(text)
        if not isinstance(parsed, dict) or not any(key in parsed for key in ("entities", "relationships")):
            raise InvalidResponseError("KG extraction response has no entity/relationship arrays")
        arrays = [parsed.get(key, []) for key in ("entities", "relationships")]
        if any(not isinstance(value, list) for value in arrays):
            raise InvalidResponseError("KG entity/relationship fields must be arrays")
        if not any(arrays):
            return  # Explicit, valid empty extraction is not a model failure.
        nodes, edges = await _process_json_extraction_result(text, "validation", 0)
    else:
        completion = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        if text == completion:
            return
        nodes, edges = await _process_extraction_result(
            text, "validation", 0,
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=completion,
        )
    if not nodes and not edges:
        raise InvalidResponseError("KG response contained records but none could be parsed")


class ExtractionCacheView:
    def __init__(self, storage, format: str):
        self.storage = storage
        self.format = format

    def __getattr__(self, name):
        return getattr(self.storage, name)

    async def get_by_id(self, key):
        row = await self.storage.get_by_id(key)
        if row and str(key).startswith("default:extract:"):
            try:
                await validate_extraction_response(row.get("return"), self.format)
            except (InvalidResponseError, ValueError, TypeError):
                logger.warning("Ignoring malformed KG extraction cache key={}", key)
                return None
        return row
