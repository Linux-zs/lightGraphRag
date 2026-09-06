"""Bound KG retries across the OpenAI and LightRAG adapters."""

import asyncio
import time
import uuid
from urllib.parse import urlsplit

from loguru import logger
from openai import APIConnectionError, APIStatusError
from tenacity import stop_after_attempt


async def complete_kg(complete, *, failure_kind, workspace, settings, **kwargs):
    # Keep LightRAG's response handling, but own the retry budget in one place.
    retry_with = getattr(complete, "retry_with", None)
    single_call = retry_with(stop=stop_after_attempt(1), reraise=True) if retry_with else complete
    kwargs = dict(kwargs)
    kwargs["openai_client_configs"] = {
        **kwargs.get("openai_client_configs", {}), "max_retries": 0,
    }
    attempts = max(1, min(3, int(settings.get("kg_llm_max_attempts", 2))))
    delay = max(0.0, min(10.0, float(settings.get("kg_llm_retry_delay", 1))))
    timeout = float(kwargs["timeout"])
    # The SDK worker allows 2 * timeout. Leave room for cleanup before its deadline.
    deadline = time.monotonic() + max(timeout, timeout * 2 - 1)
    call_id = uuid.uuid4().hex[:12]
    endpoint = urlsplit(kwargs.get("base_url") or "").hostname or "unknown"

    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        status = "cancelled"
        error_type = ""
        output_chars = 0
        logger.info(
            "KG API started workspace={} call={} attempt={}/{} model={} host={} json_mode={}",
            workspace, call_id, attempt, attempts, kwargs["model"], endpoint,
            bool(kwargs.get("response_format")),
        )
        try:
            result = await asyncio.wait_for(single_call(**kwargs), min(timeout, max(0, deadline - started)))
            output_chars = len(result) if isinstance(result, str) else 0
            status = "succeeded"
            return result
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            kind = failure_kind(exc)
            transient = (
                kind in {"timeout", "invalid_response"}
                or isinstance(exc, APIConnectionError)
                or isinstance(exc, APIStatusError) and (exc.status_code in {408, 409, 429} or exc.status_code >= 500)
                or error_type == "TransientBadRequestError"
            )
            if not transient or attempt == attempts:
                raise
            fallback = kind == "invalid_response" and kwargs.get("response_format") is not None
            wait = 0 if fallback else delay * 2 ** (attempt - 1)
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    wait = max(wait, float(response.headers.get("retry-after", "0")))
                except ValueError:
                    pass
            if time.monotonic() + wait >= deadline:
                raise
            if fallback:
                kwargs.pop("response_format", None)
                kwargs.pop("entity_extraction", None)
                kwargs["keyword_extraction"] = False
            logger.warning(
                "KG API retry workspace={} call={} reason={} json_fallback={} wait_seconds={}",
                workspace, call_id, error_type, fallback, wait,
            )
        finally:
            logger.info(
                "KG API finished workspace={} call={} attempt={} status={} elapsed_seconds={} error_type={} output_chars={}",
                workspace, call_id, attempt, status, round(time.monotonic() - started, 3), error_type, output_chars,
            )
        await asyncio.sleep(wait)
