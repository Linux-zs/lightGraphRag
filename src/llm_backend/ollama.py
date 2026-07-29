"""Ollama LLM backend implementation (chat only; embed/rerank delegate to SiliconFlow)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.exceptions import EmbeddingError, LLMError
from src.llm_backend.base import ChatResponse, EmbedResponse, LLMBackend, RerankResponse
from src.llm_backend.siliconflow import SiliconFlowBackend


class OllamaBackend(LLMBackend):
    """Ollama backend for chat; delegates embed/rerank to SiliconFlow."""

    def __init__(self, config: dict[str, Any], sf_config: dict[str, Any]) -> None:
        self.host = config.get("host", "http://localhost:11434")
        self.model = config.get("model", "qwen2.5:7b")
        self.timeout = config.get("timeout", 60)
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        # SiliconFlow for embed/rerank fallback
        self._sf = SiliconFlowBackend(sf_config)

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=self.timeout,
            )
        return self._client

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.host,
                timeout=self.timeout,
            )
        return self._sync_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        client = self._get_async_client()
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            logger.debug(f"Ollama chat: model={self.model}")
            return ChatResponse(content=content, model=self.model, usage={})
        except Exception as e:
            raise LLMError(f"Ollama chat failed: {e}") from e

    async def embed(self, texts: list[str]) -> EmbedResponse:
        """Delegate embedding to SiliconFlow."""
        return await self._sf.embed(texts)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> RerankResponse:
        """Delegate reranking to SiliconFlow."""
        return await self._sf.rerank(query, documents, top_k)

    def chat_sync(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        try:
            return asyncio.run(
                self.chat(messages, temperature, top_p, max_tokens)
            )
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.chat(messages, temperature, top_p, max_tokens)
                ).result()

    def embed_sync(self, texts: list[str]) -> EmbedResponse:
        return self._sf.embed_sync(texts)

    def rerank_sync(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> RerankResponse:
        return self._sf.rerank_sync(query, documents, top_k)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
        await self._sf.close()


def create_backend(config: dict[str, Any]) -> LLMBackend:
    """Factory: create the appropriate backend based on config.

    Args:
        config: Full application config dict.

    Returns:
        LLMBackend instance (SiliconFlowBackend or OllamaBackend).
    """
    backend_name = config.get("llm", {}).get("backend", "siliconflow")
    if backend_name == "ollama":
        return OllamaBackend(config["ollama"], config["siliconflow"])
    return SiliconFlowBackend(config["siliconflow"])
