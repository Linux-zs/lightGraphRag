"""SiliconFlow LLM backend implementation (chat + embed + rerank)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.exceptions import EmbeddingError, LLMError
from src.llm_backend.base import ChatResponse, EmbedResponse, LLMBackend, RerankResponse


class SiliconFlowBackend(LLMBackend):
    """SiliconFlow API backend supporting chat, embedding, and reranking."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = config.get("base_url", "https://api.siliconflow.cn/v1")
        self.api_key = str(config.get("api_key", "") or "").strip()
        self.chat_model = config.get("chat_model", "Qwen/Qwen2.5-7B-Instruct")
        self.embed_model = config.get("embed_model", "BAAI/bge-large-zh-v1.5")
        self.rerank_model = config.get("rerank_model", "BAAI/bge-reranker-v2-m3")
        self.embed_dim = config.get("embed_dim", 1024)
        self.timeout = config.get("timeout", 30)
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self.last_stream_finish_reason = ""
        self.last_stream_usage: dict[str, Any] = {}
        self.last_stream_done_received = False
        self.last_stream_chunks = 0
        self.last_stream_chars = 0

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=self.timeout,
            )
        return self._client

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers=self._headers(),
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
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> ChatResponse:
        client = self._get_async_client()
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        try:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            logger.debug(f"SiliconFlow chat: model={self.chat_model}, tokens={usage}")
            return ChatResponse(content=content, model=self.chat_model, usage=usage)
        except Exception as e:
            detail = f"{type(e).__name__}: {e!r}"
            if isinstance(e, httpx.HTTPStatusError):
                detail = f"{detail}; response={e.response.text[:500]}"
            raise LLMError(f"SiliconFlow chat failed ({detail})") from e

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completions from SiliconFlow API.

        Yields content tokens one at a time as they arrive.
        """
        client = self._get_async_client()
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        self.last_stream_finish_reason = ""
        self.last_stream_usage = {}
        self.last_stream_done_received = False
        self.last_stream_chunks = 0
        self.last_stream_chars = 0
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # strip "data: "
                    if data_str.strip() == "[DONE]":
                        self.last_stream_done_received = True
                        break
                    try:
                        data = json.loads(data_str)
                        choice = data.get("choices", [{}])[0]
                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            self.last_stream_finish_reason = str(finish_reason)
                        if isinstance(data.get("usage"), dict):
                            self.last_stream_usage = data["usage"]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            self.last_stream_chunks += 1
                            self.last_stream_chars += len(content)
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                logger.info(
                    "chat_stream_complete model={} finish_reason={} done_received={} "
                    "chunks={} chars={} usage={}",
                    self.chat_model,
                    self.last_stream_finish_reason or "missing",
                    self.last_stream_done_received,
                    self.last_stream_chunks,
                    self.last_stream_chars,
                    self.last_stream_usage,
                )
        except Exception as e:
            detail = f"{type(e).__name__}: {e!r}"
            if isinstance(e, httpx.HTTPStatusError):
                detail = f"{detail}; response={e.response.text[:500]}"
            raise LLMError(f"SiliconFlow stream failed ({detail})") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def embed(self, texts: list[str]) -> EmbedResponse:
        client = self._get_async_client()
        payload = {
            "model": self.embed_model,
            "input": texts,
            "encoding_format": "float",
        }
        try:
            resp = await client.post("/embeddings", json=payload)
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            usage = data.get("usage", {})
            logger.debug(f"SiliconFlow embed: {len(texts)} texts, model={self.embed_model}")
            return EmbedResponse(embeddings=embeddings, model=self.embed_model, usage=usage)
        except Exception as e:
            raise EmbeddingError(f"SiliconFlow embedding failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> RerankResponse:
        client = self._get_async_client()
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_k": top_k,
            "return_documents": True,
        }
        try:
            resp = await client.post("/rerank", json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            logger.debug(f"SiliconFlow rerank: query='{query[:30]}...', top_k={top_k}")
            return RerankResponse(results=results, model=self.rerank_model)
        except Exception as e:
            raise LLMError(f"SiliconFlow rerank failed: {e}") from e

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """Run an async coroutine safely from any context.

        Tries asyncio.run() first. If already inside a running event loop
        (FastAPI, Streamlit, etc.), spins up a separate thread to run the
        coroutine without colliding with the existing loop.
        """
        try:
            return asyncio.run(coro)
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()

    def chat_sync(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> ChatResponse:
        return self._run_async(
            self.chat(messages, temperature, top_p, max_tokens,
                      frequency_penalty, presence_penalty)
        )

    def embed_sync(self, texts: list[str]) -> EmbedResponse:
        return self._run_async(self.embed(texts))

    def rerank_sync(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> RerankResponse:
        return self._run_async(
            self.rerank(query, documents, top_k)
        )

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
