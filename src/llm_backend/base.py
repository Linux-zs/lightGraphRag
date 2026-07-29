"""Abstract base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChatResponse:
    """Response from a chat completion call."""
    content: str
    model: str
    usage: dict = field(default_factory=dict)


@dataclass
class EmbedResponse:
    """Response from an embedding call."""
    embeddings: list[list[float]]
    model: str
    usage: dict = field(default_factory=dict)


@dataclass
class RerankResponse:
    """Response from a reranking call."""
    results: list[dict]  # Each dict: {"index", "relevance_score", "text"}
    model: str


class LLMBackend(ABC):
    """Abstract base class for LLM backends providing chat, embed, and rerank."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with "role" and "content".
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum tokens to generate.

        Returns:
            ChatResponse with generated content.
        """
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbedResponse:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            EmbedResponse with embedding vectors.
        """
        ...

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> RerankResponse:
        """Rerank documents by relevance to query.

        Args:
            query: The query string.
            documents: List of document texts to rerank.
            top_k: Number of top results to return.

        Returns:
            RerankResponse with ranked results.
        """
        ...

    @abstractmethod
    def chat_sync(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """Synchronous version of chat."""
        ...

    @abstractmethod
    def embed_sync(self, texts: list[str]) -> EmbedResponse:
        """Synchronous version of embed."""
        ...

    @abstractmethod
    def rerank_sync(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> RerankResponse:
        """Synchronous version of rerank."""
        ...
