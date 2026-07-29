"""Custom exceptions for the knowledge-base workbench."""

from __future__ import annotations


class TDXKnowledgeBaseError(Exception):
    """Base exception for all workbench errors."""


class ConfigError(TDXKnowledgeBaseError):
    """Configuration loading or validation error."""


class DocProcessError(TDXKnowledgeBaseError):
    """Document processing error (loading, parsing, chunking)."""


class EmbeddingError(TDXKnowledgeBaseError):
    """Embedding generation error."""


class GraphError(TDXKnowledgeBaseError):
    """Knowledge graph operation error."""


class LLMError(TDXKnowledgeBaseError):
    """LLM backend call error."""
