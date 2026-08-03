"""Custom exceptions for the knowledge-base workbench."""

from __future__ import annotations


class KnowledgeBaseError(Exception):
    """Base exception for all workbench errors."""


class ConfigError(KnowledgeBaseError):
    """Configuration loading or validation error."""


class DocProcessError(KnowledgeBaseError):
    """Document processing error (loading, parsing, chunking)."""


class EmbeddingError(KnowledgeBaseError):
    """Embedding generation error."""


class GraphError(KnowledgeBaseError):
    """Knowledge graph operation error."""


class LLMError(KnowledgeBaseError):
    """LLM backend call error."""


class ManifestCorruptedError(KnowledgeBaseError):
    """The workspace manifest and its backup are both unreadable."""


class RetrievalError(KnowledgeBaseError):
    """A retrieval dependency failed before an answer could be generated."""


class RuntimeLockError(KnowledgeBaseError):
    """Another workbench process already owns the runtime data directory."""
