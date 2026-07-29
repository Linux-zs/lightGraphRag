"""Abstract base class for document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """Parsed document with raw text and metadata."""
    doc_id: str
    file_name: str
    file_path: str
    file_type: str
    raw_text: str
    metadata: dict = field(default_factory=dict)


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    SUPPORTED_EXTENSIONS: list[str] = []

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """Parse a document file into a Document object.

        Args:
            file_path: Path to the document file.

        Returns:
            Document with raw text and metadata.
        """
        ...

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to the document file.

        Returns:
            True if the file extension is supported.
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
