"""PDF document parser using PyMuPDF (fitz)."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger

from src.doc_processor.parsers.base_parser import Document, BaseParser


class PdfParser(BaseParser):
    """Parser for .pdf files using PyMuPDF."""

    SUPPORTED_EXTENSIONS = [".pdf"]

    def parse(self, file_path: Path) -> Document:
        """Parse a .pdf file into a Document object.

        Args:
            file_path: Path to the .pdf file.

        Returns:
            Document with extracted text and metadata.
        """
        try:
            doc = fitz.open(str(file_path))
            pages_text = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    pages_text.append(text)

            raw_text = "\n\n".join(pages_text)

            # Extract metadata
            meta = doc.metadata
            metadata = {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "page_count": len(doc),
            }

            doc_id = f"pdf_{file_path.parent.name}_{file_path.stem}"
            doc.close()
            logger.info(f"Parsed PDF: {file_path.name}, {len(pages_text)} pages with text")
            return Document(
                doc_id=doc_id,
                file_name=file_path.name,
                file_path=str(file_path),
                file_type="pdf",
                raw_text=raw_text,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise
