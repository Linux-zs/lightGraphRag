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
        doc = None
        try:
            doc = fitz.open(str(file_path))
            pages_text = []
            page_spans = []
            offset = 0
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    if pages_text:
                        offset += 2  # The separator used by raw_text.join below.
                    page_spans.append({"page": page_num + 1, "start": offset, "end": offset + len(text)})
                    pages_text.append(text)
                    offset += len(text)
                else:
                    page_spans.append({"page": page_num + 1, "start": offset, "end": offset, "text_missing": True})

            raw_text = "\n\n".join(pages_text)

            # Extract metadata
            meta = doc.metadata
            metadata = {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "page_count": len(doc),
                "page_spans": page_spans,
                "pages_without_text": [span["page"] for span in page_spans if span.get("text_missing")],
                "ocr_performed": False,
            }

            doc_id = f"pdf_{file_path.parent.name}_{file_path.stem}"
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
        finally:
            if doc is not None:
                doc.close()
