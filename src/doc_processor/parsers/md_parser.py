"""Markdown document parser."""

from __future__ import annotations

from pathlib import Path

import markdown
from loguru import logger

from src.doc_processor.parsers.base_parser import BaseParser, Document


class MdParser(BaseParser):
    """Parser for .md files."""

    SUPPORTED_EXTENSIONS = [".md", ".markdown"]

    def parse(self, file_path: Path) -> Document:
        """Parse a .md file into a Document object.

        Args:
            file_path: Path to the .md file.

        Returns:
            Document with extracted text and metadata.
        """
        try:
            raw_md = file_path.read_text(encoding="utf-8")
            # Convert markdown to plain text (strip HTML tags from rendered output)
            html = markdown.markdown(raw_md, extensions=["extra", "toc"])
            # Simple HTML-to-text: remove tags
            import re
            plain_text = re.sub(r"<[^>]+>", "", html)
            # Also keep original markdown as raw_text for better chunk quality
            raw_text = raw_md

            # Extract title from first heading if present
            title_match = re.match(r"^#\s+(.+)", raw_md)
            title = title_match.group(1) if title_match else file_path.stem

            metadata = {
                "title": title,
                "char_count": len(raw_md),
                "has_code_blocks": "```" in raw_md,
            }

            doc_id = f"md_{file_path.parent.name}_{file_path.stem}"
            logger.info(f"Parsed Markdown: {file_path.name}, {len(raw_md)} chars")
            return Document(
                doc_id=doc_id,
                file_name=file_path.name,
                file_path=str(file_path),
                file_type="md",
                raw_text=raw_text,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to parse Markdown {file_path}: {e}")
            raise
