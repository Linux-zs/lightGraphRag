"""Plain text document parser."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.doc_processor.parsers.base_parser import BaseParser, Document


class TxtParser(BaseParser):
    """Parser for plain text files."""

    SUPPORTED_EXTENSIONS = [".txt", ".text"]

    def parse(self, file_path: Path) -> Document:
        """Parse a plain text file into a Document object."""
        try:
            raw_text = file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw_text = file_path.read_text(encoding="gb18030")

        title = file_path.stem
        metadata = {
            "title": title,
            "char_count": len(raw_text),
            "line_count": len(raw_text.splitlines()),
        }

        doc_id = f"txt_{file_path.parent.name}_{file_path.stem}"
        logger.info(f"Parsed text file: {file_path.name}, {len(raw_text)} chars")
        return Document(
            doc_id=doc_id,
            file_name=file_path.name,
            file_path=str(file_path),
            file_type="txt",
            raw_text=raw_text,
            metadata=metadata,
        )
