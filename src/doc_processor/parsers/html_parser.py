"""HTML document parser using BeautifulSoup."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from loguru import logger

from src.doc_processor.parsers.base_parser import BaseParser, Document


class HtmlParser(BaseParser):
    """Parser for .html and .htm files using BeautifulSoup."""

    SUPPORTED_EXTENSIONS = [".html", ".htm"]

    def parse(self, file_path: Path) -> Document:
        """Parse an .html file into a Document object.

        Args:
            file_path: Path to the .html file.

        Returns:
            Document with extracted text and metadata.
        """
        try:
            raw_html = file_path.read_text(encoding="utf-8")
            soup = BeautifulSoup(raw_html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            raw_text = soup.get_text(separator="\n", strip=True)

            # Extract metadata from HTML meta tags
            title_tag = soup.find("title")
            title = title_tag.get_text() if title_tag else file_path.stem

            meta_tags = soup.find_all("meta")
            html_meta = {}
            for tag in meta_tags:
                name = tag.get("name", tag.get("property", ""))
                content = tag.get("content", "")
                if name and content:
                    html_meta[name] = content

            metadata = {
                "title": title,
                "char_count": len(raw_text),
                "html_meta": html_meta,
            }

            doc_id = f"html_{file_path.parent.name}_{file_path.stem}"
            logger.info(f"Parsed HTML: {file_path.name}, {len(raw_text)} chars extracted")
            return Document(
                doc_id=doc_id,
                file_name=file_path.name,
                file_path=str(file_path),
                file_type="html",
                raw_text=raw_text,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to parse HTML {file_path}: {e}")
            raise
