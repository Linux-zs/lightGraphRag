"""Document loader: scan directory and dispatch to appropriate parsers."""

from __future__ import annotations

from hashlib import md5
from pathlib import Path

from loguru import logger

from src.doc_processor.parsers.base_parser import BaseParser, Document
from src.doc_processor.parsers.docx_parser import DocxParser
from src.doc_processor.parsers.pdf_parser import PdfParser
from src.doc_processor.parsers.md_parser import MdParser
from src.doc_processor.parsers.html_parser import HtmlParser
from src.doc_processor.parsers.txt_parser import TxtParser
from src.exceptions import DocProcessError


class DocumentLoader:
    """Scans a directory for documents and dispatches parsing."""

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = [
            DocxParser(),
            PdfParser(),
            MdParser(),
            HtmlParser(),
            TxtParser(),
        ]
        self._ext_to_parser: dict[str, BaseParser] = {}
        for parser in self._parsers:
            for ext in parser.SUPPORTED_EXTENSIONS:
                self._ext_to_parser[ext] = parser

    def _get_parser(self, file_path: Path) -> BaseParser | None:
        """Get the appropriate parser for a file based on its extension."""
        ext = file_path.suffix.lower()
        return self._ext_to_parser.get(ext)

    def scan_directory(self, docs_dir: str | Path) -> list[Path]:
        """Scan a directory for supported document files.

        Args:
            docs_dir: Path to the documents directory.

        Returns:
            List of file paths that can be parsed.
        """
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise DocProcessError(f"Documents directory not found: {docs_dir}")

        supported_files = []
        for file_path in docs_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self._ext_to_parser:
                supported_files.append(file_path)

        logger.info(f"Scanned {docs_dir}: found {len(supported_files)} supported files")
        return sorted(supported_files)

    def load_document(self, file_path: Path) -> Document:
        """Load a single document using the appropriate parser.

        Args:
            file_path: Path to the document file.

        Returns:
            Parsed Document object.

        Raises:
            DocProcessError: If no parser is found or parsing fails.
        """
        parser = self._get_parser(file_path)
        if parser is None:
            raise DocProcessError(f"No parser for file type: {file_path.suffix}")

        try:
            return parser.parse(file_path)
        except Exception as e:
            raise DocProcessError(f"Failed to parse {file_path}: {e}") from e

    def load_all(self, docs_dir: str | Path) -> list[Document]:
        """Load all supported documents from a directory, deduplicating by content hash.

        Args:
            docs_dir: Path to the documents directory.

        Returns:
            List of parsed Document objects (unique by content).
        """
        files = self.scan_directory(docs_dir)
        documents = []
        seen_hashes: set[str] = set()
        skipped = 0

        for file_path in files:
            try:
                # Read file content to compute hash for dedup
                raw_bytes = file_path.read_bytes()
                content_hash = md5(raw_bytes).hexdigest()
                if content_hash in seen_hashes:
                    skipped += 1
                    logger.debug(f"Skipping duplicate (by content): {file_path}")
                    continue
                seen_hashes.add(content_hash)

                doc = self.load_document(file_path)
                documents.append(doc)
            except DocProcessError as e:
                logger.warning(f"Skipping file: {e}")

        if skipped:
            logger.info(f"Deduplicated {skipped} files with identical content")
        logger.info(f"Loaded {len(documents)} documents from {docs_dir}")
        return documents
