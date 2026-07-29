"""Text chunker: RecursiveCharacterTextSplitter preserving metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.doc_processor.parsers.base_parser import Document
from src.exceptions import DocProcessError


@dataclass
class DocumentChunk:
    """A chunk of a document with preserved metadata."""
    chunk_id: str
    doc_id: str
    doc_name: str
    doc_path: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Chunk documents using RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = separators or [
            "\n\n", "\n", "。", "！", "？", "；", " ", ""
        ]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self._separators,
            length_function=len,
        )

    def _make_chunk_id(self, doc_id: str, chunk_index: int, text: str, doc_path: str = "") -> str:
        """Generate a unique chunk ID, incorporating doc_path to avoid collisions
        when files with the same stem exist in different directories."""
        hash_input = f"{doc_path}|{doc_id}|{chunk_index}|{text}"
        hash_part = md5(hash_input.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_c{chunk_index}_{hash_part}"

    def chunk_document(self, document: Document) -> list[DocumentChunk]:
        """Split a document into chunks.

        Args:
            document: The Document to chunk.

        Returns:
            List of DocumentChunk objects.
        """
        if not document.raw_text.strip():
            logger.warning(f"Empty document: {document.file_name}")
            return []

        texts = self._splitter.split_text(document.raw_text)
        chunks = []

        # Calculate approximate start/end char positions
        total_len = len(document.raw_text)
        chunk_len_sum = 0

        for i, text in enumerate(texts):
            start_char = chunk_len_sum
            end_char = min(start_char + len(text), total_len)
            chunk_len_sum = end_char

            chunk_id = self._make_chunk_id(document.doc_id, i, text, document.file_path)
            metadata = {
                "doc_name": document.file_name,
                "doc_path": document.file_path,
                "chunk_index": i,
                "file_type": document.file_type,
            }
            # Merge doc metadata
            metadata.update(document.metadata)

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                doc_name=document.file_name,
                doc_path=document.file_path,
                chunk_index=i,
                text=text,
                start_char=start_char,
                end_char=end_char,
                metadata=metadata,
            ))

        logger.debug(
            f"Chunked {document.file_name}: {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """Chunk multiple documents.

        Args:
            documents: List of Documents to chunk.

        Returns:
            List of all DocumentChunk objects.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
