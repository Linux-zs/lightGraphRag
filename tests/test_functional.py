"""Functional smoke tests for the LightRAG-based workbench."""

from __future__ import annotations

from pathlib import Path


def test_config_loading_for_lightrag_stack():
    from src.config_loader import load_config

    config = load_config()

    assert "siliconflow" in config
    assert "chunking" in config
    assert "paths" in config
    assert "lightrag" in config
    assert "answer_generation" in config
    assert config["siliconflow"]["embed_dim"] == 1024
    assert config["chunking"]["chunk_size"] == 512
    assert config["paths"]["lightrag_dir"]
    assert "不要把原文逐条搬运成答案" in config["answer_generation"]["system_prompt"]


def test_document_loader_supports_current_upload_types(tmp_path: Path):
    from src.doc_processor.loader import DocumentLoader

    loader = DocumentLoader()
    supported = set(loader._ext_to_parser.keys())

    assert ".txt" in supported
    assert ".md" in supported
    assert ".pdf" in supported
    assert ".docx" in supported

    txt = tmp_path / "sample.txt"
    txt.write_text("新闻资讯数据测试", encoding="utf-8")
    doc = loader.load_document(txt)

    assert doc.file_type == "txt"
    assert "新闻资讯" in doc.raw_text


def test_chunker_preserves_document_metadata():
    from src.doc_processor.chunker import TextChunker
    from src.doc_processor.parsers.base_parser import Document

    doc = Document(
        doc_id="doc_test",
        file_name="sample.txt",
        file_path="/tmp/sample.txt",
        file_type="txt",
        raw_text="这是通达信行情系统的测试文档。\n\n" * 80,
        metadata={"title": "测试文档"},
    )

    chunks = TextChunker(chunk_size=300, chunk_overlap=30).chunk_document(doc)

    assert len(chunks) > 1
    assert chunks[0].doc_id == "doc_test"
    assert chunks[0].metadata["doc_name"] == "sample.txt"
    assert chunks[0].metadata["title"] == "测试文档"


def test_lightrag_service_workspace_sanitizer_and_doc_id():
    from src.doc_processor.parsers.base_parser import Document
    from src.lightrag_service import sanitize_workspace, stable_doc_id

    assert sanitize_workspace("abc_123") == "abc_123"

    doc = Document(
        doc_id="local",
        file_name="sample.txt",
        file_path="data/uploads/sample.txt",
        file_type="txt",
        raw_text="hello",
    )
    assert stable_doc_id(doc).startswith("doc_")
