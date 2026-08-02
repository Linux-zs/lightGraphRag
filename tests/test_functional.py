"""Functional smoke tests for the LightRAG-based workbench."""

from __future__ import annotations

import asyncio
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
    assert config["chunking"]["chunk_size"] == 1024
    assert config["paths"]["lightrag_dir"]
    assert "不要把原文逐条搬运成答案" in config["answer_generation"]["system_prompt"]
    assert config["lightrag"]["kg_skip_low_value_chunks"] is True
    assert config["lightrag"]["kg_skip_timed_out_chunks"] is True
    assert config["lightrag"]["entity_extract_max_entities"] == 24
    assert config["lightrag"]["entity_extract_max_records"] == 48
    assert config["siliconflow"]["kg_max_tokens"] == 1536


def test_workspace_creation_defaults_to_enhanced_graph_extraction():
    from src.api import server

    req = server.WorkspaceCreateRequest(workspace="new_kb")

    assert req.rule_template_id == "general_knowledge"
    assert req.extraction_mode == "enhanced"
    assert req.allow_other_entity_type is False


def test_runtime_model_config_includes_kg_binding():
    from src.config_loader import load_config
    from src.model_profiles import get_runtime_model_config

    runtime = get_runtime_model_config(load_config())

    assert runtime["kg"]["model"]
    assert runtime["kg"]["base_url"]
    assert runtime["kg"]["max_tokens"] == 1536
    assert runtime["kg"]["timeout"] == 90


def test_index_request_config_records_index_mode():
    from src.api import server

    req = server.BatchIndexRequest(
        workspace="abc",
        doc_names=["a.txt"],
        separators=["\n"],
        chunk_size=1024,
        chunk_overlap=100,
        index_mode="fast",
    )

    assert server._index_request_config(req)["index_mode"] == "fast"


def test_kg_chunk_filter_skips_noise_without_dropping_short_business_text(tmp_path: Path):
    from src.lightrag_service import LightRAGService

    service = LightRAGService(
        {
            "paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "lightrag")},
            "lightrag": {"kg_skip_low_value_chunks": True, "workspace": "test"},
            "siliconflow": {},
        },
        workspace="test",
    )
    chunks = {
        "noise": {"content": "|---|---|\n|---|---|\n|---|---|\n|---|---|"},
        "fact": {"content": "A药厂的核心原料药供应商是B化工公司"},
    }

    filtered, stats = service._filter_kg_chunks(chunks)

    assert "noise" not in filtered
    assert "fact" in filtered
    assert stats["skipped"] == 1


def test_kg_timeout_chunk_detection_only_matches_timeout_errors(tmp_path: Path):
    from src.lightrag_service import LightRAGService

    service = LightRAGService(
        {
            "paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "lightrag")},
            "lightrag": {"workspace": "test"},
            "siliconflow": {},
        },
        workspace="test",
    )
    chunks = {
        "doc_abc-chunk-001": {"content": "正常内容"},
        "doc_abc-chunk-002": {"content": "慢内容"},
    }

    assert (
        service._timed_out_chunk_id(
            chunks,
            TimeoutError(
                "C[2/2]: doc_abc-chunk-002: extract LLM func: "
                "Worker execution timeout after 180s"
            ),
        )
        == "doc_abc-chunk-002"
    )
    assert service._timed_out_chunk_id(
        chunks,
        RuntimeError("doc_abc-chunk-002 returned invalid JSON"),
    ) == ""


def test_kg_timeout_recovery_skips_only_the_failed_chunk(tmp_path: Path):
    from src.lightrag_service import LightRAGService

    service = LightRAGService(
        {
            "paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "lightrag")},
            "lightrag": {
                "workspace": "test",
                "kg_skip_timed_out_chunks": True,
                "kg_max_timed_out_chunks": 2,
            },
            "siliconflow": {},
        },
        workspace="test",
    )
    chunks = {
        "doc_abc-chunk-001": {"content": "正常内容"},
        "doc_abc-chunk-002": {"content": "慢内容"},
    }
    calls: list[list[str]] = []

    async def extract(current_chunks, *args, **kwargs):
        calls.append(list(current_chunks))
        if "doc_abc-chunk-002" in current_chunks:
            raise TimeoutError(
                "doc_abc-chunk-002: extract LLM func: "
                "Worker execution timeout after 180s"
            )
        return ["ok"]

    stats = {"kept": 2, "skipped": 0, "reasons": {}}
    result = asyncio.run(
        service._extract_entities_with_timeout_recovery(
            extract,
            chunks,
            (),
            {},
            stats,
        )
    )

    assert result == ["ok"]
    assert calls == [
        ["doc_abc-chunk-001", "doc_abc-chunk-002"],
        ["doc_abc-chunk-001"],
    ]
    assert stats["timed_out"] == ["doc_abc-chunk-002"]
    assert service._kg_status_for_success(skip_kg=False) == "partial"


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
