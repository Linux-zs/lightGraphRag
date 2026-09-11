import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.api import server
from src.doc_processor.parsers.base_parser import Document
from src.lightrag_service import LightRAGService


def _document() -> Document:
    return Document(
        doc_id="doc",
        file_name="sample.txt",
        file_path="sample.txt",
        file_type="txt",
        raw_text="sample content",
    )


def test_preview_graph_extraction_samples_without_merging_and_restores_guidance():
    service = object.__new__(LightRAGService)
    service.workspace = "preview_workspace"
    service._last_kg_filter_stats = {}
    service.load_graph_governance = lambda: {"extraction_mode": "assist"}
    service.graph_extraction_guidance = lambda config: "preview guidance"
    service.graph_governance_summary = lambda config=None: {"extraction_mode": "assist"}

    async def preview_chunks(*_args, **_kwargs):
        return [
            {"chunk_order_index": index, "content": f"chunk {index}"}
            for index in range(5)
        ]

    async def extract(chunks, *_args):
        assert [chunk["chunk_order_index"] for chunk in chunks.values()] == [1, 3]
        assert rag.addon_params["entity_types_guidance"] == "preview guidance"
        service._last_kg_filter_stats = {
            "policy_rejections": {"entity_name_excluded": 1}
        }
        return [
            (
                {
                    "Service": [
                        {"entity_type": "服务", "description": "short"},
                        {"entity_type": "服务", "description": "longer description"},
                    ]
                },
                {
                    ("Service", "Database"): [
                        {"keywords": "依赖", "description": "reads data"}
                    ]
                },
            )
        ]

    rag = SimpleNamespace(
        addon_params={"language": "Chinese"},
        _process_extract_entities=extract,
    )
    service.preview_document_chunks = preview_chunks
    service.get_rag = AsyncMock(return_value=rag)
    service._temporary_index_llm_and_kg_filter = lambda *_args, **_kwargs: nullcontext()

    result = asyncio.run(
        service.preview_graph_extraction(_document(), sample_chunk_count=2)
    )

    assert [item["index"] for item in result["sampled_chunks"]] == [1, 3]
    assert result["entity_count"] == 1
    assert result["entities"][0] == {
        "name": "Service",
        "entity_type": "服务",
        "description": "longer description",
        "record_count": 2,
    }
    assert result["relation_count"] == 1
    assert result["filter_stats"]["policy_rejections"]["entity_name_excluded"] == 1
    assert rag.addon_params == {"language": "Chinese"}


def test_preview_graph_extraction_endpoint_uses_workspace_lock(monkeypatch):
    result = {
        "file_name": "sample.txt",
        "total_chunk_count": 1,
        "sampled_chunks": [],
        "entities": [],
        "relations": [],
        "entity_count": 0,
        "relation_count": 0,
        "entities_truncated": False,
        "relations_truncated": False,
        "filter_stats": {},
        "elapsed_seconds": 0.1,
        "graph_rule": {},
    }
    preview = AsyncMock(return_value=result)
    monkeypatch.setattr(server, "_index_tasks", {})
    monkeypatch.setattr(server, "_load_doc_for_index", lambda *_: _document())
    monkeypatch.setattr(
        server,
        "get_lightrag_service",
        lambda _: SimpleNamespace(preview_graph_extraction=preview),
    )

    response = asyncio.run(
        server.preview_graph_extraction(
            server.GraphExtractionPreviewRequest(
                workspace="preview_workspace",
                file_name="sample.txt",
                sample_chunk_count=1,
            )
        )
    )

    assert response == result
    assert preview.await_count == 1
    assert preview.call_args.kwargs["sample_chunk_count"] == 1
