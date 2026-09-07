import asyncio

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc

from src.doc_processor.parsers.base_parser import Document
from src.lightrag_service import LightRAGService


@pytest.mark.parametrize("model_name", ["ChunkPreviewRequest", "IndexRequest", "BatchIndexRequest", "RebuildIndexRequest"])
def test_chunk_request_rejects_overlap_not_smaller_than_size(model_name):
    from pydantic import ValidationError
    from src.api import server

    with pytest.raises(ValidationError, match="chunk_overlap must be smaller"):
        getattr(server, model_name)(file_name="probe.txt", doc_names=["probe.txt"], chunk_size=128, chunk_overlap=128)


@pytest.mark.parametrize("mode", ["fast", "complete"])
def test_actual_sdk_index_matches_preview_and_per_document_settings(tmp_path, monkeypatch, mode):
    service = LightRAGService({
        "paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "rag")},
        "chunking": {"chunk_size": 1024, "chunk_overlap": 100},
        "lightrag": {"entity_extract_max_gleaning": 0, "llm_model_max_async": 2},
    }, workspace=f"parity_{mode}")

    async def embed(texts, **kwargs):
        return np.ones((len(texts), 3), dtype=np.float32)

    async def model(*args, **kwargs):
        return '{"entities":[],"relationships":[]}'

    monkeypatch.setattr(service, "_make_embedding_func", lambda: EmbeddingFunc(embedding_dim=3, func=embed))
    monkeypatch.setattr(service, "_make_llm_func_for", lambda purpose: model)
    monkeypatch.setattr(service, "_runtime_models", lambda: {"chat": {"model": "fake"}, "kg": {"model": "fake"}, "embedding": {"model": "fake", "embed_dim": 3, "base_url": "offline"}, "rerank": {"enabled": False}})
    monkeypatch.setattr(service, "_make_rerank_func", lambda: None)
    monkeypatch.setattr(service, "assert_embedding_compatible", lambda: None)
    text = ("Atlas reads Cedar. CUSTOM_SPLIT " * 100) + ("很长的中文段落" * 50)

    async def run():
        try:
            sizes = []
            for index, (size, overlap) in enumerate([(256, 20), (128, 0)]):
                doc = Document(doc_id=f"source-{index}", file_name=f"probe-{index}.txt", file_path=f"probe-{index}.txt", file_type="txt", raw_text=text + f" sample-{index}")
                expected = await service.preview_document_chunks(doc, chunk_size=size, chunk_overlap=overlap, separators=["CUSTOM_SPLIT", " "])
                item = await service.index_document(doc, chunk_size=size, chunk_overlap=overlap, separators=["CUSTOM_SPLIT", " "], index_mode=mode)
                actual = await service.get_document_chunks(item["doc_id"])
                assert item["kg_status"] == ("skipped" if mode == "fast" else "no_entities")
                assert item["kg_entity_count"] == item["kg_relation_count"] == 0
                assert [c["text"] for c in actual] == [c["content"] for c in expected]
                assert all(c["tokens"] <= size for c in expected)
                sizes.append(len(actual))
            assert sizes[1] > sizes[0]
            old_version = item["active_index_doc_id"]
            # Reindex unchanged content under the same public filename.
            replacement = await service.index_document(doc, chunk_size=256, chunk_overlap=20, index_mode=mode)
            assert replacement["active_index_doc_id"] != old_version
            assert replacement["chunks_list"]
            assert await service.get_doc_status(replacement["active_index_doc_id"]) is not None
            assert await service.get_doc_status(old_version) is None
        finally:
            await service.finalize()

    asyncio.run(run())
