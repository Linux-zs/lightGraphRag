import asyncio

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc

from src.lightrag_service import LightRAGService


def test_cancel_drains_real_sdk_children_and_provider_queue(tmp_path, monkeypatch):
    async def run():
        started, stopped = asyncio.Event(), asyncio.Event()
        calls = []

        async def model(*args, **kwargs):
            calls.append(1)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def embed(texts, **kwargs):
            return np.ones((len(texts), 3), dtype=np.float32)

        service = LightRAGService({
            "paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "rag")},
            "lightrag": {"entity_extract_max_gleaning": 0, "llm_model_max_async": 1},
        }, workspace="cancel_probe")
        monkeypatch.setattr(service, "_make_embedding_func", lambda: EmbeddingFunc(embedding_dim=3, func=embed))
        monkeypatch.setattr(service, "_make_llm_func_for", lambda purpose: model)
        monkeypatch.setattr(service, "_runtime_models", lambda: {"chat": {"model": "fake"}, "kg": {"model": "fake"}, "embedding": {"model": "fake", "embed_dim": 3, "base_url": "offline"}, "rerank": {"enabled": False}})
        monkeypatch.setattr(service, "_make_rerank_func", lambda: None)
        rag = await service.get_rag()
        unrelated = asyncio.create_task(asyncio.Event().wait())
        chunks = {f"chunk-{i}": {"content": f"Atlas reads Cedar documentation number {i}.", "tokens": 10, "chunk_order_index": i, "full_doc_id": "probe", "file_path": "probe.txt"} for i in range(4)}
        try:
            with service._temporary_index_llm_and_kg_filter(rag, skip_kg=False):
                task = asyncio.create_task(rag._process_extract_entities(chunks, {"history_messages": []}, asyncio.Lock()))
                await asyncio.wait_for(started.wait(), 3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
                assert stopped.is_set()
                assert len(calls) == 1
                assert not unrelated.done()
                assert not any("_process_with_semaphore" in repr(t.get_coro()) for t in asyncio.all_tasks() if not t.done())
        finally:
            unrelated.cancel()
            await asyncio.gather(unrelated, return_exceptions=True)
            await service.finalize()

    asyncio.run(run())
