import asyncio
import os
import subprocess
import sys
import json

import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

from src.sdk_storage_scope import isolate_file_storage_runtime
from src.index_validation import validate_vectors


async def embed(texts):
    return np.ones((len(texts), 8), dtype=np.float32)


async def no_llm(*args, **kwargs):
    raise AssertionError("Storage contract tests must not call a model")


def make_rag(root):
    rag = LightRAG(
        working_dir=str(root), workspace="kb",
        llm_model_func=no_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=512, func=embed),
    )
    isolate_file_storage_runtime(rag)
    return rag


def test_real_sdk_shadow_and_reopen_do_not_reuse_old_memory(tmp_path):
    async def run():
        active = make_rag(tmp_path / "active")
        shadow = make_rag(tmp_path / "shadow")
        await active.initialize_storages()
        await shadow.initialize_storages()
        try:
            await active.full_docs.upsert({"old": {"content": "old document"}})
            await active.full_docs.index_done_callback()
            await active.doc_status.upsert({"old": {"status": "processed"}})
            assert await shadow.doc_status.filter_keys({"old"}) == {"old"}
            assert await shadow.full_docs.get_by_id("old") is None
            assert active.full_docs._data is not shadow.full_docs._data
            await shadow.full_docs.upsert({"new": {"content": "new document"}})
            await shadow.full_docs.index_done_callback()
            await shadow.chunks_vdb.upsert({"new-chunk": {"content": "new document", "full_doc_id": "new"}})
            await shadow.chunks_vdb.index_done_callback()
            persisted = json.loads((tmp_path / "shadow" / "kb" / "vdb_chunks.json").read_text(encoding="utf-8"))
            assert validate_vectors(persisted, "vdb_chunks.json") == {"new-chunk"}
            await shadow.chunk_entity_relation_graph.upsert_node("new entity", {"entity_type": "concept"})
            await shadow.chunk_entity_relation_graph.index_done_callback()
        finally:
            await active.finalize_storages()
            await shadow.finalize_storages()
        os.replace(tmp_path / "active" / "kb", tmp_path / "backup")
        os.replace(tmp_path / "shadow" / "kb", tmp_path / "active" / "kb")
        reopened = make_rag(tmp_path / "active")
        await reopened.initialize_storages()
        try:
            assert await reopened.full_docs.get_by_id("old") is None
            assert (await reopened.full_docs.get_by_id("new"))["content"] == "new document"
            assert await reopened.chunk_entity_relation_graph.has_node("new entity")
        finally:
            await reopened.finalize_storages()
    asyncio.run(run())
    # Fresh interpreter: verify persistence, not merely the reopened cache.
    result = subprocess.run(
        [sys.executable, "-c", """
import asyncio, sys
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.kg.networkx_impl import NetworkXStorage
async def main():
    initialize_share_data()
    cfg = {'working_dir': sys.argv[1]}
    docs = JsonKVStorage(namespace='full_docs', workspace='kb', global_config=cfg, embedding_func=None)
    graph = NetworkXStorage(namespace='chunk_entity_relation', workspace='kb', global_config=cfg, embedding_func=None)
    await docs.initialize()
    await graph.initialize()
    assert await docs.get_by_id('old') is None
    assert (await docs.get_by_id('new'))['content'] == 'new document'
    assert await graph.has_node('new entity')
asyncio.run(main())
""", str(tmp_path / "active")], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
