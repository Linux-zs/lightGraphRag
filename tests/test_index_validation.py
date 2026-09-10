import json
import base64

import numpy as np

import pytest

from src.index_validation import validate_embedding_metadata, validate_index, validate_vectors


@pytest.mark.parametrize('invalid', [None, [], {'workspace': 'other'},
    {'workspace': 'kb', 'model': 'embed', 'embed_dim': 3},
    {'workspace': 'kb', 'model': '', 'embed_dim': 2}])
def test_embedding_signature_gate_rejects_missing_or_inconsistent(candidate, invalid):
    path, _ = candidate
    metadata = path / 'embedding.json'
    if invalid is not None:
        metadata.write_text(json.dumps(invalid), encoding='utf-8')
    with pytest.raises(RuntimeError, match='embedding metadata'):
        validate_embedding_metadata(path, metadata, 'kb')


def test_embedding_signature_gate_accepts_matching_vectors(candidate):
    path, _ = candidate
    metadata = path / 'embedding.json'
    metadata.write_text(json.dumps({'workspace': 'kb', 'model': 'embed', 'embed_dim': 2}), encoding='utf-8')
    validate_embedding_metadata(path, metadata, 'kb')


def test_committed_publication_verified_from_disk(candidate, monkeypatch):
    from types import SimpleNamespace
    from src.api import server

    path, manifest = candidate
    service = SimpleNamespace(workspace_dir=path, manifest_path=path / 'manifest.json',
        embedding_meta_path=path / 'embedding.json', _load_manifest=lambda: manifest)
    service.manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    service.embedding_meta_path.write_text(json.dumps({
        'workspace': 'kb', 'model': 'embed', 'embed_dim': 2}), encoding='utf-8')
    base = path / 'rebuild_shadow' / 'task'
    base.mkdir(parents=True)
    artifacts = []
    for active, source, backup in (
        (path, base / 'lightrag' / 'kb', base / 'previous' / 'workspace'),
        (service.manifest_path, base / 'manifest.json', base / 'previous' / 'manifest.json'),
        (service.embedding_meta_path, base / 'embedding_meta' / 'kb.json', base / 'previous' / 'embedding_meta.json'),
    ):
        artifacts.append(dict(active=str(active.resolve()), candidate=str(source.resolve()),
            backup=str(backup.resolve()), had_active=True, had_candidate=True))
    journal = dict(version=1, task_id='task', workspace='kb', state='committed', artifacts=artifacts)
    journal_path = base / 'publication.json'
    journal_path.write_text(json.dumps(journal), encoding='utf-8')
    monkeypatch.setattr(server, 'get_config', lambda: {'paths': {'data_dir': str(path)}})
    monkeypatch.setattr(server, 'get_lightrag_service', lambda _: service)
    task = dict(task_id='task', workspace='kb', doc_names=['source.txt'])
    assert server._verify_completed_publication(task)
    journal['artifacts'][0]['active'] = str(path.parent)
    journal_path.write_text(json.dumps(journal), encoding='utf-8')
    with pytest.raises(RuntimeError, match='path mismatch'):
        server._verify_completed_publication(task)


def test_publication_rejects_missing_signature_before_reset(candidate, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from src.api import server

    path, manifest = candidate
    reset = AsyncMock()
    shadow = SimpleNamespace(
        workspace_dir=path, embedding_meta_path=path / 'missing.json',
        finalize=AsyncMock(), _load_manifest=lambda: manifest,
    )
    monkeypatch.setattr(server, 'get_lightrag_service', lambda _: object())
    monkeypatch.setattr(server, 'reset_lightrag_service_async', reset)
    with pytest.raises(RuntimeError, match='embedding metadata'):
        asyncio.run(server._commit_shadow_rebuild(
            {'workspace': 'kb', 'doc_names': ['source.txt']}, shadow))
    reset.assert_not_awaited()
    assert (path / 'vdb_chunks.json').exists()


def vector_payload(values=(1.0, 2.0), ids=("chunk",), dimension=2):
    return {"data": [{"__id__": key} for key in ids], "embedding_dim": dimension,
            "matrix": base64.b64encode(np.array(values, dtype=np.float32).tobytes()).decode()}


@pytest.fixture
def candidate(tmp_path):
    payloads = {
        "kv_store_full_docs.json": {"doc": {"content": "source"}},
        "kv_store_doc_status.json": {"doc": {"status": "processed", "chunks_list": ["chunk"]}},
        "kv_store_text_chunks.json": {"chunk": {"full_doc_id": "doc", "content": "source"}},
        "vdb_chunks.json": vector_payload(),
    }
    for name, data in payloads.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    manifest = {"documents": {"doc": {
        "doc_name": "source.txt", "active_index_doc_id": "doc", "indexed": True,
        "chunks_list": ["chunk"], "chunk_count": 1, "kg_status": "no_entities",
    }}}
    return tmp_path, manifest


def test_restore_accepts_pending_upload_without_weakening_publication(candidate):
    path, manifest = candidate
    manifest['documents']['pending'] = {'doc_name': 'pending.txt', 'indexed': False}
    names = ['source.txt', 'pending.txt']
    assert validate_index(path, manifest, names, allow_unindexed=True)['documents'] == 1
    with pytest.raises(RuntimeError, match='missing document'):
        validate_index(path, manifest, names)


@pytest.mark.parametrize('reference', [
    {'active_index_doc_id': 'old'}, {'chunks_list': ['chunk']}, {'chunk_count': 1},
])
def test_restore_does_not_skip_inconsistent_active_document(candidate, reference):
    path, manifest = candidate
    manifest['documents']['pending'] = {'doc_name': 'pending.txt', 'indexed': False, **reference}
    with pytest.raises(RuntimeError, match='still references an active index'):
        validate_index(path, manifest, ['source.txt', 'pending.txt'], allow_unindexed=True)


def test_valid_text_only_index_is_allowed(candidate):
    path, manifest = candidate
    assert validate_index(path, manifest, ["source.txt"]) == {
        "documents": 1, "chunks": 1, "entities": 0,
    }


@pytest.mark.parametrize("filename", [
    "kv_store_full_docs.json", "kv_store_doc_status.json",
    "kv_store_text_chunks.json", "vdb_chunks.json",
])
def test_missing_store_is_rejected(candidate, filename):
    path, manifest = candidate
    (path / filename).unlink()
    with pytest.raises(RuntimeError, match="cannot read"):
        validate_index(path, manifest, ["source.txt"])


def test_missing_vector_is_rejected(candidate):
    path, manifest = candidate
    (path / "vdb_chunks.json").write_text(json.dumps(vector_payload((), ())), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing chunk or vector"):
        validate_index(path, manifest, ["source.txt"])


def test_empty_graph_cannot_contradict_extraction_result(candidate):
    path, manifest = candidate
    manifest["documents"]["doc"]["kg_entity_count"] = 1
    with pytest.raises(RuntimeError, match="persisted graph is empty"):
        validate_index(path, manifest, ["source.txt"])


def test_partial_extraction_is_not_publishable(candidate):
    path, manifest = candidate
    manifest["documents"]["doc"]["kg_status"] = "partial"
    with pytest.raises(RuntimeError, match="incomplete graph"):
        validate_index(path, manifest, ["source.txt"])


def test_corpus_cannot_shrink_silently(candidate):
    path, manifest = candidate
    with pytest.raises(RuntimeError, match="document set"):
        validate_index(path, manifest, ["source.txt", "missing.txt"])


@pytest.mark.parametrize("payload", [
    vector_payload((1.0,)),
    vector_payload((float("nan"), 1.0)),
    vector_payload((float("inf"), 1.0)),
    vector_payload((0.0, 0.0)),
    vector_payload((1.0, 2.0, 3.0, 4.0), ("chunk", "chunk")),
    vector_payload(dimension=0),
    {**vector_payload(), "matrix": "!invalid"},
])
def test_corrupt_vector_matrix_is_rejected(payload):
    with pytest.raises(RuntimeError, match="invalid vectors"):
        validate_vectors(payload, "vdb_chunks.json")


def test_empty_manifest_does_not_bypass_vector_validation(tmp_path):
    (tmp_path / "vdb_entities.json").write_text('{"data": []}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid vectors"):
        validate_index(tmp_path, {"documents": {}}, [])


def test_empty_manifest_reports_manual_graph_entities(tmp_path):
    import networkx as nx
    graph = nx.Graph()
    graph.add_node("manual")
    nx.write_graphml(graph, tmp_path / "graph_chunk_entity_relation.graphml")
    assert validate_index(tmp_path, {"documents": {}}, [])["entities"] == 1


def test_duplicate_chunk_references_are_not_a_valid_count(candidate):
    path, manifest = candidate
    manifest["documents"]["doc"].update(chunks_list=["chunk", "chunk"], chunk_count=2)
    with pytest.raises(RuntimeError, match="chunk count"):
        validate_index(path, manifest, ["source.txt"])
