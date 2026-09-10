"""Read-only publication gate for the supported local index format."""

import json
import base64
from pathlib import Path

import networkx as nx
import numpy as np


def validate_vectors(payload: dict, name: str) -> set[str]:
    """Validate NanoVectorDB metadata and the actual serialized float32 matrix."""
    try:
        dimension = payload["embedding_dim"]
        records = payload["data"]
        if type(dimension) is not int or dimension <= 0 or not isinstance(records, list):
            raise ValueError("invalid dimension or records")
        ids = [record["__id__"] for record in records]
        if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("invalid or duplicate vector IDs")
        matrix = np.frombuffer(base64.b64decode(payload["matrix"], validate=True), dtype=np.float32)
        if matrix.size != len(records) * dimension:
            raise ValueError("matrix size does not match record count and dimension")
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite vector values")
        if records and (np.linalg.norm(matrix.reshape(-1, dimension), axis=1) == 0).any():
            raise ValueError("zero vectors cannot support similarity retrieval")
        return set(ids)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Index validation: invalid vectors in {name}: {exc}") from exc


def validate_embedding_metadata(workspace_dir: Path, metadata_path: Path, workspace: str) -> None:
    """Never publish vectors under a missing or mismatched model signature."""
    stores = []
    try:
        for name in ("vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json"):
            path = workspace_dir / name
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                validate_vectors(payload, name)
                if payload["data"]:
                    stores.append(payload)
        if not stores and not metadata_path.exists():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        if metadata.get("workspace") != workspace:
            raise ValueError("metadata workspace differs from candidate")
        if not isinstance(metadata.get("model"), str) or not metadata["model"].strip():
            raise ValueError("missing embedding model")
        dimension = metadata.get("embed_dim")
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("invalid embedding dimension")
        if any(store["embedding_dim"] != dimension for store in stores):
            raise ValueError("vector dimension differs from model signature")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(f"Index validation: invalid embedding metadata: {exc}") from exc


def validate_index(workspace_dir: Path, manifest: dict, expected_names: list[str], *, allow_unindexed: bool = False) -> dict:
    def read(name):
        path = workspace_dir / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Index validation: cannot read {name}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Index validation: invalid object in {name}")
        return value

    documents = manifest.get("documents", {})
    names = [item.get("doc_name") for item in documents.values()]
    if len(names) != len(set(names)) or set(names) != set(expected_names):
        raise RuntimeError("Index validation: rebuilt document set differs from requested corpus")
    if allow_unindexed:
        # Restoring an existing generation is different from publishing a full
        # rebuild: uploaded sources need not already belong to its index.
        # Never hide a damaged active index merely because indexed is false.
        pending = [item for item in documents.values() if not item.get("indexed")]
        if any(item.get("active_index_doc_id") or item.get("chunks_list") or item.get("chunk_count")
               for item in pending):
            raise RuntimeError("Index validation: unindexed document still references an active index")
        documents = {key: item for key, item in documents.items() if item.get("indexed")}
    if not documents:
        # A source-empty workspace can still contain manually imported graph
        # knowledge. It must pass the same persisted-format checks.
        for name in ("vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json"):
            if (workspace_dir / name).exists():
                validate_vectors(read(name), name)
        graph_path = workspace_dir / "graph_chunk_entity_relation.graphml"
        graph = nx.read_graphml(graph_path) if graph_path.exists() else nx.Graph()
        chunk_count = len(read("kv_store_text_chunks.json")) if (workspace_dir / "kv_store_text_chunks.json").exists() else 0
        return {"documents": 0, "chunks": chunk_count, "entities": graph.number_of_nodes()}
    docs = read("kv_store_full_docs.json")
    statuses = read("kv_store_doc_status.json")
    chunks = read("kv_store_text_chunks.json")
    vectors = read("vdb_chunks.json")
    vector_ids = validate_vectors(vectors, "vdb_chunks.json")
    for name in ("vdb_entities.json", "vdb_relationships.json"):
        if (workspace_dir / name).exists():
            validate_vectors(read(name), name)
    required_chunks = set()
    for item in documents.values():
        doc_id = item.get("active_index_doc_id")
        ids = item.get("chunks_list") or []
        if not item.get("indexed") or not doc_id or doc_id not in docs:
            raise RuntimeError(f"Index validation: missing document {item.get('doc_name')}")
        status = statuses.get(doc_id, {})
        if status.get("status") != "processed" or set(status.get("chunks_list") or []) != set(ids):
            raise RuntimeError(f"Index validation: inconsistent status for {doc_id}")
        if not ids or len(ids) != len(set(ids)) or len(ids) != item.get("chunk_count"):
            raise RuntimeError(f"Index validation: invalid chunk count for {doc_id}")
        if item.get("kg_status") in {"partial", "failed"}:
            raise RuntimeError(f"Index validation: incomplete graph extraction for {doc_id}")
        for chunk_id in ids:
            if chunk_id not in chunks or chunk_id not in vector_ids:
                raise RuntimeError(f"Index validation: missing chunk or vector {chunk_id}")
            if chunks[chunk_id].get("full_doc_id") != doc_id:
                raise RuntimeError(f"Index validation: wrong document owner for {chunk_id}")
        required_chunks.update(ids)
    graph_path = workspace_dir / "graph_chunk_entity_relation.graphml"
    graph = nx.read_graphml(graph_path) if graph_path.exists() else nx.Graph()
    expects_graph = any(
        item.get("kg_entity_count", 0) or item.get("kg_relation_count", 0)
        for item in documents.values()
    )
    if expects_graph and graph.number_of_nodes() == 0:
        raise RuntimeError("Index validation: extraction reported entities but persisted graph is empty")
    return {"documents": len(documents), "chunks": len(required_chunks), "entities": graph.number_of_nodes()}
