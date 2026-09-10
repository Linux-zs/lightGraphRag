"""Separate local SDK runtime caches from the application's logical workspace.

The supported file stores capture their paths during construction, but bind
shared memory/locks during initialize(). Rebinding between those phases keeps
the on-disk format compatible while preventing a rebuilt or reopened service
from attaching to stale JSON namespaces. The application enforces one live
writer per physical directory; these scopes do not provide multi-writer safety.
"""

from uuid import uuid4

from lightrag import LightRAG


STORAGE_NAMES = (
    "llm_response_cache", "text_chunks", "full_docs", "full_entities",
    "full_relations", "entity_chunks", "relation_chunks",
    "chunk_entity_relation_graph", "entities_vdb", "relationships_vdb",
    "chunks_vdb", "doc_status",
)
SUPPORTED_STORES = {
    "JsonKVStorage", "JsonDocStatusStorage", "NetworkXStorage", "NanoVectorDBStorage",
}


def isolate_file_storage_runtime(rag: LightRAG) -> str:
    """Call exactly once, after construction and before initialize_storages.

    Fail closed on a new storage backend: database-backed implementations may
    interpret workspace dynamically as a persistent database partition.
    """
    stores = [getattr(rag, name) for name in STORAGE_NAMES]
    for storage in stores:
        if type(storage).__name__ not in SUPPORTED_STORES:
            raise RuntimeError(f"Unverified SDK storage scope: {type(storage).__name__}")
        if getattr(storage, "_storage_lock", None) is not None:
            raise RuntimeError("Cannot isolate an already initialized SDK storage")
    scope = f"runtime_{uuid4().hex}"
    for storage in stores:
        storage.workspace = scope
    # Pipeline reservations and graph merge locks must use the same scope.
    rag.workspace = scope
    return scope
