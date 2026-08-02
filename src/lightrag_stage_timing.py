"""Project-level per-stage timing for LightRAG indexing.

This module instruments the four UI stages shown during indexing
(解析 / Chunk向量 / KG抽取 / 图谱/落盘) **without editing the vendored
LightRAG package**. It wraps the separable, side-effect-free callables inside
LightRAG's pipeline:

  - parse        : measured in project code around document loading
                   (:func:`src.api.server._load_doc_for_index`)
  - chunk_vector : ``chunks_vdb`` / ``text_chunks`` upsert and flush.
                   NanoVectorDB defers the actual embedding call until
                   ``index_done_callback``, so both methods are measured.
                   Chunk *splitting* itself is intentionally NOT timed separately,
                   because
                   ``process_single_document`` performs ``self.chunking_func is
                   chunking_by_token_size`` identity checks that a wrapper would
                   silently break. Chunk splitting is CPU-fast anyway; the
                   embedding/vector write dominates this stage.
  - kg           : ``LightRAG._process_extract_entities``
  - merge        : ``merge_nodes_and_edges`` + non-chunk storage flushes

Storage flush callbacks may run outside the user-facing linear stage order.
They are timed, but they do not drive the UI's "current stage"; otherwise a
late/early flush can make the UI jump backward or show all elapsed time under
``图谱/落盘``.

Only one document is processed per ``ainsert`` call in this project
(``ids=[doc_id]``), and the workspace lock serialises ``ainsert`` calls, so
the collector is scoped to a single in-flight ``ainsert`` via a
:class:`contextvars.ContextVar`.
"""
from __future__ import annotations

import contextvars
import inspect
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable

_STAGE_KEYS = ("parse", "vector", "kg", "merge")
_LOGGER = logging.getLogger(__name__)

# ``merge_nodes_and_edges`` is imported into the pipeline module namespace via
# ``from lightrag.operate import merge_nodes_and_edges``; patching the pipeline
# module's global is what the running code actually resolves.
try:
    from lightrag import pipeline as _lr_pipeline
except Exception:  # pragma: no cover - defensive
    _lr_pipeline = None

# Active collector for the current ainsert scope. Wrappers read this so the
# timing is attributed to the right document even across concurrent workspaces.
_ACTIVE_COLLECTOR = contextvars.ContextVar("tdx_stage_collector", default=None)


class StageTimingCollector:
    """Collects per-stage wall-clock seconds for one indexing run."""

    def __init__(self) -> None:
        self.t: dict[str, float] = {k: 0.0 for k in _STAGE_KEYS}
        self.installed = False
        self.on_update: Callable[[dict[str, float], str, str], Any] | None = None

    def reset(self) -> None:
        for k in self.t:
            self.t[k] = 0.0

    def to_stages(self) -> dict[str, float]:
        t = self.t
        return {
            "parse": round(t["parse"], 3),
            "chunk_vector": round(t["vector"], 3),
            "kg": round(t["kg"], 3),
            "merge": round(t["merge"], 3),
        }

    async def notify(self, key: str, event: str) -> None:
        if self.on_update is None:
            return
        try:
            result = self.on_update(self.to_stages(), key, event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            _LOGGER.warning("LightRAG stage timing callback failed", exc_info=True)

    @contextmanager
    def scope(self):
        token = _ACTIVE_COLLECTOR.set(self)
        self.reset()
        try:
            yield
        finally:
            _ACTIVE_COLLECTOR.reset(token)


def _wrap_async(orig, key: str, *, activate_current_stage: bool = True):
    """Wrap an async callable, attributing its wall time to ``key``.

    If no collector is active (e.g. LightRAG is used outside our index flow),
    the wrapper is transparent and adds zero timing overhead.
    """
    if getattr(orig, "_tdx_timing_wrapped", False):
        return orig

    async def wrapper(*args, **kwargs):
        coll = _ACTIVE_COLLECTOR.get()
        if coll is None:
            return await orig(*args, **kwargs)
        start = time.perf_counter()
        if activate_current_stage:
            await coll.notify(key, "start")
        try:
            return await orig(*args, **kwargs)
        finally:
            coll.t[key] += time.perf_counter() - start
            await coll.notify(key, "finish")

    wrapper._tdx_timing_wrapped = True
    return wrapper


def _wrap_store_methods(
    store: Any,
    key: str,
    method_names: tuple[str, ...],
    *,
    activate_current_stage: bool = True,
) -> None:
    for method_name in method_names:
        method = getattr(store, method_name, None)
        if method is not None:
            setattr(
                store,
                method_name,
                _wrap_async(method, key, activate_current_stage=activate_current_stage),
            )


def install_stage_timing(rag: Any) -> StageTimingCollector:
    """Install (idempotent) stage-timing wrappers on a LightRAG instance.

    The collector lives on the instance as ``rag._tdx_stage_timing`` so it
    survives across calls; wrappers resolve the live collector via the
    :data:`_ACTIVE_COLLECTOR` context var during each scoped ``ainsert``.
    """
    collector: StageTimingCollector = getattr(rag, "_tdx_stage_timing", None)
    if collector is None:
        collector = StageTimingCollector()
        rag._tdx_stage_timing = collector

    if collector.installed:
        return collector

    # Chunk vectors/text chunks. NanoVectorDB embeds pending chunk rows during
    # index_done_callback, not during upsert, so measure both.
    for store_name in ("chunks_vdb", "text_chunks"):
        store = getattr(rag, store_name, None)
        if store is not None:
            _wrap_store_methods(store, "vector", ("upsert", "index_done_callback"))

    # KG extraction -> "kg"
    kg_orig = getattr(type(rag), "_process_extract_entities", None)
    if kg_orig is not None and not getattr(kg_orig, "_tdx_timing_wrapped", False):
        type(rag)._process_extract_entities = _wrap_async(kg_orig, "kg")

    # Graph merge -> "merge"
    if _lr_pipeline is not None:
        mne = getattr(_lr_pipeline, "merge_nodes_and_edges", None)
        if mne is not None and not getattr(mne, "_tdx_timing_wrapped", False):
            _lr_pipeline.merge_nodes_and_edges = _wrap_async(mne, "merge")

    # Non-chunk flushes (entities, relationships, graph, caches) -> "merge".
    # Do not wrap LightRAG._insert_done itself, or chunk-vector flushes would be
    # double-counted as merge. These flushes must not switch the visible current
    # stage because LightRAG may call them before/around KG extraction.
    for store_name in (
        "full_docs",
        "doc_status",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "llm_response_cache",
        "entities_vdb",
        "relationships_vdb",
        "chunk_entity_relation_graph",
    ):
        store = getattr(rag, store_name, None)
        if store is not None:
            _wrap_store_methods(
                store,
                "merge",
                ("index_done_callback",),
                activate_current_stage=False,
            )

    collector.installed = True
    return collector
