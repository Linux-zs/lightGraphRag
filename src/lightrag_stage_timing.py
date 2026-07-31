"""Project-level per-stage timing for LightRAG indexing.

This module instruments the four UI stages shown during indexing
(解析 / Chunk向量 / KG抽取 / 图谱merge) **without editing the vendored
LightRAG package**. It wraps the separable, side-effect-free callables inside
LightRAG's pipeline:

  - parse        : measured in project code around document loading
                   (:func:`src.api.server._load_doc_for_index`)
  - chunk_vector : ``chunks_vdb.upsert`` + ``text_chunks.upsert``
                   (chunk embedding + vector-store writes). Chunk *splitting*
                   itself is intentionally NOT timed separately, because
                   ``process_single_document`` performs ``self.chunking_func is
                   chunking_by_token_size`` identity checks that a wrapper would
                   silently break. Chunk splitting is CPU-fast anyway; the
                   embedding/vector write dominates this stage.
  - kg           : ``LightRAG._process_extract_entities``
  - merge        : ``merge_nodes_and_edges`` + ``LightRAG._insert_done``

Only one document is processed per ``ainsert`` call in this project
(``ids=[doc_id]``), and the workspace lock serialises ``ainsert`` calls, so
the collector is scoped to a single in-flight ``ainsert`` via a
:class:`contextvars.ContextVar`.
"""
from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Any

_STAGE_KEYS = ("parse", "vector", "kg", "merge")

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

    @contextmanager
    def scope(self):
        token = _ACTIVE_COLLECTOR.set(self)
        self.reset()
        try:
            yield
        finally:
            _ACTIVE_COLLECTOR.reset(token)


def _wrap_async(orig, key: str):
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
        try:
            return await orig(*args, **kwargs)
        finally:
            coll.t[key] += time.perf_counter() - start

    wrapper._tdx_timing_wrapped = True
    return wrapper


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

    # Vector-store upserts (chunk embedding + writes) -> "vector"
    for store_name in ("chunks_vdb", "text_chunks"):
        store = getattr(rag, store_name, None)
        if store is not None and hasattr(store, "upsert"):
            store.upsert = _wrap_async(store.upsert, "vector")

    # KG extraction -> "kg"
    kg_orig = getattr(type(rag), "_process_extract_entities", None)
    if kg_orig is not None and not getattr(kg_orig, "_tdx_timing_wrapped", False):
        type(rag)._process_extract_entities = _wrap_async(kg_orig, "kg")

    # Graph merge -> "merge"
    if _lr_pipeline is not None:
        mne = getattr(_lr_pipeline, "merge_nodes_and_edges", None)
        if mne is not None and not getattr(mne, "_tdx_timing_wrapped", False):
            _lr_pipeline.merge_nodes_and_edges = _wrap_async(mne, "merge")

    # Final storage flush -> "merge"
    ido = getattr(type(rag), "_insert_done", None)
    if ido is not None and not getattr(ido, "_tdx_timing_wrapped", False):
        type(rag)._insert_done = _wrap_async(ido, "merge")

    collector.installed = True
    return collector
