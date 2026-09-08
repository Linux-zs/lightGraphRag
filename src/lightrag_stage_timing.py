"""Project-level per-stage timing for LightRAG indexing.

This module instruments the four UI stages shown during indexing
(解析 / Chunk向量 / KG抽取 / 图谱/落盘) **without editing the vendored
LightRAG package**. It wraps the separable, side-effect-free callables inside
LightRAG's pipeline:

  - parse        : measured in project code around document loading
                   (:func:`src.api.server._load_doc_for_index`)
  - chunk_vector : concurrent ``chunks_vdb`` / ``text_chunks`` upserts.
                   Chunk *splitting* itself is intentionally NOT timed separately,
                   because
                   ``process_single_document`` performs ``self.chunking_func is
                   chunking_by_token_size`` identity checks that a wrapper would
                   silently break. Backends that defer embedding or persistence
                   until the final flush complete that work in ``merge``.
  - kg           : ``LightRAG._process_extract_entities``
  - merge        : ``merge_nodes_and_edges`` + the final ``_insert_done`` flush

The SDK runs some storage operations concurrently and calls cache flushes from
inside KG extraction.  Timing every low-level callback independently therefore
double-counts wall time and can attribute KG work to ``图谱/落盘``.  The wrappers
below time mutually exclusive high-level stages instead.  Concurrent calls in
one stage are measured as the union of their active intervals, not as the sum
of each coroutine's duration.

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
from functools import wraps
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
_ACTIVE_COLLECTOR = contextvars.ContextVar("lightgraphrag_stage_collector", default=None)
_ACTIVE_STAGES = contextvars.ContextVar("lightgraphrag_active_stages", default=frozenset())


class StageTimingCollector:
    """Collects per-stage wall-clock seconds for one indexing run."""

    def __init__(self) -> None:
        self.t: dict[str, float] = {k: 0.0 for k in _STAGE_KEYS}
        self._active_counts: dict[str, int] = {k: 0 for k in _STAGE_KEYS}
        self._active_started_at: dict[str, float | None] = {
            k: None for k in _STAGE_KEYS
        }
        self.installed = False
        self.on_update: Callable[[dict[str, float], str, str], Any] | None = None

    def reset(self) -> None:
        for k in self.t:
            self.t[k] = 0.0
            self._active_counts[k] = 0
            self._active_started_at[k] = None

    def begin(self, key: str) -> bool:
        """Start a stage interval and return whether it is the first active call."""
        first = self._active_counts[key] == 0
        if first:
            self._active_started_at[key] = time.perf_counter()
        self._active_counts[key] += 1
        return first

    def finish(self, key: str) -> bool:
        """Finish a stage interval and return whether the interval is now closed."""
        count = self._active_counts[key]
        if count <= 0:  # pragma: no cover - defensive invariant guard
            return False
        count -= 1
        self._active_counts[key] = count
        if count:
            return False
        started_at = self._active_started_at[key]
        if started_at is not None:
            self.t[key] += time.perf_counter() - started_at
        self._active_started_at[key] = None
        return True

    def _elapsed(self, key: str) -> float:
        elapsed = self.t[key]
        started_at = self._active_started_at[key]
        if started_at is not None:
            elapsed += time.perf_counter() - started_at
        return elapsed

    def to_stages(self) -> dict[str, float]:
        return {
            "parse": round(self._elapsed("parse"), 3),
            "chunk_vector": round(self._elapsed("vector"), 3),
            "kg": round(self._elapsed("kg"), 3),
            "merge": round(self._elapsed("merge"), 3),
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


def time_index_stage(key: str):
    """Time an adapter stage once, suppressing nested SDK timing hooks."""
    return lambda func: _wrap_async(func, key)


def _wrap_async(orig, key: str, *, activate_current_stage: bool = True):
    """Wrap an async callable, attributing its wall time to ``key``.

    If no collector is active (e.g. LightRAG is used outside our index flow),
    the wrapper is transparent and adds zero timing overhead.
    """
    if getattr(orig, "_lightgraphrag_timing_wrapped", False):
        return orig

    @wraps(orig)
    async def wrapper(*args, **kwargs):
        coll = _ACTIVE_COLLECTOR.get()
        # High-level stages are mutually exclusive.  In particular, cache or
        # vector callbacks invoked by KG/merge must stay part of that outer
        # stage instead of being counted a second time under another label.
        if coll is None or _ACTIVE_STAGES.get():
            return await orig(*args, **kwargs)
        token = _ACTIVE_STAGES.set(_ACTIVE_STAGES.get() | {key})
        first = coll.begin(key)
        try:
            if activate_current_stage and first:
                await coll.notify(key, "start")
            return await orig(*args, **kwargs)
        finally:
            closed = coll.finish(key)
            _ACTIVE_STAGES.reset(token)
            if closed:
                await coll.notify(key, "finish")

    wrapper._lightgraphrag_timing_wrapped = True
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

    The collector lives on the instance as ``rag._lightgraphrag_stage_timing`` so it
    survives across calls; wrappers resolve the live collector via the
    :data:`_ACTIVE_COLLECTOR` context var during each scoped ``ainsert``.
    """
    collector: StageTimingCollector = getattr(rag, "_lightgraphrag_stage_timing", None)
    if collector is None:
        collector = StageTimingCollector()
        rag._lightgraphrag_stage_timing = collector

    if collector.installed:
        return collector

    # Chunk/text upserts run concurrently.  The collector measures their union
    # so two one-second calls report about one second, not two.
    for store_name in ("chunks_vdb", "text_chunks"):
        store = getattr(rag, store_name, None)
        if store is not None:
            _wrap_store_methods(store, "vector", ("upsert",))

    # KG extraction -> "kg"
    kg_orig = getattr(type(rag), "_process_extract_entities", None)
    if kg_orig is not None and not getattr(
        kg_orig, "_lightgraphrag_timing_wrapped", False
    ):
        type(rag)._process_extract_entities = _wrap_async(kg_orig, "kg")

    # Graph merge -> "merge"
    if _lr_pipeline is not None:
        mne = getattr(_lr_pipeline, "merge_nodes_and_edges", None)
        if mne is not None and not getattr(
            mne, "_lightgraphrag_timing_wrapped", False
        ):
            _lr_pipeline.merge_nodes_and_edges = _wrap_async(mne, "merge")

    # Final storage flush -> "merge".  Wrapping the high-level method measures
    # the concurrent gather once and prevents its child callbacks from being
    # added again.  It also excludes cache flushes that happen during KG calls.
    insert_done = getattr(type(rag), "_insert_done", None)
    if insert_done is not None and not getattr(
        insert_done, "_lightgraphrag_timing_wrapped", False
    ):
        type(rag)._insert_done = _wrap_async(insert_done, "merge")

    collector.installed = True
    return collector
