"""Exercise the installed SDK scheduler without external model requests."""

import asyncio
import json
from collections import Counter

import pytest
from lightrag.operate import extract_entities

from src.lightrag_service import LightRAGService
from src.lightrag_stage_timing import StageTimingCollector, _wrap_async


class MemoryCache:
    global_config = {"enable_llm_cache_for_entity_extract": True}

    def __init__(self):
        self.rows = {}

    async def get_by_id(self, key):
        return self.rows.get(key)

    async def upsert(self, rows):
        self.rows.update(rows)


def make_service(tmp_path, **settings):
    return LightRAGService(
        {
            "paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "rag")},
            "lightrag": {"workspace": "test", "llm_model_max_async": 2, **settings},
            "siliconflow": {},
        },
        workspace="test",
    )


@pytest.mark.parametrize("cache_enabled", [True, False])
@pytest.mark.parametrize("failure_kind", ["timeout", "invalid_response"])
def test_sdk_failure_does_not_restart_or_cancel_healthy_chunks(
    tmp_path, cache_enabled, failure_kind,
):
    service = make_service(tmp_path, llm_model_max_async=3)
    calls = Counter()
    cancelled = Counter()

    class InvalidResponseError(Exception):
        pass

    async def run():
        slow_started = asyncio.Event()
        fast_done = asyncio.Event()
        failed = asyncio.Event()

        async def model(prompt, **kwargs):
            name = next(name for name in ("FAST_OK", "FAIL_BLOCK", "SLOW_OK") if name in prompt)
            calls[name] += 1
            try:
                if name == "FAST_OK":
                    fast_done.set()
                elif name == "FAIL_BLOCK":
                    await slow_started.wait()
                    await fast_done.wait()
                    failed.set()
                    if failure_kind == "timeout":
                        raise TimeoutError("Worker execution timeout after 90s")
                    raise InvalidResponseError("Received empty content from OpenAI API")
                else:
                    slow_started.set()
                    await failed.wait()
                    await asyncio.sleep(0.03)
                return json.dumps({
                    "entities": [
                        {"name": name, "type": "service", "description": "Source service"},
                        {"name": "Storage", "type": "database", "description": "Shared storage"},
                    ],
                    "relationships": [{
                        "source": name, "target": "Storage",
                        "description": "Writes records", "keywords": "writes",
                    }],
                })
            except asyncio.CancelledError:
                cancelled[name] += 1
                raise

        config = {
            "role_llm_funcs": {"extract": model},
            "entity_extract_max_gleaning": 0,
            "entity_extract_max_records": 48,
            "entity_extract_max_entities": 24,
            "llm_model_max_async": 3,
            "entity_extraction_use_json": True,
            "addon_params": {"language": "Chinese"},
        }
        cache = MemoryCache() if cache_enabled else None
        chunks = {
            f"doc_probe-chunk-{index:03d}": {"content": name}
            for index, name in enumerate(("FAST_OK", "FAIL_BLOCK", "SLOW_OK"))
        }

        async def extract(current, *args, **kwargs):
            return await extract_entities(current, config, llm_response_cache=cache)

        stats = {"kept": 3, "skipped": 0, "reasons": {}}
        result = await service._extract_entities_with_recovery(extract, chunks, (), {}, stats)
        assert len(result) == 2
        assert stats["kept"] == 2
        assert stats["skipped"] == 1
        assert [set(nodes) for nodes, _edges in result] == [
            {"FAST_OK", "Storage"}, {"SLOW_OK", "Storage"},
        ]
        source_ids = {
            edge["source_id"]
            for _nodes, edges in result for records in edges.values() for edge in records
        }
        assert source_ids == {"doc_probe-chunk-000", "doc_probe-chunk-002"}
        assert stats["chunk_diagnostics"]["doc_probe-chunk-001"]["status"] == "skipped"
        for diagnostic in stats["chunk_diagnostics"].values():
            assert diagnostic["queue_seconds"] >= 0
            assert diagnostic["processing_seconds"] >= 0

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    assert calls == {"FAST_OK": 1, "FAIL_BLOCK": 1, "SLOW_OK": 1}
    assert not cancelled


def test_chunk_concurrency_is_bounded_and_results_are_preserved(tmp_path):
    service = make_service(tmp_path)

    async def run():
        active = peak = 0
        seen = Counter()

        async def extract(current, *args, **kwargs):
            nonlocal active, peak
            assert len(current) == 1
            name = next(iter(current))
            seen[name] += 1
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.005)
                return [name]
            finally:
                active -= 1

        chunks = {f"chunk-{index}": {} for index in range(9)}
        results = await service._extract_entities_with_recovery(
            extract, chunks, (), {}, {"kept": 9, "skipped": 0},
        )
        assert results == list(chunks)
        assert seen == Counter(chunks.keys())
        assert peak == 2

    asyncio.run(run())


def test_document_cancellation_drains_extraction_workers(tmp_path):
    service = make_service(tmp_path)

    async def run():
        active = 0
        started = asyncio.Event()

        async def extract(current, *args, **kwargs):
            nonlocal active
            active += 1
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                active -= 1

        task = asyncio.create_task(service._extract_entities_with_recovery(
            extract, {"a": {}, "b": {}, "c": {}}, (), {}, {"kept": 3, "skipped": 0},
        ))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active == 0

    asyncio.run(run())


def test_skipping_limit_still_fails_document_without_replaying_chunks(tmp_path):
    service = make_service(tmp_path, kg_max_timed_out_chunks=1)
    seen = Counter()

    async def extract(current, *args, **kwargs):
        name = next(iter(current))
        seen[name] += 1
        raise TimeoutError(f"{name}: Worker execution timeout after 90s")

    with pytest.raises(TimeoutError):
        asyncio.run(service._extract_entities_with_recovery(
            extract, {"a": {}, "b": {}, "c": {}}, (), {}, {"kept": 3, "skipped": 0},
        ))
    assert max(seen.values()) == 1


def test_disabled_skipping_propagates_error_but_drains_healthy_inflight_chunk(tmp_path):
    service = make_service(tmp_path, kg_skip_timed_out_chunks=False)

    async def run():
        healthy_started = asyncio.Event()
        failed = asyncio.Event()
        completed = []

        async def extract(current, *args, **kwargs):
            name = next(iter(current))
            if name == "bad":
                await healthy_started.wait()
                failed.set()
                raise TimeoutError("provider timeout")
            healthy_started.set()
            await failed.wait()
            await asyncio.sleep(0.01)
            completed.append(name)
            return [name]

        with pytest.raises(TimeoutError):
            await service._extract_entities_with_recovery(
                extract, {"bad": {}, "healthy": {}, "queued": {}}, (), {},
                {"kept": 3, "skipped": 0},
            )
        assert completed == ["healthy"]

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_all_skipped_chunks_return_empty_results_and_partial_status(tmp_path):
    service = make_service(tmp_path)

    async def extract(current, *args, **kwargs):
        raise TimeoutError("provider timeout")

    stats = {"kept": 2, "skipped": 1, "reasons": {"blank": 1}}
    result = asyncio.run(service._extract_entities_with_recovery(
        extract, {"a": {}, "b": {}}, (), {}, stats,
    ))
    assert result == []
    assert stats["kept"] == 0
    assert stats["skipped"] == 3
    assert service._kg_status_for_success(skip_kg=False) == "partial"


def test_parallel_chunk_timings_emit_one_document_stage(tmp_path):
    service = make_service(tmp_path)

    async def run():
        collector = StageTimingCollector()
        events = []
        collector.on_update = lambda timings, stage, event: events.append((stage, event))

        async def extract(current, *args, **kwargs):
            await asyncio.sleep(0.03)
            return list(current)

        # The SDK class hook is already instrumented by install_stage_timing.
        wrapped = _wrap_async(extract, "kg")
        loop = asyncio.get_running_loop()
        start = loop.time()
        with collector.scope():
            await service._extract_entities_with_recovery(
                wrapped, {"a": {}, "b": {}}, (), {}, {"kept": 2, "skipped": 0},
            )
        elapsed = loop.time() - start
        assert events == [("kg", "start"), ("kg", "finish")]
        assert collector.t["kg"] <= elapsed + 0.002
        assert collector.t["kg"] >= 0.02

    asyncio.run(run())
