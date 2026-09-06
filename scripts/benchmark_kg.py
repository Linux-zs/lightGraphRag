"""Opt-in live KG benchmark using configured providers and synthetic sources only."""

import argparse
import asyncio
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightrag.operate import extract_entities
from loguru import logger

from src.config_loader import get_config
from src.lightrag_service import LightRAGService
from src.model_profiles import get_profile_with_key, get_runtime_model_config


SAMPLES = [
    (
        "Atlas 服务部署在 node-a 上，从 Cedar 数据库读取库存，向 Birch 队列写入审计事件。"
        "Beacon 服务消费 Birch 队列的事件，通过 Delta 网关发送告警。"
        "Cedar 不可用时，Atlas 无法校验新订单。恢复数据库后，运维人员重启 Atlas 的连接池。",
        [("Atlas", "node-a"), ("Atlas", "Cedar"), ("Atlas", "Birch"), ("Beacon", "Birch"), ("Beacon", "Delta")],
    ),
    (
        "North 药厂从 East 公司采购核心原料。North 向 West 医院供应药品。"
        "East 公司于 2024 年停产三个月，资料未说明其目前是否复产。"
        "不能由这条历史记录确定 West 医院当前已经断供。",
        [("North", "East"), ("North", "West")],
    ),
    (
        "# 部署记录\nOrion 服务读取 config.yaml，连接 Maple 数据库。\n"
        "配置示例：DB_HOST=maple.internal；PORT=5432。\n"
        "Orion 通过 Quartz 队列向 Vega 服务传递任务。Vega 将执行结果写入 Maple。\n"
        "故障排查：先确认 Maple 连接正常，再检查 Quartz 积压。日志编号 20240901 不是服务名。",
        [("Orion", "Maple"), ("Orion", "Quartz"), ("Vega", "Quartz"), ("Vega", "Maple")],
    ),
]


async def benchmark(args):
    config = get_config()
    baseline = get_runtime_model_config(config)["kg"]
    for candidate in args.candidate:
        profile_id, model = candidate.split("=", 1)
        profile = get_profile_with_key(profile_id, config)
        runtime = {**baseline, "model": model, "base_url": profile["api_base"],
                   "api_key": profile["api_key"], "timeout": args.timeout}
        service = LightRAGService.__new__(LightRAGService)
        service.config = deepcopy(config)
        service.config.setdefault("lightrag", {})["kg_llm_max_attempts"] = 1
        service.workspace = "synthetic-benchmark"
        service._runtime_models = lambda: {"kg": runtime}
        for index, (source, expected) in enumerate(SAMPLES[:args.samples]):
            model_call = service._make_kg_llm_func()
            call_seconds = []

            async def extract_model(prompt, **kwargs):
                start = time.perf_counter()
                try:
                    return await model_call(prompt, **{**service._llm_kwargs("kg"), **kwargs})
                finally:
                    call_seconds.append(time.perf_counter() - start)

            graph_config = {
                "role_llm_funcs": {"extract": extract_model},
                "entity_extract_max_gleaning": 0, "entity_extract_max_records": 48,
                "entity_extract_max_entities": 24, "llm_model_max_async": 1,
                "entity_extraction_use_json": True, "addon_params": {"language": "Chinese"},
            }
            start = time.perf_counter()
            result = {"profile": profile_id, "model": model, "sample": index + 1}
            try:
                extracted = await asyncio.wait_for(extract_entities(
                    {f"synthetic-{index}": {"content": source}}, graph_config,
                ), args.timeout + 5)
                nodes = {name for group, _ in extracted for name in group}
                edges = {tuple(pair) for _, group in extracted for pair in group}
                hits = []
                for left, right in expected:
                    hits.append(any(
                        (left.lower() in a.lower() and right.lower() in b.lower())
                        or (right.lower() in a.lower() and left.lower() in b.lower())
                        for a, b in edges
                    ))
                result.update(status="succeeded", entities=sorted(nodes), edges=sorted(edges),
                              expected_edges=len(expected), matched_edges=sum(hits))
            except Exception as exc:
                result.update(status="failed", error=service._kg_failure_detail(exc))
            result.update(seconds=round(time.perf_counter() - start, 3),
                          api_seconds=[round(value, 3) for value in call_seconds])
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize billable calls to configured providers")
    parser.add_argument("--candidate", action="append", required=True, help="profile_id=model_id")
    parser.add_argument("--samples", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not args.live or args.timeout <= 0 or any("=" not in item for item in args.candidate):
        parser.error("--live, positive --timeout and profile_id=model_id candidates are required")
    logger.remove()
    asyncio.run(benchmark(args))
