"""Evaluate a user-labelled corpus; never invent relevance labels.

JSON manifest:
{"workspace":"evaluation", "cases":[
  {"id":"t1", "group":"technical_text", "query":"load balancer", "relevant_docs":["diagram.pdf"]},
  {"id":"p1", "group":"photo_image", "image":"queries/cat.jpg", "relevant_docs":["cat.jpg"]}
]}
The corpus must already be indexed. Image paths resolve relative to the manifest.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8101")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = {group: [] for group in ("technical_text", "technical_image", "photo_text", "photo_image")}
    results = []
    headers = {"X-App-Token": os.environ.get("LIGHTGRAPHRAG_APP_API_TOKEN", "")}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=180) as client:
        for case in manifest["cases"]:
            if case["group"] not in groups or not case.get("relevant_docs"):
                raise ValueError("Every case needs a supported group and human-labelled relevant_docs")
            payload = {"workspace": manifest["workspace"], "query": case.get("query", ""), "top_k": 5}
            if case.get("image"):
                path = args.manifest.parent / case["image"]
                response = client.post("/api/images/attachments", params={"workspace": manifest["workspace"]}, files={"file": (path.name, path.read_bytes())})
                response.raise_for_status(); payload["image_id"] = response.json()["image_id"]
            started = time.perf_counter()
            response = client.post("/api/recall/images", json=payload)
            response.raise_for_status()
            names = {source["doc_name"] for hit in response.json()["hits"] for source in hit["sources"]}
            result = {"id": case["id"], "group": case["group"], "hit_at_5": bool(names.intersection(case["relevant_docs"])),
                      "seconds": time.perf_counter()-started, "returned_docs": sorted(names)}
            results.append(result); groups[case["group"]].append(result)
    summary = {}
    for group, rows in groups.items():
        rate = sum(row["hit_at_5"] for row in rows) / len(rows) if rows else None
        summary[group] = {"count": len(rows), "hit_at_5": rate, "target_met": len(rows) >= 10 and rate >= .8,
                          "median_seconds": statistics.median(row["seconds"] for row in rows) if rows else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
