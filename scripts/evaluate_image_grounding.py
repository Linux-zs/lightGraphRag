"""Opt-in live grounding regression. Manifest cases need explicit expected_images,
including [] for negative cases. Paths are relative to the manifest; no index writes.
Checks image identities, not merely document names (different pictures share a doc).
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config_loader import get_config
from src.model_profiles import get_runtime_model_config
from src.multimodal.provider import MultimodalProvider, prepare_image
from src.multimodal.store import ImageStore
from src.multimodal.grounding import verify_candidates
from src.lightrag_service import sanitize_workspace


async def evaluate(manifest_path, output):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    config = get_config()
    store = ImageStore(Path(config['paths']['data_dir']) / 'multimodal' / sanitize_workspace(manifest['workspace']))
    provider = MultimodalProvider(get_runtime_model_config(config))
    results = []
    for case in manifest['cases']:
        if not isinstance(case.get('expected_images'), list):
            raise ValueError('Explicit image identity labels required; [] means no matching image')
        raw = prepare_image((manifest_path.parent / case['image']).read_bytes())[0] if case.get('image') else None
        started = time.monotonic()
        hits = await store.search(provider, case['query'], raw, 8)
        if 'candidate_ids' in case:
            hits = [h for h in hits if h['image_id'] in case['candidate_ids']]
            if {h['image_id'] for h in hits} != set(case['candidate_ids']):
                raise ValueError('Requested hard-negative candidates were not retrieved')
        accepted, trace, summary = await verify_candidates(store, provider, case['query'], raw, hits)
        actual, expected = {h['image_id'] for h in accepted}, set(case['expected_images'])
        results.append({'id': case['id'], 'passed': actual == expected and not trace['error'],
                        'false_positives': sorted(actual - expected), 'misses': sorted(expected - actual),
                        'seconds': round(time.monotonic()-started, 2), 'trace': trace, 'query_summary': summary})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: results[-1][k] for k in ('id', 'passed', 'false_positives', 'misses', 'seconds')}), flush=True)
    if not all(row['passed'] for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(evaluate(args.manifest, args.output))
