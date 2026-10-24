"""A nearest neighbour is a candidate, never automatically an answer citation."""
import asyncio
import json
import httpx
from .provider import RetryableModelError
from .timeouts import timeout


async def verify_candidates(store, provider, question, query_image, hits):
    candidates = [hit for hit in hits if hit.get('sources')][:3]
    trace = {'mode': 'image' if query_image is not None else 'text',
             'policy_version': 'image-grounding-v2',
             'verifier': {key: getattr(provider, 'models', {}).get('vision', {}).get(key) for key in ('base_url', 'model')},
             'candidates': [{'image_id': h['image_id'], 'score': h.get('score'),
                             'similarities': h.get('similarities', {})} for h in hits],
             'verified': [], 'selected': [], 'error': ''}
    if not candidates:
        return [], trace, ''
    try:
        payload = [(h['image_id'], store.asset_path(h['image_id']).read_bytes()) for h in candidates]
        allowed = 'same_image' if query_image is not None else 'relevant'
        result = {'matches': [], 'query_summary': ''}
        trace['verified'] = result['matches']
        async with timeout(75):
            for candidate in payload:
                for attempt in range(2):
                    try:
                        checked = await provider.verify_evidence(question, query_image, [candidate])
                        break
                    except (httpx.TransportError, RetryableModelError, json.JSONDecodeError):
                        if attempt:
                            raise
                        await asyncio.sleep(1)
                result['matches'].extend(checked['matches'])
                result['query_summary'] = checked.get('query_summary', '')
                if query_image is not None and any(m['verdict'] == allowed for m in checked['matches']):
                    break
        decisions = {m['id']: m for m in result['matches']}
        trace['verified'] = result['matches']
        current = {(s['image_id'], s['doc_id']) for s in store.occurrences()}
        accepted = []
        for hit in candidates:
            decision = decisions.get(hit['image_id'], {})
            if decision.get('verdict') != allowed or not decision.get('reason', '').strip():
                continue
            sources = [s for s in hit['sources'] if (hit['image_id'], s['doc_id']) in current]
            if sources:
                accepted.append({**hit, 'sources': sources, 'verification': decision})
        trace['selected'] = [h['image_id'] for h in accepted]
        return accepted, trace, result.get('query_summary', '')
    except Exception as exc:
        trace['error'] = type(exc).__name__
        trace['error_detail'] = str(exc)[:400]
        return [], trace, ''
