"""Post-extraction policy applied before SDK graph merging."""

import hashlib
import json


def policy_fingerprint(config, guidance):
    payload = {key: config.get(key) for key in (
        "extraction_mode", "entity_types", "relation_types", "allow_other_entity_type")}
    payload["guidance"] = guidance
    payload["policy_version"] = 1
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def validate_extraction_policy(config):
    if config.get("extraction_mode") == "strict" and not any(
        str(value).strip() for value in (config.get("entity_types") or [])
    ):
        raise ValueError("Strict extraction requires a nonempty entity type whitelist")


def enforce_entity_types(results, config, stats):
    validate_extraction_policy(config)
    if config.get("extraction_mode") != "strict":
        return results
    allowed = {str(value).strip().casefold() for value in config.get("entity_types", []) if str(value).strip()}
    if not allowed:
        raise ValueError("Strict extraction requires a nonempty entity type whitelist")
    allowed_relations = {str(value).strip().casefold() for value in config.get("relation_types", []) if str(value).strip()}
    rejected_relation_types = 0
    accepted = []
    filtered_nodes = []
    admitted_names = set()
    rejected_entities = rejected_relations = 0
    for nodes, edges in results:
        kept_nodes = {}
        for name, records in nodes.items():
            kept = [record for record in records
                    if str(record.get("entity_type", "")).strip().casefold() in allowed]
            rejected_entities += len(records) - len(kept)
            if kept:
                kept_nodes[name] = kept
        filtered_nodes.append(kept_nodes)
        admitted_names.update(kept_nodes)
    for kept_nodes, (_, edges) in zip(filtered_nodes, results):
        kept_edges = {}
        for endpoints, records in edges.items():
            if all(endpoint in admitted_names for endpoint in endpoints):
                kept = [record for record in records if not allowed_relations or
                        str(record.get("keywords", "")).strip().casefold() in allowed_relations]
                rejected_relation_types += len(records) - len(kept)
                if kept:
                    kept_edges[endpoints] = kept
            else:
                rejected_relations += len(records)
        accepted.append((kept_nodes, kept_edges))
    stats["policy_rejections"] = {
        "entity_type_not_allowed": rejected_entities,
        "relation_endpoint_not_allowed": rejected_relations,
    }
    if allowed_relations:
        stats["policy_rejections"]["relation_type_not_allowed"] = rejected_relation_types
    stats["entity_count"] = len({name for nodes, _ in accepted for name in nodes})
    stats["relation_count"] = len({tuple(sorted(edge)) for _, edges in accepted for edge in edges})
    return accepted
