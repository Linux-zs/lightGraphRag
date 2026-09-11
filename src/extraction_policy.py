"""Post-extraction policy applied before SDK graph merging."""

import hashlib
import json


_EXCLUSION_OPERATORS = {"exact", "contains", "prefix", "suffix"}
POLICY_FINGERPRINT_VERSION = 3


def _compile_exclusion_rules(values, *, field_name):
    if values is None:
        return ()
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{field_name} must be a list")
    if len(values) > 200:
        raise ValueError(f"{field_name} cannot contain more than 200 rules")
    compiled = []
    for raw_value in values:
        value = str(raw_value).strip()
        if not value:
            continue
        if len(value) > 200:
            raise ValueError(f"Each {field_name} rule must be at most 200 characters")
        operator, separator, operand = value.partition(":")
        normalized_operator = operator.strip().casefold()
        if separator and normalized_operator in _EXCLUSION_OPERATORS:
            normalized_operand = operand.strip().casefold()
            if not normalized_operand:
                raise ValueError(f"{field_name} rule '{value}' has an empty match value")
            compiled.append((normalized_operator, normalized_operand))
        else:
            compiled.append(("exact", value.casefold()))
    return tuple(compiled)


def _matches_exclusion(value, compiled_rules):
    candidate = str(value or "").strip().casefold()
    if not candidate:
        return False
    for operator, operand in compiled_rules:
        if operator == "exact" and candidate == operand:
            return True
        if operator == "contains" and operand in candidate:
            return True
        if operator == "prefix" and candidate.startswith(operand):
            return True
        if operator == "suffix" and candidate.endswith(operand):
            return True
    return False


def policy_fingerprint(config, guidance):
    payload = {key: config.get(key) for key in (
        "extraction_mode", "entity_types", "relation_types", "allow_other_entity_type",
        "entity_exclusion_rules", "relation_exclusion_rules")}
    payload["guidance"] = guidance
    # Bump when guidance, KG-input sanitation, or post-extraction enforcement
    # changes so persisted graphs are explicitly marked for rebuild.
    payload["policy_version"] = POLICY_FINGERPRINT_VERSION
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def validate_extraction_policy(config):
    if config.get("extraction_mode") == "strict" and not any(
        str(value).strip() for value in (config.get("entity_types") or [])
    ):
        raise ValueError("Strict extraction requires a nonempty entity type whitelist")
    _compile_exclusion_rules(
        config.get("entity_exclusion_rules"), field_name="entity_exclusion_rules"
    )
    _compile_exclusion_rules(
        config.get("relation_exclusion_rules"), field_name="relation_exclusion_rules"
    )


def enforce_entity_types(results, config, stats):
    validate_extraction_policy(config)
    strict = config.get("extraction_mode") == "strict"
    entity_exclusions = _compile_exclusion_rules(
        config.get("entity_exclusion_rules"), field_name="entity_exclusion_rules"
    )
    relation_exclusions = _compile_exclusion_rules(
        config.get("relation_exclusion_rules"), field_name="relation_exclusion_rules"
    )
    if not strict and not entity_exclusions and not relation_exclusions:
        return results
    allowed = {
        str(value).strip().casefold()
        for value in config.get("entity_types", [])
        if str(value).strip()
    } if strict else set()
    if strict and not allowed:
        raise ValueError("Strict extraction requires a nonempty entity type whitelist")
    allowed_relations = {
        str(value).strip().casefold()
        for value in config.get("relation_types", [])
        if str(value).strip()
    } if strict else set()
    rejected_relation_types = 0
    excluded_relation_types = 0
    accepted = []
    filtered_nodes = []
    admitted_names = set()
    rejected_entities = rejected_entity_names = rejected_relations = 0
    for nodes, edges in results:
        kept_nodes = {}
        for name, records in nodes.items():
            if _matches_exclusion(name, entity_exclusions):
                rejected_entity_names += len(records)
                continue
            kept = records if not strict else [
                record for record in records
                if str(record.get("entity_type", "")).strip().casefold() in allowed
            ]
            if strict:
                rejected_entities += len(records) - len(kept)
            if kept:
                kept_nodes[name] = kept
        filtered_nodes.append(kept_nodes)
        admitted_names.update(kept_nodes)
    for kept_nodes, (_, edges) in zip(filtered_nodes, results):
        kept_edges = {}
        for endpoints, records in edges.items():
            endpoint_rejected = (
                (strict and not all(endpoint in admitted_names for endpoint in endpoints))
                or any(_matches_exclusion(endpoint, entity_exclusions) for endpoint in endpoints)
            )
            if endpoint_rejected:
                rejected_relations += len(records)
                continue
            kept = []
            for record in records:
                relation_type = str(record.get("keywords", "")).strip()
                if allowed_relations and relation_type.casefold() not in allowed_relations:
                    rejected_relation_types += 1
                    continue
                if _matches_exclusion(relation_type, relation_exclusions):
                    excluded_relation_types += 1
                    continue
                kept.append(record)
            if kept:
                kept_edges[endpoints] = kept
        accepted.append((kept_nodes, kept_edges))
    rejections = {}
    if strict:
        rejections["entity_type_not_allowed"] = rejected_entities
    if strict or entity_exclusions:
        rejections["relation_endpoint_not_allowed"] = rejected_relations
    if entity_exclusions:
        rejections["entity_name_excluded"] = rejected_entity_names
    if allowed_relations:
        rejections["relation_type_not_allowed"] = rejected_relation_types
    if relation_exclusions:
        rejections["relation_type_excluded"] = excluded_relation_types
    stats["policy_rejections"] = rejections
    stats["entity_count"] = len({name for nodes, _ in accepted for name in nodes})
    stats["relation_count"] = len({tuple(sorted(edge)) for _, edges in accepted for edge in edges})
    return accepted
