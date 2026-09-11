import pytest

from src.extraction_policy import enforce_entity_types


def test_policy_fingerprint_ignores_metadata_but_tracks_effective_guidance():
    from src.extraction_policy import policy_fingerprint
    policy = {'entity_types': ['Service'], 'extraction_mode': 'strict'}
    original = policy_fingerprint(policy, 'reference content')
    assert original == policy_fingerprint({**policy, 'updated_at': 'later'}, 'reference content')
    assert original != policy_fingerprint(policy, 'changed reference content')
    assert original != policy_fingerprint({**policy, 'entity_types': ['Location']}, 'reference content')
    assert original != policy_fingerprint(
        {**policy, 'entity_exclusion_rules': ['contains:/tmp/']},
        'reference content',
    )


def test_rule_summary_uses_supplied_snapshot_instead_of_current_settings():
    from src.lightrag_service import LightRAGService
    service = object.__new__(LightRAGService)
    service.load_graph_governance = lambda: {'rule_template_name': 'new rule', 'entity_types': ['A', 'B']}
    snapshot = {'rule_template_name': 'old rule', 'entity_types': ['A'], 'extraction_mode': 'strict'}
    summary = service.graph_governance_summary(snapshot)
    assert summary['rule_template_name'] == 'old rule'
    assert summary['entity_type_count'] == 1
    assert summary['extraction_mode'] == 'strict'
    assert service.graph_governance_summary()['rule_template_name'] == 'new rule'


def test_extraction_uses_copied_policy_even_when_configuration_changes():
    import asyncio
    from types import SimpleNamespace
    from src.lightrag_service import LightRAGService

    service = object.__new__(LightRAGService)
    policy = {'extraction_mode': 'strict', 'entity_types': ['Service']}
    service.load_graph_governance = lambda: policy
    service._runtime_models = lambda: {'kg': {}}
    service._make_kg_llm_func = lambda: None
    service._llm_kwargs = lambda _: {}
    service._filter_kg_chunks = lambda chunks: (chunks, {})
    async def extract(*_):
        policy['entity_types'][:] = ['Location']
        return [({'A': [{'entity_type': 'Service'}], 'B': [{'entity_type': 'Location'}]}, {})]
    rag = SimpleNamespace(_process_extract_entities=extract)
    with service._temporary_index_llm_and_kg_filter(rag, skip_kg=False, policy_config=policy):
        result = asyncio.run(rag._process_extract_entities(['chunk']))
    assert set(result[0][0]) == {'A'}


def test_strict_policy_removes_disallowed_entities_and_their_edges():
    results = [({"server": [{"entity_type": " Service "}],
                 "city": [{"entity_type": "Location"}]},
                {("server", "city"): [{"keywords": "located"}]})]
    stats = {}
    filtered = enforce_entity_types(results, {"extraction_mode": "strict", "entity_types": ["service"]}, stats)
    assert set(filtered[0][0]) == {"server"}
    assert filtered[0][1] == {}
    assert stats["policy_rejections"] == {"entity_type_not_allowed": 1, "relation_endpoint_not_allowed": 1}
    assert "city" in results[0][0]  # SDK result remains unchanged.


def test_enhanced_mode_does_not_apply_strict_whitelist():
    results = [({"city": [{"entity_type": "Location"}]}, {})]
    assert enforce_entity_types(results, {"extraction_mode": "enhanced"}, {}) is results


def test_entity_exclusions_apply_in_enhanced_mode_and_remove_connected_edges():
    results = [(
        {
            "Unknown": [{"entity_type": "Other"}],
            "/usr/local/app": [{"entity_type": "File"}],
            "第12页": [{"entity_type": "Other"}],
            "server.log": [{"entity_type": "File"}],
            "Server": [{"entity_type": "Service"}],
            "Database": [{"entity_type": "Database"}],
        },
        {
            ("Unknown", "Server"): [{"keywords": "mentions"}],
            ("/usr/local/app", "Server"): [{"keywords": "mentions"}],
            ("第12页", "Server"): [{"keywords": "mentions"}],
            ("server.log", "Server"): [{"keywords": "mentions"}],
            ("Server", "Database"): [{"keywords": "depends on"}],
        },
    )]
    stats = {}
    filtered = enforce_entity_types(results, {
        "extraction_mode": "enhanced",
        "entity_exclusion_rules": [
            "unknown", "contains:/usr/", "prefix:第", "suffix:.log",
        ],
    }, stats)

    assert set(filtered[0][0]) == {"Server", "Database"}
    assert set(filtered[0][1]) == {("Server", "Database")}
    assert stats["policy_rejections"] == {
        "relation_endpoint_not_allowed": 4,
        "entity_name_excluded": 4,
    }
    assert "Unknown" in results[0][0]


def test_relation_exclusions_match_keywords_without_filtering_descriptions():
    records = [
        {"keywords": "Unknown", "description": "real relationship"},
        {"keywords": "temporary link", "description": "real relationship"},
        {"keywords": "depends on", "description": "contains temporary link"},
    ]
    stats = {}
    filtered = enforce_entity_types(
        [({"A": [{}], "B": [{}]}, {("A", "B"): records})],
        {
            "extraction_mode": "assist",
            "relation_exclusion_rules": ["unknown", "contains:temporary"],
        },
        stats,
    )

    assert filtered[0][1][("A", "B")] == [records[2]]
    assert stats["policy_rejections"] == {"relation_type_excluded": 2}


@pytest.mark.parametrize("rules", [["contains:"], ["x" * 201]])
def test_invalid_exclusion_rules_are_configuration_errors(rules):
    with pytest.raises(ValueError, match="entity_exclusion_rules"):
        enforce_entity_types([], {
            "extraction_mode": "assist",
            "entity_exclusion_rules": rules,
        }, {})


def test_empty_strict_whitelist_is_configuration_error():
    with pytest.raises(ValueError, match="whitelist"):
        enforce_entity_types([], {"extraction_mode": "strict"}, {})


def test_strict_policy_preserves_relations_to_entities_in_other_chunks():
    results = [
        ({"A": [{"entity_type": "Service"}]}, {("A", "B"): [{"keywords": "depends on"}]}),
        ({"B": [{"entity_type": "Service"}]}, {}),
    ]
    stats = {}
    filtered = enforce_entity_types(results, {"extraction_mode": "strict", "entity_types": ["Service"]}, stats)
    assert ("A", "B") in filtered[0][1]
    assert stats["entity_count"] == 2
    assert stats["relation_count"] == 1
    assert stats["policy_rejections"]["relation_endpoint_not_allowed"] == 0


def test_blank_type_does_not_allow_untyped_entities():
    with pytest.raises(ValueError, match="whitelist"):
        enforce_entity_types([], {"extraction_mode": "strict", "entity_types": [" "]}, {})


def test_strict_relation_categories_filter_records_without_inventing_labels():
    records = [{'keywords': ' Depends On '}, {'keywords': 'unrelated'},
               {'keywords': 'depends on, unrelated'}, {'description': 'depends on'}]
    results = [({'A': [{'entity_type': 'Service'}], 'B': [{'entity_type': 'Service'}]},
                {('A', 'B'): records})]
    stats = {}
    filtered = enforce_entity_types(results, {'extraction_mode': 'strict',
        'entity_types': ['Service'], 'relation_types': ['depends on']}, stats)
    assert filtered[0][1][('A', 'B')] == [records[0]]
    assert len(records) == 4
    assert stats['policy_rejections']['relation_type_not_allowed'] == 3
    assert stats['relation_count'] == 1
