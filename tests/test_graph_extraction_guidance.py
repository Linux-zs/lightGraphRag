from src.lightrag_service import LightRAGService


def test_graph_guidance_assist_mode_treats_rules_as_hints(tmp_path):
    service = LightRAGService(
        config={"paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "lightrag")}},
        workspace="butterfly_test",
    )

    guidance = service.graph_extraction_guidance(
        config={
            "entity_types": ["产品"],
            "relation_types": ["供应"],
            "aliases_text": "",
            "extraction_prompt": "优先抽取供应链实体。",
            "reference_files": [],
            "extraction_mode": "assist",
            "allow_other_entity_type": True,
        }
    )

    assert "rules below are guidance" in guidance
    assert "not the source of truth" in guidance
    assert "Treat configured entity and relation types only as hints" in guidance
    assert "classify the entity as `Other` instead of dropping it" in guidance


def test_graph_guidance_strict_mode_is_explicit(tmp_path):
    service = LightRAGService(
        config={"paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "lightrag")}},
        workspace="strict_test",
    )

    guidance = service.graph_extraction_guidance(
        config={
            "entity_types": ["服务"],
            "relation_types": ["依赖"],
            "aliases_text": "",
            "extraction_prompt": "",
            "reference_files": [],
            "extraction_mode": "strict",
            "allow_other_entity_type": False,
        }
    )

    assert "strict whitelist" in guidance
    assert "hard constraints" in guidance
    assert "do not use `Other`" in guidance
