import pytest

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
            "entity_exclusion_rules": ["contains:/tmp/"],
            "relation_exclusion_rules": ["unknown"],
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
    assert "contains:/tmp/" in guidance
    assert "unknown" in guidance
    assert "enforced again before graph storage" in guidance


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
    assert "Other unless explicitly listed" in guidance
    assert "Do not force unrelated entities into the nearest type" in guidance


def test_apply_graph_rule_template_supports_noise_reducing_profile(tmp_path):
    service = LightRAGService(
        config={"paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "lightrag")}},
        workspace="new_workspace",
    )

    config = service.apply_graph_rule_template(
        "general_knowledge",
        extraction_mode="enhanced",
        allow_other_entity_type=False,
    )

    assert config["rule_template_id"] == "general_knowledge"
    assert config["extraction_mode"] == "enhanced"
    assert config["allow_other_entity_type"] is False
    assert "domain enhanced" in config["effective_extraction_prompt"]
    assert "do not use `Other`" in config["effective_extraction_prompt"]


def test_custom_rule_template_round_trips_exclusion_rules(tmp_path):
    service = LightRAGService(
        config={"paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "lightrag")}},
        workspace="template_exclusions",
    )
    saved = service.save_graph_rule_template({
        "name": "低噪声模板",
        "entity_types": ["服务"],
        "relation_types": ["依赖"],
        "entity_exclusion_rules": ["contains:/tmp/"],
        "relation_exclusion_rules": ["unknown"],
    })

    config = service.apply_graph_rule_template(saved["id"], extraction_mode="enhanced")

    assert config["entity_exclusion_rules"] == ["contains:/tmp/"]
    assert config["relation_exclusion_rules"] == ["unknown"]
    assert service.load_graph_governance()["entity_exclusion_rules"] == ["contains:/tmp/"]


def test_invalid_exclusion_rule_is_not_saved_as_template(tmp_path):
    service = LightRAGService(
        config={"paths": {"data_dir": str(tmp_path), "lightrag_dir": str(tmp_path / "lightrag")}},
        workspace="invalid_template",
    )

    with pytest.raises(ValueError, match="entity_exclusion_rules"):
        service.save_graph_rule_template({
            "name": "无效模板",
            "entity_exclusion_rules": ["contains:"],
        })

    assert service._load_custom_graph_rule_templates() == []
