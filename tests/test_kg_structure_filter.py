import pytest

from src.lightrag_service import LightRAGService


@pytest.fixture
def service(tmp_path):
    return LightRAGService({"paths": {"data_dir": str(tmp_path)}}, workspace="kb")


SETTINGS = {"min_substantive_chars": 4, "symbol_ratio_threshold": 0.99,
            "structured_line_ratio": 0.75}


@pytest.mark.parametrize("indent", ["    ", "\t"])
def test_indentation_is_preserved_for_code_noise_detection(service, indent):
    text = "\n".join(indent + "x []|:." for _ in range(4))
    assert service._low_value_kg_chunk_reason(text, SETTINGS) == "code_noise"


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_fenced_content_counts_as_code_not_only_fence_markers(service, fence):
    text = fence + "\n" + "\n".join("x []|:." for _ in range(6)) + "\n" + fence
    assert service._low_value_kg_chunk_reason(text, SETTINGS) == "code_noise"


def test_meaningful_operations_code_is_not_excluded_just_for_being_code(service):
    text = "启用审计插件之前备份配置并检查版本兼容性。\n```sql\nINSTALL PLUGIN audit_log SONAME 'audit_log.so';\nSHOW VARIABLES LIKE 'audit_log%';\n```"
    assert service._low_value_kg_chunk_reason(text, SETTINGS) == ""
