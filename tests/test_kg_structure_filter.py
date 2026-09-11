import pytest

from src.lightrag_service import LightRAGService


@pytest.fixture
def service(tmp_path):
    return LightRAGService({"paths": {"data_dir": str(tmp_path)}}, workspace="kb")


SETTINGS = {"min_substantive_chars": 4, "symbol_ratio_threshold": 0.99,
            "structured_line_ratio": 0.75, "strip_bulk_record_noise": True,
            "bulk_record_min_lines": 6}


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


def test_bulk_sql_rows_are_removed_from_kg_copy_but_surrounding_prose_is_kept(service):
    rows = "\n".join(
        ["+----+----------+"]
        + [f"| {index} | city-{index} |" for index in range(1, 8)]
        + ["+----+----------+"]
    )
    original = f"查询仅用于验证升级结果。\n{rows}\n至此升级完成。"
    chunks = {"chunk-1": {"content": original, "file_path": "upgrade.md"}}

    filtered, stats = service._filter_kg_chunks(chunks)

    assert filtered["chunk-1"]["content"] == "查询仅用于验证升级结果。\n\n至此升级完成。"
    assert chunks["chunk-1"]["content"] == original
    assert stats["sanitized"] == 1
    assert stats["sanitized_reasons"] == {"bulk_table_rows": 9}
    assert stats["skipped"] == 0


def test_short_markdown_table_remains_available_to_kg_extraction(service):
    text = "参数说明：\n| 参数 | 含义 |\n| --- | --- |\n| max_connections | 最大连接数 |"

    cleaned, reasons, removed = service._sanitize_kg_chunk_text(text, SETTINGS)

    assert cleaned == text
    assert reasons == {}
    assert removed == 0


def test_successful_object_inventory_is_removed_after_repetition_threshold(service):
    listing = "\n".join(
        f"mysql.\n\nobject_{index}                              OK"
        for index in range(6)
    )
    text = f"检查表完整性：\n{listing}\n所有对象检查完成。"

    cleaned, reasons, removed = service._sanitize_kg_chunk_text(text, SETTINGS)

    assert "object_" not in cleaned
    assert "mysql." not in cleaned
    assert "检查表完整性" in cleaned
    assert "所有对象检查完成" in cleaned
    assert reasons == {"successful_status_listing": 12}
    assert removed == 12


def test_repeated_schema_inventory_is_removed_but_single_warning_is_kept(service):
    records = "\n".join(
        f"sakila.\n\ntable_{index}.\n\ncolumn_{index} - column's default character set: utf8"
        for index in range(4)
    )
    text = f"字符集检查发现如下对象：\n{records}\n建议升级为 utf8mb4。"

    cleaned, reasons, removed = service._sanitize_kg_chunk_text(text, SETTINGS)

    assert "table_" not in cleaned
    assert "column_" not in cleaned
    assert "建议升级为 utf8mb4" in cleaned
    assert reasons == {"schema_inventory": 12}
    assert removed == 12


def test_truncated_console_table_and_single_inventory_record_are_removed(service):
    text = """Database changed
MySQL [world]> select * from city limit 10;
+----+----------+
| ID | Name     |
+----+----------+
| 1  | Kabul    |
sakila.film_in_stock - PROCEDURE uses obsolete NO_AUTO_CREATE_USER sql_mode"""

    cleaned, reasons, removed = service._sanitize_kg_chunk_text(text, SETTINGS)

    assert "Kabul" not in cleaned
    assert "film_in_stock" not in cleaned
    assert "select * from city" in cleaned
    assert reasons == {
        "bulk_table_rows": 4,
        "obsolete_object_inventory": 1,
    }
    assert removed == 5
