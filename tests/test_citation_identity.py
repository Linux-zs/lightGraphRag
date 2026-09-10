from src.api.server import _format_chat_citations, _dump_chat_citations
from src.lightrag_service import LightRAGService


def test_stable_chunk_id_survives_chat_serialization(tmp_path):
    service = LightRAGService({"paths": {"data_dir": str(tmp_path)}}, workspace="kb")
    raw = {"data": {"chunks": [{
        "file_path": "source.txt", "chunk_id": "chunk-eight",
        "chunk_order_index": 8, "content": "real evidence",
    }]}}
    result = _dump_chat_citations(_format_chat_citations(service._citations_from_raw(raw)))
    assert result[0]["index"] == 1
    assert result[0]["chunk_index"] == 8
    assert result[0]["chunk_id"] == "chunk-eight"


def test_reference_number_is_not_a_stable_chunk_id(tmp_path):
    service = LightRAGService({"paths": {"data_dir": str(tmp_path)}}, workspace="kb")
    citations = service._citations_from_raw({"data": {"chunks": [
        {"file_path": "same.txt", "reference_id": "1", "content": "first passage"},
        {"file_path": "same.txt", "reference_id": "1", "content": "second passage"},
    ]}})
    assert len(citations) == 2
    assert all(c["chunk_id"] == "" and c["chunk_index"] == -1 for c in citations)


def test_answer_evidence_is_not_limited_to_ui_excerpt_or_six_sources(tmp_path, monkeypatch):
    from src.api import server
    monkeypatch.setattr(server, "_load_workspace_settings", lambda _: {})
    service = LightRAGService({"paths": {"data_dir": str(tmp_path)}}, workspace="kb")
    citations = service._citations_from_raw({"data": {"chunks": [
        {"file_path": f"source-{i}.txt", "chunk_id": f"chunk-{i}",
         "content": "背景内容。" * 70 + f"关键结论{i}。"}
        for i in range(12)
    ]}})
    prompt = server._build_answer_messages("问题", citations, [], "kb")[-1]["content"]
    assert "关键结论0" in prompt
    assert "关键结论11" in prompt
    assert "[12]" in prompt
    assert all(len(c["excerpt"]) <= 240 for c in citations)
    serialized = server._dump_chat_citations(server._format_chat_citations(citations))
    assert all("answer_content" not in c for c in serialized)


def test_technical_evidence_preserves_code_identifiers_and_comparisons(monkeypatch):
    from src.api import server
    monkeypatch.setattr(server, "_load_workspace_settings", lambda _: {})
    source = '配置示例：\n```sql\nSET max_connections = 200;\nSELECT user_id FROM users WHERE age >= 18;\n```\n| name | value |\n| --- | --- |\n| worker_count | 4 |'
    citations = [{"index": 1, "doc_name": "config.md", "answer_content": source}]
    prompt = server._build_answer_messages('如何配置？', citations, [])[-1]['content']
    assert source in prompt
    assert len(server._clean_excerpt_for_answer(source, 30)) <= 30
    assert server._clean_excerpt_for_answer(source, 0) == ''


def test_graph_evidence_and_provenance_reach_answer_prompt(monkeypatch):
    from src.api import server
    monkeypatch.setattr(server, "_load_workspace_settings", lambda _: {})
    data = {"relationships": [{"src_id": "A", "tgt_id": "B",
            "description": "depends on", "source_id": "chunk-eight"}]}
    messages = server._build_answer_messages("why", [], [], retrieval_data=data)
    assert "depends on" in messages[-1]["content"]
    assert "chunk-eight" in messages[-1]["content"]
    assert "不得编造引用编号" in messages[0]["content"]
