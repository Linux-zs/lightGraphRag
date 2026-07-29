from src.api import server


def test_workspace_prompts_are_saved_independently(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE_SETTINGS_DIR", tmp_path / "settings")

    server._save_workspace_settings(
        "workspace_a",
        {"answer_system_prompt": "A knowledge-base prompt"},
    )
    server._save_workspace_settings(
        "workspace_b",
        {"answer_system_prompt": "B knowledge-base prompt"},
    )

    assert (
        server._load_workspace_settings("workspace_a")["answer_system_prompt"]
        == "A knowledge-base prompt"
    )
    assert (
        server._load_workspace_settings("workspace_b")["answer_system_prompt"]
        == "B knowledge-base prompt"
    )


def test_non_default_workspace_uses_generic_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE_SETTINGS_DIR", tmp_path / "settings")

    prompt = server._load_workspace_settings("medical_kb")["answer_system_prompt"]

    assert "知识库问答助手" in prompt
    assert "通达信" not in prompt


def test_fallback_answer_contains_no_legacy_domain_playbook():
    answer = server._fallback_answer_from_citations(
        "rsync 和新闻资讯应该怎么处理",
        [
            {
                "index": 1,
                "doc_name": "general.txt",
                "excerpt": "蝴蝶效应描述初始条件的微小变化可能对长期结果产生显著影响。",
            }
        ],
    )

    assert "[1]" in answer
    assert "通达信总部" not in answer
    assert "infohost" not in answer
    assert "zxdbtools" not in answer
