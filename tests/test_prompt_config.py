import asyncio
from types import SimpleNamespace

from src.api import server
from src.api.server import DEFAULT_ANSWER_SYSTEM_PROMPT, ModelConfig, _build_answer_messages


def test_model_config_includes_answer_system_prompt():
    config = ModelConfig()
    assert config.answer_system_prompt == DEFAULT_ANSWER_SYSTEM_PROMPT
    assert config.answer_prompt_template_id == "recommended"


def test_answer_prompt_instructs_synthesis_not_copying():
    messages = _build_answer_messages(
        "新闻资讯数据不全是什么原因",
        [
            {
                "index": 1,
                "doc_name": "客户端.txt",
                "chunk_index": 0,
                "excerpt": "客户端从infohost取新闻资讯数据并展示",
            }
        ],
        [],
    )

    system = messages[0]["content"]
    assert "不要把原文逐条搬运成答案" in system
    assert "综合多条资料" in system


def test_detects_numeric_model_noise():
    answer = "### 结论\n这是有依据的回答。[1]\n\n11111\n111li\n11 1\n1"
    assert server._is_bad_generated_answer(answer) is True


def test_salvages_numeric_noise_without_replacing_useful_answer():
    answer = "### 结论\n这是已经组织好的回答。[1]\n\n11111\n111li\n11\n1\n\n### 建议\n继续观察。"
    cleaned, remaining = server._salvage_generated_answer(answer)

    assert remaining == []
    assert "这是已经组织好的回答" in cleaned
    assert "继续观察" in cleaned
    assert "11111" not in cleaned
    assert "111li" not in cleaned


def test_missing_inline_citation_does_not_replace_good_answer(monkeypatch):
    class FakeBackend:
        def __init__(self, _config):
            pass

        async def chat(self, **_kwargs):
            return SimpleNamespace(content="这是模型组织后的正常回答，没有行内引用编号。")

        async def close(self):
            pass

    monkeypatch.setattr(server, "SiliconFlowBackend", FakeBackend)
    monkeypatch.setattr(
        server,
        "get_runtime_model_config",
        lambda _cfg: {
            "chat": {
                "base_url": "http://127.0.0.1/v1",
                "api_key": "",
                "model": "test-chat",
            }
        },
    )
    monkeypatch.setattr(server, "get_config", lambda: {})

    answer = asyncio.run(
        server._generate_answer_text(
            "问题",
            [{"index": 1, "doc_name": "a.txt", "chunk_index": 0, "excerpt": "参考资料"}],
            [],
        )
    )
    assert answer == "这是模型组织后的正常回答，没有行内引用编号。"


def test_incomplete_answer_is_retried_with_conservative_parameters(monkeypatch):
    calls = []

    class FakeBackend:
        def __init__(self, _config):
            pass

        async def chat(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(content="### 原因\n已有正常说明。\n\n### 排查步骤\n1")
            return SimpleNamespace(content="### 原因\n已有正常说明。\n\n### 排查步骤\n1. 检查供应商状态。[1]")

        async def close(self):
            pass

    monkeypatch.setattr(server, "SiliconFlowBackend", FakeBackend)
    monkeypatch.setattr(
        server,
        "get_runtime_model_config",
        lambda _cfg: {
            "chat": {
                "base_url": "http://127.0.0.1/v1",
                "api_key": "",
                "model": "test-chat",
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.2,
            }
        },
    )
    monkeypatch.setattr(server, "get_config", lambda: {})

    answer = asyncio.run(
        server._generate_answer_text(
            "问题",
            [{"index": 1, "doc_name": "a.txt", "chunk_index": 0, "excerpt": "参考资料"}],
            [],
        )
    )

    assert len(calls) == 2
    assert calls[1]["temperature"] == 0.3
    assert calls[1]["frequency_penalty"] == 0.0
    assert answer.endswith("检查供应商状态。[1]")


def test_role_marker_tail_is_removed_and_marked_incomplete():
    cleaned = server._strip_lightrag_noise(
        "### 原因\n正常内容。\n\nassistant\n### 原因\n重复内容。"
    )
    assert cleaned == "### 原因\n正常内容。"


def test_prompt_templates_can_be_created_updated_and_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROMPT_TEMPLATES_PATH", tmp_path / "templates.json")

    created = asyncio.run(
        server.create_prompt_template(
            server.PromptTemplateRequest(
                name="排障模板",
                description="用于技术排障",
                content="请基于资料归纳排障步骤。",
            )
        )
    )
    assert created["id"].startswith("prompt_")
    assert len(server._load_prompt_templates()) == 2

    updated = asyncio.run(
        server.update_prompt_template(
            created["id"],
            server.PromptTemplateRequest(
                name="更新后的模板",
                description="",
                content="更新后的提示词",
            ),
        )
    )
    assert updated["content"] == "更新后的提示词"

    asyncio.run(server.delete_prompt_template(created["id"]))
    assert [item["id"] for item in server._load_prompt_templates()] == ["recommended"]
