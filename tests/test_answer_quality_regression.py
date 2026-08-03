import asyncio
from types import SimpleNamespace

from src.api import server


def test_unrelated_question_rejects_knowledge_base_context():
    citations = [
        {
            "index": 1,
            "doc_name": "Rsync学习文档.md",
            "file_path": "Rsync学习文档.md",
            "excerpt": "Rsync 用于文件同步，客户端通过脚本连接服务端。",
        }
    ]

    assert not server._citations_are_relevant("今天天气怎么样", citations, [])


def test_normal_business_numbers_are_not_flagged_as_noise():
    answer = (
        "### 原因\n"
        "B化工公司位于X省Y市，主要生产车间在2024年因环保违规被停产整顿3个月，"
        "这会影响A药厂的原料供应。[1]"
    )

    assert server._generated_answer_quality_issues(answer) == []


def test_bare_numeric_line_is_preserved():
    answer = (
        "### 原因\n"
        "C医院存在断供风险，原因是A药厂的核心原料供应商停产。[1]\n\n"
        "### 链路\n"
        "1\n\n"
        "### 排查步骤\n"
        "1. 联系供应商确认复产时间。[1]"
    )

    assert server._generated_answer_quality_issues(answer) == []
    cleaned, remaining = server._salvage_generated_answer(answer)
    assert remaining == []
    assert "\n1\n" in f"\n{cleaned}\n"
    assert "联系供应商确认复产时间" in cleaned


def test_answer_prompt_forbids_copying_and_document_listing():
    messages = server._build_answer_messages(
        "新闻资讯数据不全是什么原因",
        [
            {
                "index": 1,
                "doc_name": "客户端.txt",
                "chunk_index": 0,
                "excerpt": "客户端从infohost取新闻资讯数据并展示",
            },
            {
                "index": 2,
                "doc_name": "uts.txt",
                "chunk_index": 0,
                "excerpt": "uts从总部数据库同步数据到本地sqlserver",
            },
        ],
        [],
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "不要把原文逐条搬运成答案" in combined
    assert "不要逐条照抄资料" in combined
    assert "不要把命中的文档逐个列成清单" in combined
    assert "资料不足" in combined


def test_missing_citations_returns_context_insufficient_without_model_call(monkeypatch):
    class FailIfCalled:
        def __init__(self, _config):
            raise AssertionError("model should not be called without citations")

    monkeypatch.setattr(server, "SiliconFlowBackend", FailIfCalled)

    answer = asyncio.run(server._generate_answer_text("无关问题", [], []))

    assert "知识库上下文不足" in answer


def test_good_synthesized_answer_is_not_replaced_by_fallback(monkeypatch):
    class FakeBackend:
        def __init__(self, _config):
            pass

        async def chat(self, **_kwargs):
            return SimpleNamespace(
                content=(
                    "### 原因\n"
                    "新闻资讯数据不全可能出在客户端展示链路和本地资讯文件同步链路之间。[1][2]\n\n"
                    "### 排查步骤\n"
                    "1. 先确认客户端是否能从 infohost 正常读取资讯数据。[1]\n"
                    "2. 再检查 uts 同步到本地 sqlserver 的任务是否成功。[2]"
                )
            )

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
            "新闻资讯数据不全是什么原因",
            [
                {"index": 1, "doc_name": "客户端.txt", "chunk_index": 0, "excerpt": "客户端从infohost取新闻资讯数据并展示"},
                {"index": 2, "doc_name": "uts.txt", "chunk_index": 0, "excerpt": "uts从总部数据库同步数据到本地sqlserver"},
            ],
            [],
        )
    )

    assert "回答生成出现异常" not in answer
    assert "客户端展示链路" in answer
    assert "[1]" in answer
