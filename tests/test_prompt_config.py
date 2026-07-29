from src.api.server import DEFAULT_ANSWER_SYSTEM_PROMPT, ModelConfig, _build_answer_messages


def test_model_config_includes_answer_system_prompt():
    config = ModelConfig()
    assert config.answer_system_prompt == DEFAULT_ANSWER_SYSTEM_PROMPT


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
