from src.api.server import _citations_are_relevant


def test_matching_filename_is_not_answer_evidence():
    citations = [{"doc_name": "MySQL.md", "file_path": "MySQL/audit.md",
                  "excerpt": "员工示例表包含姓名和城市。"}]
    assert not _citations_are_relevant("MySQL 如何启用审计日志", citations, [])


def test_evidence_after_excerpt_and_eighth_source_is_considered():
    citations = [{"excerpt": "无关示例内容。"} for _ in range(8)]
    citations.append({"excerpt": "背景资料。", "answer_content": "背景资料。" * 100 + "MySQL 审计日志需要启用审计插件。"})
    assert _citations_are_relevant("MySQL 如何启用审计日志", citations, [])


def test_news_info_business_question_is_not_rejected_as_general_news():
    citations = [
        {
            "index": 1,
            "doc_name": "客户端.txt",
            "file_path": "客户端.txt",
            "excerpt": "客户端从infohost取新闻资讯数据并展示",
        },
        {
            "index": 2,
            "doc_name": "zxdbtools.txt",
            "file_path": "zxdbtools.txt",
            "excerpt": "zxdbtools从数据库将数据导出为本地资讯文件",
        },
    ]

    assert _citations_are_relevant("新闻资讯数据不全是什么原因", citations, [])


def test_weather_question_still_rejects_accidental_context_overlap():
    citations = [
        {
            "index": 1,
            "doc_name": "Rsync学习文档.md",
            "file_path": "Rsync学习文档.md",
            "excerpt": "Rsync 用于文件同步，客户端通过脚本连接服务端。",
        }
    ]

    assert not _citations_are_relevant("今天天气怎么样", citations, [])
