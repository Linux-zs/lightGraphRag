import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.lightrag_service import LightRAGService


def test_preview_context_passes_history_to_sdk(tmp_path, monkeypatch):
    service = LightRAGService({"paths": {"data_dir": str(tmp_path)}}, workspace="kb")
    query = AsyncMock(return_value={})
    service._rag = SimpleNamespace(aquery_llm=query)
    monkeypatch.setattr(service, "assert_embedding_compatible", lambda: None)
    monkeypatch.setattr(service, "_ensure_current_index_versions", AsyncMock())
    history = [{"role": "user", "content": "MySQL 审计插件有哪些？"}]
    asyncio.run(service.preview_context("第二种如何启用？", history=history))
    assert query.call_args.kwargs["param"].conversation_history == history
    assert query.call_args.kwargs["param"].only_need_context is True
