import asyncio
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from src import lightrag_service as module
from src.doc_processor.parsers.base_parser import Document
from src.lightrag_service import LightRAGService


@pytest.mark.parametrize('max_chars,max_tokens,text', [
    (64, 480, 'The application writes durable records into the database. ' * 25 + 'TAIL_EVIDENCE'),
    (1000, 32, '数据库配置🙂必须保留中文和标点。' * 35 + '末尾关键证据'),
])
def test_actual_embedding_inputs_cover_all_preview_chunks_without_prefix_truncation(tmp_path, monkeypatch, max_chars, max_tokens, text):
    service = LightRAGService({'paths': {'data_dir': str(tmp_path)},
                              'chunking': {'chunk_size': 256, 'chunk_overlap': 20}}, workspace='kb')
    embedding = {'model': 'fixture', 'api_key': 'fixture', 'base_url': 'http://fixture.invalid',
                 'embed_dim': 8, 'embed_max_chars': max_chars, 'embed_max_tokens': max_tokens}
    monkeypatch.setattr(service, '_runtime_models', lambda: {
        'embedding': embedding, 'chat': {'model': 'fixture'}, 'kg': {'model': 'fixture'},
        'rerank': {'model': '', 'base_url': '', 'api_key': '', 'enabled': False}})
    captured = []

    async def embed(texts, **kwargs):
        captured.extend(texts)
        return np.ones((len(texts), 8), dtype=np.float32)

    async def no_model(*args, **kwargs):
        raise AssertionError('Fast indexing must not call the extraction model')

    monkeypatch.setattr(module, 'openai_embed', SimpleNamespace(func=embed))
    monkeypatch.setattr(service, '_make_llm_func_for', lambda _: no_model)
    doc = Document(doc_id='source', file_name='source.txt', file_path='source.txt', file_type='txt', raw_text=text)

    async def run():
        try:
            preview = await service.preview_document_chunks(doc)
            assert len(preview) > 1
            covered = set()
            for row in preview:
                span = row['_source_span']
                assert text[span['start']:span['end']] == row['content']
                covered.update(range(span['start'], span['end']))
                normalized = module.EMBEDDING_WHITESPACE_RE.sub(' ', row['content']).strip()
                assert service._prepare_embedding_text(row['content'], max_chars, max_tokens) == normalized
            assert all(index in covered for index, char in enumerate(text) if not char.isspace())
            result = await service.index_document(doc, index_mode='fast')
            actual = await service.get_document_chunks(result['doc_id'])
            assert [row['text'] for row in actual] == [row['content'] for row in preview]
            assert Counter(captured) == Counter(module.EMBEDDING_WHITESPACE_RE.sub(' ', row['content']).strip() for row in preview)
            assert all(len(value) <= max_chars for value in captured)
            assert '\ufffd' not in ''.join(captured)
        finally:
            await service.finalize()

    asyncio.run(run())
