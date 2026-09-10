"""Actual SDK extraction/storage + application shadow publication, no hosted models."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from src import lightrag_service as service_module
from src.api import server
from src.doc_processor.parsers.base_parser import Document
from src.index_validation import validate_index, validate_embedding_metadata
from src.lightrag_service import LightRAGService


@pytest.mark.parametrize('interrupt_publication', [False, True])
@pytest.mark.parametrize('use_task_worker', [False, True])
def test_real_extraction_rebuild_publishes_graph_or_restores_previous_generation(tmp_path, monkeypatch, interrupt_publication, use_task_worker):
    config = {'paths': {'data_dir': str(tmp_path), 'lightrag_dir': str(tmp_path / 'active')},
              'chunking': {'chunk_size': 256, 'chunk_overlap': 16},
              'lightrag': {'entity_extract_max_gleaning': 0, 'entity_extraction_use_json': True}}
    model = {'model': 'fixture', 'base_url': 'http://fixture.invalid', 'api_key': 'fixture', 'timeout': 5}
    runtime = {'chat': model, 'kg': model, 'rerank': {**model, 'enabled': False},
               'embedding': {**model, 'embed_dim': 8, 'embed_max_chars': 4096, 'embed_max_tokens': 480}}
    calls = {'llm': [], 'embedding': []}

    async def complete(*args, **kwargs):
        prompt = kwargs.get('prompt', args[0] if args else '')
        name = 'CurrentAtlas' if 'CurrentAtlas' in prompt else 'LegacyAtlas'
        calls['llm'].append(name)
        return json.dumps({'entities': [
            {'name': name, 'type': 'concept', 'description': f'{name} is the documented application.'},
            {'name': 'CedarStore', 'type': 'concept', 'description': 'CedarStore persists application records.'},
        ], 'relationships': [
            {'source': name, 'target': 'CedarStore', 'keywords': 'uses',
             'description': f'{name} uses CedarStore to persist its application records.'},
        ]})

    async def embed(texts, **kwargs):
        calls['embedding'].extend(texts)
        return np.ones((len(texts), 8), dtype=np.float32)

    monkeypatch.setattr(service_module, 'get_runtime_model_config', lambda _: deepcopy(runtime))
    monkeypatch.setattr(service_module, 'openai_complete_if_cache', complete)
    monkeypatch.setattr(service_module, 'openai_embed', SimpleNamespace(func=embed))
    monkeypatch.setattr(server, 'get_config', lambda: config)
    monkeypatch.setattr(server, '_index_tasks', {})
    monkeypatch.setattr(server, '_uploaded_files', {})
    monkeypatch.setattr(server, 'UPLOAD_DIR', tmp_path / 'uploads')
    monkeypatch.setattr(server, 'RAW_TEXT_DIR', tmp_path / 'raw_text')
    monkeypatch.setattr(server, 'INDEX_TASKS_DIR', tmp_path / 'index_tasks')
    active = LightRAGService(config, workspace='kb')
    monkeypatch.setattr(server, 'get_lightrag_service', lambda _: active)

    async def reset(_):
        await active.finalize()

    monkeypatch.setattr(server, 'reset_lightrag_service_async', reset)
    task = {'task_id': 'abcdef123456', 'workspace': 'kb', 'kind': 'rebuild',
            'status': 'running', 'doc_names': ['source.txt'], 'request': {}}
    server._index_tasks[task['task_id']] = task

    def document(name):
        return Document(doc_id='source', file_name='source.txt', file_path=str(tmp_path / 'source.txt'),
                        file_type='txt', raw_text=f'{name} uses CedarStore to persist application records. '
                        'The storage service retains records across application restarts.', metadata={})

    def graph(service):
        return nx.read_graphml(service.workspace_dir / 'graph_chunk_entity_relation.graphml')

    async def run():
        shadow = None
        reopened = None
        try:
            await active.index_document(document('LegacyAtlas'))
            await active.finalize()
            assert graph(active).has_edge('LegacyAtlas', 'CedarStore')
            old_manifest = deepcopy(active._load_manifest())
            shadow = await server._prepare_shadow_rebuild(task, active)
            original_commit = server._commit_shadow_rebuild

            async def checked_commit(current_task, candidate):
                nonlocal shadow
                shadow = candidate
                await shadow.finalize()
                assert graph(shadow).has_edge('CurrentAtlas', 'CedarStore')
                assert 'LegacyAtlas' not in graph(shadow)
                assert graph(active).has_edge('LegacyAtlas', 'CedarStore')
                assert active._load_manifest() == old_manifest
                await original_commit(current_task, shadow)

            monkeypatch.setattr(server, '_commit_shadow_rebuild', checked_commit)

            if interrupt_publication:
                original_replace = server.os.replace
                def replace(source, destination):
                    if str(source) == str(shadow.manifest_path.resolve()):
                        raise OSError('fixture interruption after directory publication')
                    return original_replace(source, destination)
                monkeypatch.setattr(server.os, 'replace', replace)

            if use_task_worker:
                source = server._resolve_upload_path('source.txt', 'kb', create_dir=True)
                source.write_text(document('CurrentAtlas').raw_text, encoding='utf-8')
                await server._run_index_task(task['task_id'], server.BatchIndexRequest(
                    workspace='kb', doc_names=['source.txt'], index_mode='complete'))
                assert task['status'] == ('failed' if interrupt_publication else 'succeeded')
                assert task['phase'] == 'done'
                assert task['results'][0]['kg_entity_count'] == 2
                assert task['results'][0]['kg_relation_count'] == 1
                saved_task = json.loads((server.INDEX_TASKS_DIR / f"{task['task_id']}.json").read_text(encoding='utf-8'))
                assert saved_task['status'] == task['status']
                assert saved_task['phase'] == 'done'
                assert saved_task['results'] == task['results']
            elif interrupt_publication:
                await shadow.index_document(document('CurrentAtlas'))
                with pytest.raises(OSError, match='fixture interruption'):
                    await server._commit_shadow_rebuild(task, shadow)
            else:
                await shadow.index_document(document('CurrentAtlas'))
                await server._commit_shadow_rebuild(task, shadow)

            if interrupt_publication:
                assert active._load_manifest() == old_manifest
                assert graph(shadow).has_edge('CurrentAtlas', 'CedarStore')
            else:
                assert server._verify_completed_publication(task)

            expected = 'LegacyAtlas' if interrupt_publication else 'CurrentAtlas'
            absent = 'CurrentAtlas' if interrupt_publication else 'LegacyAtlas'
            summary = validate_index(active.workspace_dir, active._load_manifest(), ['source.txt'])
            validate_embedding_metadata(active.workspace_dir, active.embedding_meta_path, 'kb')
            assert summary['entities'] == 2
            assert summary['chunks'] > 0
            reopened = LightRAGService(config, workspace='kb')
            rag = await reopened.get_rag()
            assert await rag.chunk_entity_relation_graph.has_edge(expected, 'CedarStore')
            assert not await rag.chunk_entity_relation_graph.has_node(absent)
            item = next(iter(reopened._load_manifest()['documents'].values()))
            assert (await rag.full_docs.get_by_id(item['active_index_doc_id']))['content'].startswith(expected)
            edge = await rag.chunk_entity_relation_graph.get_edge(expected, 'CedarStore')
            assert edge['source_id'] in item['chunks_list']
            for chunk_id in item['chunks_list']:
                chunk = await rag.text_chunks.get_by_id(chunk_id)
                assert chunk['full_doc_id'] == item['active_index_doc_id']
                assert expected in chunk['content']
                assert absent not in chunk['content']
            assert item['kg_status'] == 'complete'
            assert calls['llm'] == ['LegacyAtlas', 'CurrentAtlas']
            assert calls['embedding']
        finally:
            for service in (active, shadow, reopened):
                if service is not None:
                    await service.finalize()

    asyncio.run(run())
