import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api import server


def test_applying_template_preserves_strict_mode_and_other_setting(monkeypatch):
    from unittest.mock import Mock
    apply = Mock(return_value={})
    service = SimpleNamespace(load_graph_governance=lambda: {
        'extraction_mode': 'strict', 'allow_other_entity_type': False}, apply_graph_rule_template=apply)
    monkeypatch.setattr(server, '_index_tasks', {})
    monkeypatch.setattr(server, 'get_lightrag_service', lambda _: service)
    asyncio.run(server.apply_graph_rule_template(server.GraphRuleTemplateApplyRequest(workspace='kb', template_id='template')))
    apply.assert_called_once_with('template', extraction_mode='strict', allow_other_entity_type=False)


def test_invalid_strict_rules_are_rejected_before_service_write(monkeypatch):
    monkeypatch.setattr(server, '_index_tasks', {})
    monkeypatch.setattr(server, 'get_lightrag_service', lambda _: pytest.fail('must not persist invalid rules'))
    request = server.GraphGovernanceUpdate(workspace='kb', extraction_mode='strict', entity_types=[])
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.update_graph_governance_config(request))
    assert error.value.status_code == 400
    assert error.value.detail['code'] == 'INVALID_EXTRACTION_POLICY'


def test_backfill_does_not_merge_changed_policy_into_existing_graph():
    from unittest.mock import AsyncMock
    from src.lightrag_service import LightRAGService
    service = object.__new__(LightRAGService)
    service.assert_embedding_compatible = lambda: None
    service._load_manifest = lambda: {'documents': {'doc': {
        'doc_name': 'source.txt', 'indexed': True, 'kg_status': 'partial',
        'kg_policy_fingerprint': 'old-policy', 'chunks_list': ['chunk']}}}
    service.load_graph_governance = lambda: {'entity_types': ['Service']}
    service.graph_extraction_guidance = lambda config: 'new rules'
    service.get_rag = AsyncMock()
    with pytest.raises(ValueError, match='不能将新规则追加到旧图谱'):
        asyncio.run(service.backfill_document_graph('source.txt'))
    service.get_rag.assert_not_awaited()


def test_document_list_distinguishes_stale_current_and_unknown_policy(monkeypatch):
    from unittest.mock import AsyncMock
    from src.extraction_policy import policy_fingerprint
    from src.lightrag_service import LightRAGService
    service = object.__new__(LightRAGService)
    policy = {'entity_types': ['Service'], 'extraction_mode': 'strict'}
    fingerprint = policy_fingerprint(policy, 'guidance')
    base = dict(indexed=True, status='processed', kg_status='complete')
    service._load_manifest = lambda: {'documents': {
        'current': {**base, 'kg_policy_fingerprint': fingerprint},
        'stale': {**base, 'kg_policy_fingerprint': 'old'},
        'legacy': base,
        'fast': {**base, 'kg_status': 'skipped', 'kg_policy_fingerprint': 'old'},
    }}
    service.load_graph_governance = lambda: policy
    service.graph_extraction_guidance = lambda config: 'guidance'
    service.get_rag = AsyncMock(side_effect=RuntimeError('offline'))
    docs = {item['doc_id']: item for item in asyncio.run(service.list_documents())}
    assert docs['current']['kg_policy_stale'] is False
    assert docs['stale']['kg_policy_stale'] is True
    assert docs['legacy']['kg_policy_stale'] is None
    assert docs['fast']['kg_policy_stale'] is None


def test_persisted_policy_snapshot_survives_config_changes_and_is_not_public():
    from src.extraction_policy import policy_fingerprint
    config = {'entity_types': ['Service'], 'extraction_mode': 'strict'}
    task = {'graph_policy_snapshot': {'config': config, 'guidance': 'original references'},
            'graph_policy_fingerprint': policy_fingerprint(config, 'original references')}
    # A validated persisted snapshot must not read live governance at resume.
    server._assert_task_policy_snapshot(task, object())
    assert 'graph_policy_snapshot' not in server._public_index_task(task)
    task['graph_policy_snapshot']['guidance'] = 'altered'
    with pytest.raises(RuntimeError, match='快照损坏'):
        server._assert_task_policy_snapshot(task, object())


def test_task_policy_fingerprint_detects_changed_reference_content():
    policy = {'extraction_mode': 'strict', 'entity_types': ['Service']}
    service = SimpleNamespace(load_graph_governance=lambda: policy,
        graph_extraction_guidance=lambda config: 'original references')
    task = {'graph_policy_fingerprint': server._graph_policy_fingerprint(service)}
    server._assert_task_policy_snapshot(task, service)
    service.graph_extraction_guidance = lambda config: 'changed references'
    with pytest.raises(RuntimeError, match='抽取规则或参考资料'):
        server._assert_task_policy_snapshot(task, service)


@pytest.mark.parametrize('kind', ['single', 'batch', 'kg_backfill'])
@pytest.mark.parametrize('status', ['queued', 'running'])
def test_active_tasks_freeze_policy_only_in_their_workspace(monkeypatch, kind, status):
    monkeypatch.setattr(server, '_index_tasks', {'task': dict(task_id='task', workspace='kb', kind=kind, status=status)})
    with pytest.raises(HTTPException) as error:
        server._ensure_graph_policy_mutable('kb')
    assert error.value.detail['code'] == 'GRAPH_POLICY_BUSY'
    server._ensure_graph_policy_mutable('other')


def test_reference_upload_rechecks_after_reading_file(monkeypatch):
    monkeypatch.setattr(server, '_index_tasks', {})
    async def read(_):
        server._index_tasks['task'] = dict(task_id='task', workspace='kb', kind='batch', status='running')
        return 'reference content'
    monkeypatch.setattr(server, '_read_governance_reference_upload', read)
    monkeypatch.setattr(server, 'get_lightrag_service', lambda _: pytest.fail('must not save reference'))
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.upload_graph_reference(SimpleNamespace(filename='ref.txt'), 'kb'))
    assert error.value.detail['code'] == 'GRAPH_POLICY_BUSY'
