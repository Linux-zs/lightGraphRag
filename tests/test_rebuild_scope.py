from pathlib import Path

import pytest
from fastapi import HTTPException

from src.api import server


@pytest.mark.parametrize('replay_errors', [[], ['failed entity']])
def test_empty_rebuild_records_publication_phase_and_preserves_failure(replay_errors, tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    task = dict(task_id='task', workspace='kb', kind='rebuild', status='queued')
    monkeypatch.setattr(server, '_index_tasks', {})
    monkeypatch.setattr(server, '_persist_index_task', lambda _: None)
    monkeypatch.setattr(server, '_workspace_doc_names_for_rebuild', lambda _: [])
    monkeypatch.setattr(server, 'get_lightrag_service', lambda _: object())
    async def create(*_):
        server._index_tasks['task'] = task
        return task
    monkeypatch.setattr(server, '_create_index_task', create)
    shadow = SimpleNamespace(workspace_dir=tmp_path,
        replay_graph_audit=AsyncMock(return_value={'errors': replay_errors}), finalize=AsyncMock())
    monkeypatch.setattr(server, '_prepare_shadow_rebuild', AsyncMock(return_value=shadow))
    async def commit(current, _):
        assert current['phase'] == 'committing'
    publish = AsyncMock(side_effect=commit)
    monkeypatch.setattr(server, '_commit_shadow_rebuild', publish)
    result, _ = asyncio.run(server._start_workspace_rebuild(
        server.RebuildIndexRequest(workspace='kb'), reason='test', allow_empty=True, workspace_lock_held=True))
    if replay_errors:
        publish.assert_not_awaited()
        assert result['status'] == 'failed'
        assert '原索引保留' in result['message']
    else:
        publish.assert_awaited_once()
        assert result['status'] == 'succeeded'


def test_restart_completes_verified_publication_without_reindexing(monkeypatch):
    import asyncio
    task = dict(task_id='task', workspace='kb', kind='rebuild', status='running',
                phase='committing', doc_names=['source.txt'])
    monkeypatch.setattr(server, '_index_tasks', {'task': task})
    monkeypatch.setattr(server, '_persist_index_task', lambda _: None)
    monkeypatch.setattr(server, '_verify_completed_publication', lambda _: True)
    monkeypatch.setattr(server, '_spawn_background', lambda _: pytest.fail('must not reindex'))
    asyncio.run(server._resume_persisted_index_tasks())
    assert task['status'] == 'succeeded'
    assert task['phase'] == 'done'
    assert task['current'] == 1


@pytest.mark.parametrize('endpoint', ['list_documents', 'system_stats', 'delete_workspace'])
def test_recovery_guard_blocks_index_access_before_service(endpoint, monkeypatch):
    import asyncio
    monkeypatch.setattr(server, '_index_tasks', {
        'task': {'task_id': 'task', 'workspace': 'kb', 'phase': 'recovery_required', 'status': 'failed'}})
    def unexpected_service(_):
        pytest.fail('must not open an inconsistent index')
    monkeypatch.setattr(server, 'get_lightrag_service', unexpected_service)
    with pytest.raises(HTTPException) as error:
        asyncio.run(getattr(server, endpoint)('kb'))
    assert error.value.detail['code'] == 'INDEX_RECOVERY_REQUIRED'


def test_recovery_guard_blocks_model_signature_mutation(monkeypatch):
    monkeypatch.setattr(server, '_index_tasks', {
        'task': {'task_id': 'task', 'workspace': 'kb', 'phase': 'recovery_required', 'status': 'failed'}})
    with pytest.raises(HTTPException) as error:
        server._ensure_model_config_mutable()
    assert error.value.detail['code'] == 'INDEX_RECOVERY_REQUIRED'


def test_interrupted_commit_remains_quarantined_after_restart(tmp_path, monkeypatch):
    import asyncio

    task = {'task_id': 'interrupted', 'workspace': 'kb', 'kind': 'rebuild',
            'status': 'running', 'phase': 'committing', 'doc_names': ['source.txt']}
    monkeypatch.setattr(server, '_index_tasks', {'interrupted': task})
    monkeypatch.setattr(server, '_persist_index_task', lambda _: None)
    asyncio.run(server._resume_persisted_index_tasks())
    assert task['status'] == 'failed'
    assert task['phase'] == 'recovery_required'
    for _ in range(2):
        with pytest.raises(HTTPException) as error:
            server._ensure_workspace_available('kb')
        assert error.value.detail['code'] == 'INDEX_RECOVERY_REQUIRED'
        asyncio.run(server._resume_persisted_index_tasks())
    server._ensure_workspace_available('other')


def test_rebuild_prepare_preserves_existing_candidate_and_backup(tmp_path, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    candidate = tmp_path / 'rebuild_shadow' / 'task'
    backup = candidate / 'previous' / 'workspace'
    backup.mkdir(parents=True)
    sentinel = backup / 'old-index.txt'
    sentinel.write_text('recoverable', encoding='utf-8')
    update = AsyncMock()
    monkeypatch.setattr(server, 'get_config', lambda: {'paths': {'data_dir': str(tmp_path)}})
    monkeypatch.setattr(server, '_update_index_task', update)
    with pytest.raises(RuntimeError, match='retained for recovery'):
        asyncio.run(server._prepare_shadow_rebuild({'task_id': 'task'}, object()))
    assert sentinel.read_text(encoding='utf-8') == 'recoverable'
    update.assert_not_awaited()


def test_rebuild_prepare_rejects_non_leaf_task_id(tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setattr(server, 'get_config', lambda: {'paths': {'data_dir': str(tmp_path)}})
    with pytest.raises((ValueError, HTTPException)):
        asyncio.run(server._prepare_shadow_rebuild({'task_id': '../outside'}, object()))


class FakeService:
    def __init__(self, manifest):
        self._manifest = manifest

    def _load_manifest(self):
        return self._manifest


def test_rebuild_uses_only_workspace_manifest_documents(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "current.txt").write_text("current", encoding="utf-8")
    (upload_dir / "other.txt").write_text("other", encoding="utf-8")

    manifest = {
        "documents": {
            "doc_current": {
                "doc_name": "current.txt",
                "updated_at": "2026-07-29T10:00:00+08:00",
            }
        }
    }

    monkeypatch.setattr(server, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(server, "get_lightrag_service", lambda workspace: FakeService(manifest))

    assert server._workspace_doc_names_for_rebuild("workspace_a") == ["current.txt"]


def test_rebuild_rejects_manifest_documents_missing_from_upload_dir(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    manifest = {
        "documents": {
            "doc_missing": {
                "doc_name": "missing.txt",
                "updated_at": "2026-07-29T10:00:00+08:00",
            }
        }
    }

    monkeypatch.setattr(server, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(server, "get_lightrag_service", lambda workspace: FakeService(manifest))

    with pytest.raises(HTTPException) as error:
        server._workspace_doc_names_for_rebuild("workspace_a")
    assert error.value.status_code == 409
    assert error.value.detail["code"] == "REBUILD_SOURCES_UNAVAILABLE"
    assert "missing.txt" in error.value.detail["documents"][0]


def test_rebuild_rejects_partial_source_set(tmp_path, monkeypatch):
    (tmp_path / "present.txt").write_text("present", encoding="utf-8")
    manifest = {"documents": {
        "present": {"doc_name": "present.txt"},
        "missing": {"doc_name": "missing.txt"},
    }}
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(server, "get_lightrag_service", lambda workspace: FakeService(manifest))
    with pytest.raises(HTTPException) as error:
        server._workspace_doc_names_for_rebuild("workspace_a")
    assert error.value.detail["documents"] == ["missing.txt (source missing)"]


@pytest.mark.parametrize("names", [[""], ["present.txt", "present.txt"]])
def test_rebuild_rejects_ambiguous_manifest_instead_of_skipping_entries(tmp_path, monkeypatch, names):
    (tmp_path / "present.txt").write_text("present", encoding="utf-8")
    manifest = {"documents": {str(i): {"doc_name": name} for i, name in enumerate(names)}}
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(server, "get_lightrag_service", lambda workspace: FakeService(manifest))
    with pytest.raises(HTTPException) as error:
        server._workspace_doc_names_for_rebuild("workspace_a")
    assert error.value.status_code == 409
    assert error.value.detail["code"] == "REBUILD_SOURCES_UNAVAILABLE"
