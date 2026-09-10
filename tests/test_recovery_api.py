import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.api import server


@pytest.fixture
def task(monkeypatch):
    value = dict(task_id='task', workspace='kb', kind='rebuild', status='failed', phase='recovery_required')
    monkeypatch.setattr(server, '_index_tasks', {'task': value})
    monkeypatch.setattr(server, '_persist_index_task', lambda _: None)
    return value


@pytest.mark.parametrize('committed', [True, False])
def test_recovery_endpoint_clears_quarantine_only_after_verification(task, monkeypatch, committed):
    restore = AsyncMock(return_value=True)
    monkeypatch.setattr(server, '_verify_completed_publication', lambda _: committed)
    monkeypatch.setattr(server, '_restore_interrupted_publication', restore)
    asyncio.run(server.recover_index_task('task', 'kb'))
    assert task['phase'] == 'done'
    assert task['status'] == ('succeeded' if committed else 'failed')
    assert restore.await_count == (0 if committed else 1)


def test_recovery_failure_retains_quarantine(task, monkeypatch):
    monkeypatch.setattr(server, '_verify_completed_publication', lambda _: False)
    monkeypatch.setattr(server, '_restore_interrupted_publication', AsyncMock(side_effect=RuntimeError('ambiguous')))
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.recover_index_task('task', 'kb'))
    assert error.value.detail['code'] == 'INDEX_RECOVERY_REQUIRED'
    assert task['phase'] == 'recovery_required'


def test_recovery_rejects_other_workspace_and_active_writer(task, monkeypatch):
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.recover_index_task('task', 'other'))
    assert error.value.status_code == 404
    server._index_tasks['writer'] = dict(workspace='kb', status='running')
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.recover_index_task('task', 'kb'))
    assert error.value.status_code == 409
    assert task['phase'] == 'recovery_required'
