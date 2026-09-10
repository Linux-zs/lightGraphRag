import pytest

from src.publication_recovery import Artifact, rollback_artifacts


@pytest.mark.parametrize('interrupt_after', range(1, 7))
def test_fresh_process_recovers_after_abrupt_exit_at_every_rename(tmp_path, interrupt_after):
    import subprocess
    import sys
    from pathlib import Path

    for index in range(3):
        root = tmp_path / str(index)
        root.mkdir()
        (root / 'active').write_text(f'new-{index}', encoding='utf-8')
        (root / 'backup').write_text(f'old-{index}', encoding='utf-8')
    script = '''
import os, sys
from pathlib import Path
from src.publication_recovery import Artifact, rollback_artifacts
root = Path(sys.argv[1])
boundary = int(sys.argv[2])
items = [Artifact(root / str(i) / 'active', root / str(i) / 'candidate',
                  root / str(i) / 'backup', True, True) for i in range(3)]
rename = os.rename
count = 0
def interrupted(source, destination):
    global count
    rename(source, destination)
    count += 1
    if count == boundary:
        os._exit(73)
if boundary:
    os.rename = interrupted
rollback_artifacts(items)
'''
    repo = Path(__file__).resolve().parents[1]
    stopped = subprocess.run([sys.executable, '-c', script, str(tmp_path), str(interrupt_after)],
        cwd=repo, capture_output=True, text=True, timeout=20)
    assert stopped.returncode == 73, stopped.stderr
    recovered = subprocess.run([sys.executable, '-c', script, str(tmp_path), '0'],
        cwd=repo, capture_output=True, text=True, timeout=20)
    assert recovered.returncode == 0, recovered.stderr
    for index in range(3):
        root = tmp_path / str(index)
        assert (root / 'active').read_text(encoding='utf-8') == f'old-{index}'
        assert (root / 'candidate').read_text(encoding='utf-8') == f'new-{index}'
        assert not (root / 'backup').exists()


@pytest.mark.parametrize('pending_upload', [False, True])
def test_startup_restore_uses_validated_disk_journal(tmp_path, monkeypatch, pending_upload):
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from src.api import server

    base = tmp_path / 'rebuild_shadow' / 'task'
    active = tmp_path / 'active' / 'kb'
    active.mkdir(parents=True)
    (active / 'marker').write_text('new')
    backup = base / 'previous' / 'workspace'
    backup.mkdir(parents=True)
    (backup / 'marker').write_text('old')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"documents": {}}')
    if pending_upload:
        manifest.write_text(json.dumps({'documents': {'upload': {
            'doc_name': 'pending.txt', 'indexed': False, 'chunk_count': 0,
        }}}))
    candidate_manifest = base / 'manifest.json'
    candidate_manifest.write_text('{"documents": {}}')
    meta = tmp_path / 'embedding.json'
    expected = [(active, base / 'lightrag' / 'kb', backup, True, True),
        (manifest, candidate_manifest, base / 'previous' / 'manifest.json', True, True),
        (meta, base / 'embedding_meta' / 'kb.json', base / 'previous' / 'embedding_meta.json', False, False)]
    journal = dict(version=1, task_id='task', workspace='kb', state='prepared', artifacts=[
        dict(active=str(a.resolve()), candidate=str(c.resolve()), backup=str(b.resolve()),
             had_active=ha, had_candidate=hc) for a, c, b, ha, hc in expected])
    (base / 'publication.json').write_text(json.dumps(journal))
    service = SimpleNamespace(workspace_dir=active, manifest_path=manifest,
        embedding_meta_path=meta, _load_manifest=lambda: json.loads(manifest.read_text()))
    monkeypatch.setattr(server, 'get_config', lambda: {'paths': {'data_dir': str(tmp_path)}})
    monkeypatch.setattr(server, 'get_lightrag_service', lambda _: service)
    monkeypatch.setattr(server, 'reset_lightrag_service_async', AsyncMock())
    task = dict(task_id='task', workspace='kb')
    assert asyncio.run(server._restore_interrupted_publication(task))
    assert asyncio.run(server._restore_interrupted_publication(task))
    assert (active / 'marker').read_text() == 'old'
    assert (base / 'lightrag' / 'kb' / 'marker').read_text() == 'new'
    assert json.loads((base / 'publication.json').read_text())['state'] == 'rolled_back'


def test_first_generation_rollback_preserves_candidate_directory(tmp_path):
    active, candidate, backup = [tmp_path / name for name in ('active', 'candidate', 'backup')]
    active.mkdir()
    (active / 'data').write_text('new')
    item = Artifact(active, candidate, backup, False, True)
    rollback_artifacts([item])
    rollback_artifacts([item])
    assert not active.exists()
    assert (candidate / 'data').read_text() == 'new'


def test_overlapping_paths_are_rejected(tmp_path):
    with pytest.raises(RuntimeError, match='Overlapping'):
        rollback_artifacts([Artifact(tmp_path, tmp_path / 'candidate', tmp_path / 'backup', True, True)])


@pytest.mark.parametrize('stage', [0, 1, 2])
def test_rollback_every_publication_boundary_is_repeatable(tmp_path, stage):
    active, candidate, backup = [tmp_path / name for name in ('active', 'candidate', 'backup')]
    active.write_text('old')
    candidate.write_text('new')
    if stage >= 1:
        active.rename(backup)
    if stage >= 2:
        candidate.rename(active)
    artifact = Artifact(active, candidate, backup, True, True)
    rollback_artifacts([artifact])
    rollback_artifacts([artifact])
    assert active.read_text() == 'old'
    assert candidate.read_text() == 'new'
    assert not backup.exists()


def test_ambiguous_later_artifact_prevents_all_moves(tmp_path):
    items = []
    for name in ('first', 'second'):
        root = tmp_path / name
        root.mkdir()
        active, candidate, backup = [root / part for part in ('active', 'candidate', 'backup')]
        active.write_text('new')
        backup.write_text('old')
        items.append(Artifact(active, candidate, backup, True, True))
    items[1].candidate.write_text('unexpected')
    with pytest.raises(RuntimeError, match='Ambiguous'):
        rollback_artifacts(items)
    assert items[0].active.read_text() == 'new'
    assert items[0].backup.read_text() == 'old'


def test_interrupted_restore_can_resume(tmp_path, monkeypatch):
    import src.publication_recovery as recovery
    active, candidate, backup = [tmp_path / name for name in ('active', 'candidate', 'backup')]
    active.write_text('new')
    backup.write_text('old')
    item = Artifact(active, candidate, backup, True, True)
    rename = recovery.os.rename
    def interrupted(source, target):
        if source == backup:
            raise OSError('interrupted')
        rename(source, target)
    monkeypatch.setattr(recovery.os, 'rename', interrupted)
    with pytest.raises(OSError):
        rollback_artifacts([item])
    monkeypatch.setattr(recovery.os, 'rename', rename)
    rollback_artifacts([item])
    assert active.read_text() == 'old'
    assert candidate.read_text() == 'new'
