import json

import pytest

from src.lightrag_service import LightRAGService


@pytest.fixture
def service(tmp_path, monkeypatch):
    result = LightRAGService({'paths': {'data_dir': str(tmp_path)}}, workspace='kb')
    runtime = {'embedding': {'model': 'fixture', 'base_url': 'http://fixture.invalid',
                            'api_key': 'fixture', 'embed_dim': 8,
                            'embed_max_chars': 1000, 'embed_max_tokens': 256}}
    monkeypatch.setattr(result, '_runtime_models', lambda: runtime)
    return result, runtime


def test_sdk_embedding_limit_tracks_runtime_configuration(service):
    instance, runtime = service
    assert instance._make_embedding_func().max_token_size == 256
    runtime['embedding']['embed_max_tokens'] = 1024
    assert instance._make_embedding_func().max_token_size == 1024


@pytest.mark.parametrize('field', ['embed_max_chars', 'embed_max_tokens'])
def test_changed_input_limit_requires_rebuild_even_for_same_model(service, field):
    instance, runtime = service
    instance._save_manifest({'documents': {'source': {'indexed': True}}})
    instance.record_embedding_signature()
    assert instance.embedding_compatibility()['compatible']
    runtime['embedding'][field] += 1
    assert not instance.embedding_compatibility()['compatible']
    with pytest.raises(RuntimeError, match='输入限制'):
        instance.assert_embedding_compatible()


@pytest.mark.parametrize('legacy_metadata', [False, True])
def test_unknown_historical_input_limits_are_not_silently_claimed_compatible(service, legacy_metadata):
    instance, _ = service
    instance._save_manifest({'documents': {'source': {'indexed': True}}})
    if legacy_metadata:
        instance.record_embedding_signature()
        payload = json.loads(instance.embedding_meta_path.read_text(encoding='utf-8'))
        payload.pop('embed_max_chars')
        payload.pop('embed_max_tokens')
        instance.embedding_meta_path.write_text(json.dumps(payload), encoding='utf-8')
    before = instance.embedding_meta_path.read_bytes() if legacy_metadata else None
    assert not instance.embedding_compatibility(initialize_legacy=True)['compatible']
    after = instance.embedding_meta_path.read_bytes() if instance.embedding_meta_path.exists() else None
    assert before == after


def test_empty_workspace_can_initialize_new_input_signature(service):
    instance, _ = service
    assert instance.embedding_compatibility()['compatible']
    assert not instance.embedding_meta_path.exists()


@pytest.mark.parametrize('payload', [[], 'not metadata', None])
def test_invalid_metadata_shape_reports_incompatibility_without_crashing(service, payload):
    instance, _ = service
    instance.record_embedding_signature()
    instance.embedding_meta_path.write_text(json.dumps(payload), encoding='utf-8')
    result = instance.embedding_compatibility()
    assert not result['compatible']
    assert 'must be an object' in result['reason']


def test_truncation_never_inserts_replacement_characters_or_interprets_special_spellings(service, monkeypatch):
    from src import lightrag_service as module
    instance, _ = service
    class ByteTokenizer:
        def encode(self, text, *, disallowed_special):
            assert disallowed_special == ()
            return list(text.encode('utf-8'))

        def decode_bytes(self, ids):
            return bytes(ids)

    monkeypatch.setattr(module, '_get_embed_tokenizer', lambda: ByteTokenizer())
    assert instance._prepare_embedding_text('A🙂tail', 100, 3) == 'A'
    assert instance._prepare_embedding_text('<|endoftext|>tail', 100, 3) == '<|e'
    with pytest.raises(ValueError, match='empty text'):
        instance._prepare_embedding_text('🙂', 100, 1)
