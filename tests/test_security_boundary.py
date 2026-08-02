import asyncio
import json

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from src.api import server
from src import model_profiles
from src.llm_backend.siliconflow import SiliconFlowBackend


def _request(host: str, token: str = "") -> Request:
    headers = []
    if token:
        headers.append((b"x-app-token", token.encode("utf-8")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/health",
            "headers": headers,
            "client": (host, 12345),
            "server": ("127.0.0.1", 8101),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_remote_request_requires_configured_token(monkeypatch):
    monkeypatch.setattr(server, "APP_API_TOKEN", "expected-token")

    async def call_next(_request):
        return JSONResponse({"ok": True})

    denied = asyncio.run(
        server.require_remote_api_token(_request("192.168.1.10"), call_next)
    )
    allowed = asyncio.run(
        server.require_remote_api_token(
            _request("192.168.1.10", "expected-token"),
            call_next,
        )
    )

    assert denied.status_code == 403
    assert allowed.status_code == 200


def test_loopback_request_does_not_require_token(monkeypatch):
    monkeypatch.setattr(server, "APP_API_TOKEN", "")

    async def call_next(_request):
        return JSONResponse({"ok": True})

    response = asyncio.run(
        server.require_remote_api_token(_request("127.0.0.1"), call_next)
    )

    assert response.status_code == 200


def test_model_keys_are_encrypted_at_rest(tmp_path):
    config = {"paths": {"data_dir": str(tmp_path / "data")}}
    secret = "test-secret-value"

    model_profiles._save_keys({"profile": secret}, config)

    encrypted_path = model_profiles._keys_path(config)
    assert encrypted_path.exists()
    assert secret.encode("utf-8") not in encrypted_path.read_bytes()
    assert model_profiles._load_keys(config) == {"profile": secret}


def test_plaintext_model_key_store_is_migrated(tmp_path):
    config = {"paths": {"data_dir": str(tmp_path / "data")}}
    legacy_path = model_profiles._legacy_keys_path(config)
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        json.dumps({"profile": "legacy-secret"}),
        encoding="utf-8",
    )

    loaded = model_profiles._load_keys(config)

    assert loaded == {"profile": "legacy-secret"}
    assert model_profiles._keys_path(config).exists()
    assert not legacy_path.exists()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data",
        "http://0.0.0.0/v1",
        "http://192.168.1.20/v1",
        "https://user:pass@example.com/v1",
    ],
)
def test_model_api_base_rejects_unsafe_targets(url, monkeypatch):
    monkeypatch.delenv("LIGHTGRAPHRAG_ALLOW_PRIVATE_MODEL_HOSTS", raising=False)
    with pytest.raises(ValueError):
        model_profiles._normalize_api_base(url)


def test_model_api_base_allows_public_and_loopback_targets():
    assert (
        model_profiles._normalize_api_base("https://api.siliconflow.cn/v1/")
        == "https://api.siliconflow.cn/v1"
    )
    assert model_profiles._normalize_api_base("http://127.0.0.1:11434/v1") == (
        "http://127.0.0.1:11434/v1"
    )


def test_siliconflow_backend_omits_empty_authorization_header():
    backend = SiliconFlowBackend({"api_key": ""})

    assert "Authorization" not in backend._headers()
