"""Model provider profiles and per-purpose model bindings.

API keys are encrypted at rest under data/secrets. Windows uses the current
user's DPAPI protection; other platforms use a permission-restricted local
Fernet key. The public profile store only keeps non-sensitive metadata.
"""

from __future__ import annotations

import ctypes
import hashlib
import ipaddress
import json
import os
import tempfile
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from cryptography.fernet import Fernet

from src.config_loader import get_config


DEFAULT_PROFILE_ID = "siliconflow-default"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
_BLOCKED_MODEL_HOSTS = {
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.azure.internal",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_dir(config: dict[str, Any] | None = None) -> Path:
    cfg = config or get_config()
    return Path(cfg.get("paths", {}).get("data_dir", "./data"))


def _profiles_path(config: dict[str, Any] | None = None) -> Path:
    return _data_dir(config) / "model_profiles.json"


def _keys_path(config: dict[str, Any] | None = None) -> Path:
    return _data_dir(config) / "secrets" / "model_keys.enc"


def _legacy_keys_path(config: dict[str, Any] | None = None) -> Path:
    return _data_dir(config) / "secrets" / "model_keys.json"


def _fallback_key_path(config: dict[str, Any] | None = None) -> Path:
    return _data_dir(config) / "secrets" / ".model_keys.key"


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _dpapi_transform(data: bytes, *, protect: bool) -> bytes:
    buffer = ctypes.create_string_buffer(data)
    input_blob = _DataBlob(
        len(data),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)),
    )
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    if protect:
        ok = crypt32.CryptProtectData(
            ctypes.byref(input_blob),
            ctypes.c_wchar_p("LightGraphRAG model keys"),
            None,
            None,
            None,
            0x1,
            ctypes.byref(output_blob),
        )
    else:
        ok = crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            None,
            None,
            None,
            None,
            0x1,
            ctypes.byref(output_blob),
        )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(output_blob.pbData)


def _fallback_fernet(config: dict[str, Any] | None = None) -> Fernet:
    key_path = _fallback_key_path(config)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists():
        key = key_path.read_bytes().strip()
    else:
        key = Fernet.generate_key()
        temp_path = key_path.with_suffix(f".{os.getpid()}.tmp")
        temp_path.write_bytes(key)
        try:
            os.chmod(temp_path, 0o600)
        except OSError:
            pass
        temp_path.replace(key_path)
    return Fernet(key)


def _encrypt_keys(data: bytes, config: dict[str, Any] | None = None) -> bytes:
    if os.name == "nt":
        return b"DPAPI1" + _dpapi_transform(data, protect=True)
    return b"FERNET1" + _fallback_fernet(config).encrypt(data)


def _decrypt_keys(data: bytes, config: dict[str, Any] | None = None) -> bytes:
    if data.startswith(b"DPAPI1"):
        if os.name != "nt":
            raise RuntimeError("DPAPI-protected model keys can only be read on Windows")
        return _dpapi_transform(data[6:], protect=False)
    if data.startswith(b"FERNET1"):
        return _fallback_fernet(config).decrypt(data[7:])
    raise ValueError("Unknown model key storage format")


def _new_id(name: str, api_base: str) -> str:
    digest = hashlib.md5(f"{name}|{api_base}|{_now_iso()}".encode("utf-8")).hexdigest()[:12]
    return f"profile_{digest}"


def _key_preview(api_key: str) -> str:
    api_key = api_key.strip()
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:3]}****{api_key[-4:]}"


def _auth_headers(api_key: str, *, json_content: bool = False) -> dict[str, str]:
    headers = {"Content-Type": "application/json"} if json_content else {}
    key = (api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _normalize_api_base(api_base: str) -> str:
    value = str(api_base or "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API 地址必须是有效的 http:// 或 https:// 地址")
    if parsed.username or parsed.password:
        raise ValueError("API 地址不能包含用户名或密码")
    host = parsed.hostname.lower()
    if host in _BLOCKED_MODEL_HOSTS:
        raise ValueError("该 API 地址指向受保护的主机")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if address.is_link_local or address.is_multicast or address.is_unspecified:
            raise ValueError("该 API 地址指向不允许的网络地址")
        allow_private = os.environ.get("TDX_ALLOW_PRIVATE_MODEL_HOSTS", "") == "1"
        if address.is_private and not address.is_loopback and not allow_private:
            raise ValueError(
                "私有网络模型地址默认禁用；确认可信后设置 TDX_ALLOW_PRIVATE_MODEL_HOSTS=1"
            )
    return value


def _require_api_key(profile: dict[str, Any]) -> str:
    key = str(profile.get("api_key") or "").strip()
    if not key:
        raise ValueError("当前连接档案尚未保存 API Key，请在模型设置中编辑该连接并保存 API Key")
    return key


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else default
    except Exception:
        return default


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            Path(temp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _default_store(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or get_config()
    sf = cfg.get("siliconflow", {})
    profile = {
        "id": DEFAULT_PROFILE_ID,
        "name": "SiliconFlow 默认",
        "api_base": sf.get("base_url", DEFAULT_BASE_URL),
        "api_type": "openai_compatible",
        "models_cache": [],
        "last_used_at": "",
        "last_tested_at": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    return {
        "profiles": [profile],
        "bindings": {
            "chat": {
                "profile_id": DEFAULT_PROFILE_ID,
                "model": sf.get("chat_model", "Qwen/Qwen2.5-7B-Instruct"),
            },
            "embedding": {
                "profile_id": DEFAULT_PROFILE_ID,
                "model": sf.get("embed_model", "BAAI/bge-large-zh-v1.5"),
                "embed_dim": int(sf.get("embed_dim", 1024)),
                "embed_max_chars": int(sf.get("embed_max_chars", 700)),
            },
            "rerank": {
                "profile_id": DEFAULT_PROFILE_ID,
                "model": sf.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
                "enabled": True,
            },
        },
    }


def _load_store(config: dict[str, Any] | None = None) -> dict[str, Any]:
    path = _profiles_path(config)
    data = _load_json(path, _default_store(config))
    default = _default_store(config)
    if not isinstance(data.get("profiles"), list):
        data["profiles"] = default["profiles"]
    if not isinstance(data.get("bindings"), dict):
        data["bindings"] = default["bindings"]
    for purpose, binding in default["bindings"].items():
        data["bindings"].setdefault(purpose, binding)
    return data


def _save_store(data: dict[str, Any], config: dict[str, Any] | None = None) -> None:
    _save_json(_profiles_path(config), data)


def _load_keys(config: dict[str, Any] | None = None) -> dict[str, str]:
    encrypted_path = _keys_path(config)
    if encrypted_path.exists():
        try:
            payload = json.loads(
                _decrypt_keys(encrypted_path.read_bytes(), config).decode("utf-8")
            )
            if isinstance(payload, dict):
                return {
                    str(k): str(v)
                    for k, v in payload.items()
                    if isinstance(v, str)
                }
        except Exception as exc:
            raise RuntimeError(f"Failed to decrypt model API keys: {exc}") from exc

    legacy_path = _legacy_keys_path(config)
    keys = _load_json(legacy_path, {})
    clean = {str(k): str(v) for k, v in keys.items() if isinstance(v, str)}
    cfg = config or get_config()
    configured_default = str(cfg.get("siliconflow", {}).get("api_key") or "").strip()
    if configured_default and not clean.get(DEFAULT_PROFILE_ID):
        clean[DEFAULT_PROFILE_ID] = configured_default
    if clean:
        _save_keys(clean, config)
        if legacy_path.exists():
            legacy_path.unlink()
    return clean


def _save_keys(keys: dict[str, str], config: dict[str, Any] | None = None) -> None:
    path = _keys_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(keys, ensure_ascii=False).encode("utf-8")
    encrypted = _encrypt_keys(payload, config)
    temp_path = path.with_suffix(f".{os.getpid()}.tmp")
    temp_path.write_bytes(encrypted)
    try:
        os.chmod(temp_path, 0o600)
    except OSError:
        pass
    temp_path.replace(path)


def _public_profile(profile: dict[str, Any], keys: dict[str, str]) -> dict[str, Any]:
    api_key = keys.get(profile["id"], "")
    return {
        **profile,
        "has_api_key": bool(api_key),
        "api_key_preview": _key_preview(api_key),
    }


def list_profiles(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    store = _load_store(config)
    keys = _load_keys(config)
    return [_public_profile(profile, keys) for profile in store["profiles"]]


def get_bindings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return _load_store(config)["bindings"]


def save_bindings(bindings: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    store = _load_store(config)
    current = store["bindings"]
    for purpose in ("chat", "embedding", "rerank"):
        if purpose in bindings and isinstance(bindings[purpose], dict):
            current[purpose] = {**current.get(purpose, {}), **bindings[purpose]}
    store["bindings"] = current
    _save_store(store, config)
    return current


def upsert_profile(payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    store = _load_store(config)
    keys = _load_keys(config)
    profile_id = payload.get("id") or _new_id(payload.get("name", "Provider"), payload.get("api_base", ""))
    existing = next((item for item in store["profiles"] if item.get("id") == profile_id), None)
    now = _now_iso()
    profile = {
        **(existing or {}),
        "id": profile_id,
        "name": payload.get("name") or (existing or {}).get("name") or "模型连接",
        "api_base": _normalize_api_base(
            payload.get("api_base")
            or (existing or {}).get("api_base")
            or DEFAULT_BASE_URL
        ),
        "api_type": payload.get("api_type") or (existing or {}).get("api_type") or "openai_compatible",
        "models_cache": payload.get("models_cache", (existing or {}).get("models_cache", [])),
        "last_used_at": (existing or {}).get("last_used_at", ""),
        "last_tested_at": (existing or {}).get("last_tested_at", ""),
        "created_at": (existing or {}).get("created_at", now),
        "updated_at": now,
    }
    if existing is None:
        store["profiles"].insert(0, profile)
    else:
        store["profiles"] = [profile if item.get("id") == profile_id else item for item in store["profiles"]]
    api_key = payload.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        keys[profile_id] = api_key.strip()
        _save_keys(keys, config)
    _save_store(store, config)
    return _public_profile(profile, keys)


def delete_profile(profile_id: str, config: dict[str, Any] | None = None) -> dict[str, str]:
    store = _load_store(config)
    store["profiles"] = [item for item in store["profiles"] if item.get("id") != profile_id]
    for binding in store["bindings"].values():
        if isinstance(binding, dict) and binding.get("profile_id") == profile_id:
            binding["profile_id"] = DEFAULT_PROFILE_ID
    keys = _load_keys(config)
    keys.pop(profile_id, None)
    _save_store(store, config)
    _save_keys(keys, config)
    return {"deleted": profile_id}


def get_profile_with_key(profile_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    store = _load_store(config)
    profile = next((item for item in store["profiles"] if item.get("id") == profile_id), None)
    if profile is None:
        raise KeyError(profile_id)
    return {**profile, "api_key": _load_keys(config).get(profile_id, "")}


def get_runtime_model_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or get_config()
    store = _load_store(cfg)
    sf = cfg.get("siliconflow", {})

    def resolve(purpose: str) -> dict[str, Any]:
        binding = store["bindings"].get(purpose, {})
        profile_id = binding.get("profile_id", DEFAULT_PROFILE_ID)
        try:
            profile = get_profile_with_key(profile_id, cfg)
        except KeyError:
            keys = _load_keys(cfg)
            profile = {
                "api_base": sf.get("base_url", DEFAULT_BASE_URL),
                "api_key": keys.get(DEFAULT_PROFILE_ID, ""),
                "name": "Fallback",
            }
        return {**binding, **profile}

    chat = resolve("chat")
    embedding = resolve("embedding")
    rerank = resolve("rerank")
    return {
        "chat": {
            "base_url": chat.get("api_base", DEFAULT_BASE_URL),
            "api_key": chat.get("api_key", ""),
            "model": chat.get("model") or sf.get("chat_model", "Qwen/Qwen2.5-7B-Instruct"),
            "temperature": sf.get("chat_temperature", 0.7),
            "top_p": sf.get("chat_top_p", 0.9),
            "max_tokens": sf.get("chat_max_tokens", 4096),
            "frequency_penalty": sf.get("frequency_penalty", 0.3),
            "presence_penalty": sf.get("presence_penalty", 0.2),
            "timeout": sf.get("timeout", 90),
        },
        "embedding": {
            "base_url": embedding.get("api_base", DEFAULT_BASE_URL),
            "api_key": embedding.get("api_key", ""),
            "model": embedding.get("model") or sf.get("embed_model", "BAAI/bge-large-zh-v1.5"),
            "embed_dim": int(embedding.get("embed_dim") or sf.get("embed_dim", 1024)),
            "embed_max_chars": int(embedding.get("embed_max_chars") or sf.get("embed_max_chars", 700)),
        },
        "rerank": {
            "base_url": rerank.get("api_base", DEFAULT_BASE_URL),
            "api_key": rerank.get("api_key", ""),
            "model": rerank.get("model") or sf.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
            "enabled": bool(rerank.get("enabled", True)),
            "timeout": sf.get("timeout", 90),
        },
    }


async def discover_models(api_base: str, api_key: str, timeout: int = 30) -> list[dict[str, Any]]:
    normalized_base = _normalize_api_base(api_base)
    headers = _auth_headers(api_key)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{normalized_base}/models", headers=headers)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 401:
            if not (api_key or "").strip():
                raise ValueError("模型发现失败：该接口需要 API Key，请先填写并保存有效 API Key") from exc
            raise ValueError("模型发现失败：接口返回 401 Unauthorized，请确认 API Key 有效且有权限访问该 API 地址") from exc
        raise
    models = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(models, list):
        return []
    out = []
    for item in models:
        if isinstance(item, str):
            out.append({"id": item, "type": "unknown"})
        elif isinstance(item, dict) and item.get("id"):
            out.append({"id": item["id"], "type": item.get("type") or item.get("owned_by") or "unknown"})
    return out


async def test_chat(profile_id: str, model: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = get_profile_with_key(profile_id, config)
    headers = _auth_headers(_require_api_key(profile), json_content=True)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{profile['api_base'].rstrip('/')}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return {"ok": True, "model": model, "usage": data.get("usage", {})}


async def test_embedding(profile_id: str, model: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = get_profile_with_key(profile_id, config)
    headers = _auth_headers(_require_api_key(profile), json_content=True)
    payload = {"model": model, "input": ["你好"], "encoding_format": "float"}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{profile['api_base'].rstrip('/')}/embeddings", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    vec = (data.get("data") or [{}])[0].get("embedding") or []
    return {"ok": True, "model": model, "dimensions": len(vec), "preview": vec[:10]}


async def test_rerank(profile_id: str, model: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = get_profile_with_key(profile_id, config)
    headers = _auth_headers(_require_api_key(profile), json_content=True)
    payload = {"model": model, "query": "新闻资讯", "documents": ["客户端读取新闻资讯", "无关内容"], "top_n": 1}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{profile['api_base'].rstrip('/')}/rerank", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return {"ok": True, "model": model, "results": data.get("results", data.get("data", []))}
