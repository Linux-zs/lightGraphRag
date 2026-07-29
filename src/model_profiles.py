"""Model provider profiles and per-purpose model bindings.

API keys are stored under data/secrets, which is ignored by git. The public
profile store only keeps non-sensitive metadata and cached model ids.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.config_loader import get_config


DEFAULT_PROFILE_ID = "siliconflow-default"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_dir(config: dict[str, Any] | None = None) -> Path:
    cfg = config or get_config()
    return Path(cfg.get("paths", {}).get("data_dir", "./data"))


def _profiles_path(config: dict[str, Any] | None = None) -> Path:
    return _data_dir(config) / "model_profiles.json"


def _keys_path(config: dict[str, Any] | None = None) -> Path:
    return _data_dir(config) / "secrets" / "model_keys.json"


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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    data = _load_json(_keys_path(config), {})
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def _save_keys(keys: dict[str, str], config: dict[str, Any] | None = None) -> None:
    _save_json(_keys_path(config), keys)


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
        "api_base": (payload.get("api_base") or (existing or {}).get("api_base") or DEFAULT_BASE_URL).rstrip("/"),
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
            profile = {
                "api_base": sf.get("base_url", DEFAULT_BASE_URL),
                "api_key": sf.get("api_key", ""),
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
    normalized_base = api_base.strip().rstrip("/")
    if not normalized_base:
        raise ValueError("API 地址不能为空")
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
