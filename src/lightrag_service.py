"""LightRAG SDK adapter used by the FastAPI workbench and CLI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import networkx as nx
import re
from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import wrap_embedding_func_with_attrs
from loguru import logger

from src.config_loader import get_config
from src.doc_processor.parsers.base_parser import Document
from src.lightrag_stage_timing import install_stage_timing
from src.model_profiles import get_runtime_model_config


DEFAULT_WORKSPACE = "tdx_default"
DEFAULT_MODE = "mix"
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WORKSPACE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _atomic_write_json(path: Path, data: Any) -> None:
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

GRAPH_CATEGORY_MAP = {
    "organization": "服务层",
    "company": "服务层",
    "person": "服务层",
    "location": "服务层",
    "artifact": "内容",
    "data": "内容",
    "event": "传输",
    "entity": "核心系统",
    "technology": "传输层",
    "system": "核心系统",
    "component": "核心系统",
    "process": "传输",
}

_EMBED_TOKENIZER = None


def _get_embed_tokenizer():
    """Lazily load a tiktoken encoding for embedding token truncation.

    Returns ``None`` if tiktoken is unavailable or the encoding cannot be loaded,
    in which case the caller falls back to the character cap.
    """
    global _EMBED_TOKENIZER
    if _EMBED_TOKENIZER is None:
        try:
            import tiktoken

            _EMBED_TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _EMBED_TOKENIZER = False
    return _EMBED_TOKENIZER or None

TDX_GRAPH_RULE_TEMPLATE = {
    "id": "tdx_ops",
    "name": "通达信运维知识库",
    "description": "适合通达信部署、运维、同步链路、服务排查类文档。",
    "entity_types": ["产品", "模块", "服务", "配置项", "故障现象", "排查步骤", "文件", "数据库"],
    "relation_types": ["依赖于", "部署在", "读取", "写入", "同步到", "导致", "排查", "包含"],
    "aliases_text": "",
    "extraction_prompt": (
        "请从通达信运维/部署文档中抽取稳定、可复用的业务实体和技术实体。"
        "优先抽取系统、模块、服务、配置项、文件、数据库、故障现象和排查动作；"
        "关系应表达真实依赖、数据流向、部署位置、读写关系、故障原因和排查路径。"
        "不要把普通段落标题、孤立编号、无意义变量值抽成实体。"
    ),
    "built_in": True,
    "created_at": "",
    "updated_at": "",
}

GENERAL_GRAPH_RULE_TEMPLATE = {
    "id": "general_knowledge",
    "name": "通用知识库",
    "description": "适合非特定行业文档，保守抽取人物、组织、地点、概念、事件、物品和因果关系。",
    "entity_types": ["人物", "组织", "地点", "概念", "事件", "物品", "问题", "结论"],
    "relation_types": ["属于", "包含", "影响", "导致", "关联", "发生在", "参与", "说明"],
    "aliases_text": "",
    "extraction_prompt": (
        "请从当前文档中抽取对理解内容有帮助的稳定实体和关系。"
        "实体应是文档主题中的人物、组织、地点、概念、事件、物品、问题或结论；"
        "关系应表达包含、影响、导致、关联、时间地点、参与者或解释说明。"
        "不要因为模板中出现某个类型就强行抽取；不要把普通句子、孤立编号、页眉页脚或无意义短语抽成实体。"
    ),
    "built_in": True,
    "created_at": "",
    "updated_at": "",
}

SUPPLY_CHAIN_GRAPH_RULE_TEMPLATE = {
    "id": "supply_chain",
    "name": "供应链/风险分析",
    "description": "适合供应链、药品、库存、生产、采购、断供风险类资料。",
    "entity_types": ["机构", "产品", "原料", "供应商", "生产环节", "库存", "风险", "原因", "措施"],
    "relation_types": ["采购", "供应", "生产", "储备", "依赖", "导致", "缓解", "影响"],
    "aliases_text": "",
    "extraction_prompt": (
        "请围绕供应链和风险分析抽取实体关系。优先抽取机构、产品、原料、供应商、生产环节、库存状态、风险因素和缓解措施；"
        "关系应体现采购、供应、生产、储备、依赖、导致、缓解和影响。"
        "不要把普通段落标题或没有业务含义的编号抽成实体。"
    ),
    "built_in": True,
    "created_at": "",
    "updated_at": "",
}

BUILTIN_GRAPH_RULE_TEMPLATES = [
    TDX_GRAPH_RULE_TEMPLATE,
    GENERAL_GRAPH_RULE_TEMPLATE,
    SUPPLY_CHAIN_GRAPH_RULE_TEMPLATE,
]

GRAPH_EXTRACTION_MODES = {"assist", "enhanced", "strict"}


@dataclass
class LightRAGQueryResult:
    content: str
    raw_data: dict[str, Any]
    citations: list[dict[str, Any]]


@dataclass
class LightRAGStreamResult:
    iterator: AsyncIterator[str]
    raw_data: dict[str, Any]
    citations: list[dict[str, Any]]


@dataclass
class LightRAGDocStatus:
    doc_id: str
    status: str
    chunk_count: int = 0
    chunks_list: list[str] | None = None
    error_msg: str = ""


def sanitize_workspace(workspace: str | None) -> str:
    value = (workspace or DEFAULT_WORKSPACE).strip()
    if not WORKSPACE_RE.fullmatch(value):
        raise ValueError("Workspace must be 1-64 chars: letters, numbers, underscore or hyphen")
    return value


def stable_doc_id(doc: Document) -> str:
    """Create a stable LightRAG doc id from the uploaded file identity."""
    explicit = str(doc.metadata.get("lightrag_doc_id") or "").strip()
    if explicit:
        return explicit
    digest = hashlib.md5(
        f"{Path(doc.file_path).as_posix()}|{doc.file_name}".encode("utf-8")
    ).hexdigest()[:16]
    return f"doc_{digest}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _basename(path_or_name: str) -> str:
    return Path(path_or_name).name if path_or_name else ""


def _dir_size(path: str | Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(str(path)):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                continue
    return total


class LightRAGService:
    """Single runtime entry point for indexing, querying and document status."""

    def __init__(self, config: dict | None = None, workspace: str | None = None) -> None:
        self.config = config or get_config()
        paths = self.config.setdefault("paths", {})
        data_dir = Path(paths.get("data_dir", "./data"))
        self.data_dir = data_dir
        self.working_dir = Path(paths.get("lightrag_dir", data_dir / "lightrag"))
        default_workspace = self.config.get("lightrag", {}).get("workspace", DEFAULT_WORKSPACE)
        self.workspace = sanitize_workspace(workspace or default_workspace)
        if self.workspace == default_workspace:
            self.manifest_path = Path(paths.get("lightrag_manifest", data_dir / "lightrag_manifest.json"))
        else:
            self.manifest_path = data_dir / "lightrag_manifests" / f"{self.workspace}.json"
        self._rag: LightRAG | None = None
        self._init_lock = asyncio.Lock()
        self._last_stage_timings: dict[str, float] = {}

    @property
    def rag(self) -> LightRAG:
        if self._rag is None:
            raise RuntimeError("LightRAG is not initialized")
        return self._rag

    @property
    def workspace_dir(self) -> Path:
        return self.working_dir / self.workspace

    @property
    def graphml_path(self) -> Path:
        return self.workspace_dir / "graph_chunk_entity_relation.graphml"

    @property
    def graph_governance_path(self) -> Path:
        return self.data_dir / "graph_governance" / f"{self.workspace}.json"

    @property
    def graph_reference_dir(self) -> Path:
        return self.data_dir / "graph_governance_refs" / self.workspace

    @property
    def graph_rule_templates_path(self) -> Path:
        return self.data_dir / "graph_rule_templates.json"

    @property
    def embedding_meta_path(self) -> Path:
        return self.data_dir / "lightrag_embedding_meta" / f"{self.workspace}.json"

    def current_embedding_signature(
        self,
        embedding_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        embedding = embedding_config or self._runtime_models()["embedding"]
        return {
            "base_url": str(embedding.get("base_url") or "").rstrip("/"),
            "model": str(embedding.get("model") or ""),
            "embed_dim": int(embedding.get("embed_dim") or 0),
        }

    def record_embedding_signature(
        self,
        embedding_config: dict[str, Any] | None = None,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if self.embedding_meta_path.exists() and not overwrite:
            try:
                return json.loads(self.embedding_meta_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Replacing unreadable embedding metadata: {}", self.embedding_meta_path)
        payload = {
            **self.current_embedding_signature(embedding_config),
            "workspace": self.workspace,
            "updated_at": _now_iso(),
        }
        _atomic_write_json(self.embedding_meta_path, payload)
        return payload

    def embedding_compatibility(
        self,
        embedding_config: dict[str, Any] | None = None,
        *,
        initialize_legacy: bool = True,
    ) -> dict[str, Any]:
        current = self.current_embedding_signature(embedding_config)
        manifest = self._load_manifest()
        indexed = any(bool(item.get("indexed")) for item in manifest.get("documents", {}).values())
        stored = None
        if self.embedding_meta_path.exists():
            try:
                stored = json.loads(self.embedding_meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {
                    "compatible": False,
                    "reason": f"Embedding metadata is unreadable: {exc}",
                    "current": current,
                    "stored": None,
                }
        elif indexed and initialize_legacy:
            stored = self.record_embedding_signature(current)

        compatible = not indexed or stored is None or all(
            stored.get(key) == current.get(key)
            for key in ("base_url", "model", "embed_dim")
        )
        return {
            "compatible": compatible,
            "reason": "" if compatible else "Embedding model configuration differs from the existing index",
            "current": current,
            "stored": stored,
            "indexed_documents": sum(
                1 for item in manifest.get("documents", {}).values() if item.get("indexed")
            ),
        }

    def assert_embedding_compatible(self) -> None:
        compatibility = self.embedding_compatibility()
        if not compatibility["compatible"]:
            stored = compatibility.get("stored") or {}
            current = compatibility.get("current") or {}
            raise RuntimeError(
                "当前嵌入模型与该知识库已有索引不兼容，请先重建索引。"
                f" 已有: {stored.get('model')} / {stored.get('embed_dim')}，"
                f"当前: {current.get('model')} / {current.get('embed_dim')}"
            )

    async def get_rag(self) -> LightRAG:
        if self._rag is not None:
            return self._rag
        async with self._init_lock:
            if self._rag is None:
                self.working_dir.mkdir(parents=True, exist_ok=True)
                self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
                lightrag_cfg = self.config.get("lightrag", {})
                self._rag = LightRAG(
                    working_dir=str(self.working_dir),
                    workspace=self.workspace,
                    chunk_token_size=self.config.get("chunking", {}).get("chunk_size", 512),
                    chunk_overlap_token_size=self.config.get("chunking", {}).get("chunk_overlap", 50),
                    embedding_func=self._make_embedding_func(),
                    llm_model_func=self._make_llm_func(),
                    llm_model_name=self._runtime_models()["chat"]["model"],
                    llm_model_kwargs=self._llm_kwargs(),
                    entity_extraction_use_json=lightrag_cfg.get("entity_extraction_use_json", True),
                    entity_extract_max_gleaning=lightrag_cfg.get("entity_extract_max_gleaning", 1),
                    llm_model_max_async=lightrag_cfg.get("llm_model_max_async", 4),
                    rerank_model_func=self._make_rerank_func(),
                    addon_params={
                        "language": "Chinese",
                        "chunker": {
                            "recursive_character": {
                                "separators": self.config.get("chunking", {}).get("separators", [])
                            }
                        },
                    },
                )
                await self._rag.initialize_storages()
                logger.info("LightRAG initialized at {}", self.working_dir)
        return self._rag

    def ensure_workspace(self) -> dict[str, Any]:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._save_manifest({"documents": {}})
        return {
            "workspace": self.workspace,
            "manifest_path": str(self.manifest_path),
            "workspace_path": str(self.workspace_dir),
        }

    def _make_embedding_func(self):
        embed = self._runtime_models()["embedding"]
        embed_model = embed["model"]
        embed_dim = int(embed["embed_dim"])
        embed_max_chars = int(embed["embed_max_chars"])
        embed_max_tokens = int(embed.get("embed_max_tokens") or 0)
        base_url = embed["base_url"]
        api_key = embed["api_key"]

        @wrap_embedding_func_with_attrs(
            embedding_dim=embed_dim,
            max_token_size=480,
            model_name=embed_model,
        )
        async def siliconflow_embed(texts: list[str]):
            safe_texts = [
                self._prepare_embedding_text(text, embed_max_chars, embed_max_tokens)
                for text in texts
            ]
            return await openai_embed.func(
                safe_texts,
                model=embed_model,
                base_url=base_url,
                api_key=api_key,
            )

        return siliconflow_embed

    def _prepare_embedding_text(
        self,
        text: str,
        max_chars: int,
        max_tokens: int | None = None,
    ) -> str:
        """Normalize and truncate text before embedding.

        Truncates by tokens first (when a tokenizer is available) because the
        embedding model has a hard *token* limit, not a character limit. A
        character-based cap alone is unsafe for Chinese text, where each character
        can map to ~1 token: e.g. SiliconFlow's ``BAAI/bge-large-zh-v1.5`` rejects
        inputs above 512 tokens, so a 700-character Chinese chunk is refused with
        HTTP 400 / error code 20015 ("The parameter is invalid"), which aborts the
        whole indexing flush. The character cap stays as a cheap fallback when no
        tokenizer can be loaded.
        """
        safe = CONTROL_CHARS_RE.sub(" ", text or "")
        safe = safe.strip()
        if not safe:
            safe = " "
        if max_tokens and max_tokens > 0:
            try:
                enc = _get_embed_tokenizer()
                if enc is not None:
                    ids = enc.encode(safe)
                    if len(ids) > max_tokens:
                        safe = enc.decode(ids[:max_tokens]).strip()
            except Exception:
                logger.debug("Token-based embedding truncation unavailable; using char cap")
        if len(safe) > max_chars:
            safe = safe[:max_chars]
        return safe

    @staticmethod
    def _thinking_extra_body(model: str) -> dict[str, Any]:
        """Disable hidden chain-of-thought for hybrid thinking models.

        Models like Qwen3-8B / GLM-4.5 / DeepSeek-V3.1 default to thinking mode
        on SiliconFlow, which makes each entity-extraction call take 60-300s and
        often yields empty JSON. `enable_thinking: false` turns it off.
        Pure instruct/coder variants do not accept this parameter, so skip them.
        """
        m = (model or "").lower()
        if "instruct" in m or "coder" in m:
            return {}
        hybrid_markers = ("qwen3", "glm-4.5", "glm-4.6", "glm-5", "deepseek-v3.1", "deepseek-v3.2", "hunyuan-a13b")
        if any(marker in m for marker in hybrid_markers):
            return {"enable_thinking": False}
        return {}

    def _llm_kwargs(self) -> dict[str, Any]:
        chat = self._runtime_models()["chat"]
        kwargs: dict[str, Any] = {
            "temperature": chat.get("temperature", self.config.get("llm", {}).get("temperature", 0.7)),
            "top_p": chat.get("top_p", self.config.get("llm", {}).get("top_p", 0.9)),
            "max_tokens": chat.get("max_tokens", self.config.get("llm", {}).get("max_tokens", 4096)),
            "frequency_penalty": chat.get("frequency_penalty", 0.3),
            "presence_penalty": chat.get("presence_penalty", 0.2),
        }
        disable_thinking = bool(self.config.get("lightrag", {}).get("disable_thinking", True))
        if disable_thinking:
            extra_body = self._thinking_extra_body(chat.get("model", ""))
            if extra_body:
                kwargs["extra_body"] = extra_body
        return kwargs

    def _make_llm_func(self):
        chat = self._runtime_models()["chat"]
        model = chat["model"]
        base_url = chat["base_url"]
        api_key = chat["api_key"]
        timeout = chat.get("timeout", 30)

        async def siliconflow_complete(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            keyword_extraction: bool = False,
            **kwargs: Any,
        ):
            return await openai_complete_if_cache(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                keyword_extraction=keyword_extraction,
                **kwargs,
            )

        return siliconflow_complete

    def _make_rerank_func(self):
        rerank = self._runtime_models()["rerank"]
        model = rerank["model"]
        base_url = rerank["base_url"].rstrip("/")
        api_key = rerank["api_key"]
        timeout = rerank.get("timeout", 30)
        enabled = bool(rerank.get("enabled", True))

        async def siliconflow_rerank(query: str, documents: list[str], top_n: int | None = None, **_: Any):
            if not enabled or not api_key or not documents:
                return []
            payload = {
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": top_n or len(documents),
                "return_documents": False,
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(f"{base_url}/rerank", json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                results = data.get("results") or data.get("data") or []
                normalized = []
                for item in results:
                    if "index" not in item:
                        continue
                    normalized.append({
                        "index": item["index"],
                        "relevance_score": item.get(
                            "relevance_score",
                            item.get("score", item.get("similarity", 0.0)),
                        ),
                    })
                return normalized
            except Exception as exc:
                logger.warning("SiliconFlow rerank failed, continuing without rerank: {}", exc)
                return []

        return siliconflow_rerank

    def _runtime_models(self) -> dict[str, Any]:
        return get_runtime_model_config(self.config)

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"documents": {}}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("documents"), dict):
                return data
        except Exception:
            logger.exception("Failed to read LightRAG manifest")
        return {"documents": {}}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        _atomic_write_json(self.manifest_path, manifest)

    def _manifest_doc_id(self, doc: Document, manifest: dict[str, Any]) -> str:
        explicit = str(doc.metadata.get("lightrag_doc_id") or "").strip()
        if explicit:
            return explicit
        for doc_id, item in manifest.get("documents", {}).items():
            if isinstance(item, dict) and item.get("doc_name") == doc.file_name:
                doc.metadata["lightrag_doc_id"] = doc_id
                return doc_id
        return stable_doc_id(doc)

    def _default_graph_governance(self) -> dict[str, Any]:
        template = TDX_GRAPH_RULE_TEMPLATE if self.workspace == DEFAULT_WORKSPACE else GENERAL_GRAPH_RULE_TEMPLATE
        cfg = {
            "workspace": self.workspace,
            "rule_template_id": template["id"],
            "rule_template_name": template["name"],
            "extraction_mode": "assist",
            "allow_other_entity_type": True,
            "entity_types": list(template["entity_types"]),
            "relation_types": list(template["relation_types"]),
            "aliases_text": template["aliases_text"],
            "extraction_prompt": template["extraction_prompt"],
            "reference_files": [],
            "updated_at": _now_iso(),
            "audit_log": [],
        }
        cfg["effective_extraction_prompt"] = self.graph_extraction_guidance(config=cfg)
        return cfg

    def load_graph_governance(self) -> dict[str, Any]:
        default = self._default_graph_governance()
        path = self.graph_governance_path
        if not path.exists():
            return default
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return default
        except Exception:
            logger.exception("Failed to read graph governance config")
            return default
        merged = {**default, **data}
        merged["workspace"] = self.workspace
        merged["rule_template_id"] = str(merged.get("rule_template_id") or default["rule_template_id"])
        merged["rule_template_name"] = str(merged.get("rule_template_name") or default["rule_template_name"])
        merged["extraction_mode"] = str(merged.get("extraction_mode") or default["extraction_mode"])
        if merged["extraction_mode"] not in GRAPH_EXTRACTION_MODES:
            merged["extraction_mode"] = default["extraction_mode"]
        merged["allow_other_entity_type"] = bool(merged.get("allow_other_entity_type", True))
        merged["entity_types"] = list(merged.get("entity_types") or [])
        merged["relation_types"] = list(merged.get("relation_types") or [])
        merged["reference_files"] = list(merged.get("reference_files") or [])
        merged["audit_log"] = list(merged.get("audit_log") or [])
        merged["effective_extraction_prompt"] = self.graph_extraction_guidance(config=merged)
        return merged

    def save_graph_governance(self, config: dict[str, Any]) -> dict[str, Any]:
        current = self.load_graph_governance()
        allowed = {
            "entity_types",
            "relation_types",
            "aliases_text",
            "extraction_prompt",
            "rule_template_id",
            "rule_template_name",
            "extraction_mode",
            "allow_other_entity_type",
            "reference_files",
            "audit_log",
        }
        for key in allowed:
            if key in config:
                current[key] = config[key]
        current["workspace"] = self.workspace
        current["updated_at"] = _now_iso()
        to_save = {k: v for k, v in current.items() if k != "effective_extraction_prompt"}
        _atomic_write_json(self.graph_governance_path, to_save)
        return self.load_graph_governance()

    def _load_custom_graph_rule_templates(self) -> list[dict[str, Any]]:
        if not self.graph_rule_templates_path.exists():
            return []
        try:
            data = json.loads(self.graph_rule_templates_path.read_text(encoding="utf-8"))
            templates = data.get("templates") if isinstance(data, dict) else data
            if isinstance(templates, list):
                return [item for item in templates if isinstance(item, dict)]
        except Exception:
            logger.exception("Failed to read graph rule templates")
        return []

    def _save_custom_graph_rule_templates(self, templates: list[dict[str, Any]]) -> None:
        _atomic_write_json(self.graph_rule_templates_path, {"templates": templates})

    def list_graph_rule_templates(self) -> list[dict[str, Any]]:
        custom = self._load_custom_graph_rule_templates()
        builtin_ids = {item["id"] for item in BUILTIN_GRAPH_RULE_TEMPLATES}
        clean_custom = [
            {**item, "built_in": False}
            for item in custom
            if item.get("id") and item.get("id") not in builtin_ids
        ]
        return [dict(item) for item in BUILTIN_GRAPH_RULE_TEMPLATES] + clean_custom

    def save_graph_rule_template(self, template: dict[str, Any]) -> dict[str, Any]:
        template_id = str(template.get("id") or "").strip()
        if not template_id:
            template_id = "rule_" + hashlib.md5(
                f"{template.get('name', 'rule')}|{_now_iso()}".encode("utf-8")
            ).hexdigest()[:12]
        if template_id in {item["id"] for item in BUILTIN_GRAPH_RULE_TEMPLATES}:
            raise ValueError("Built-in graph rule templates cannot be overwritten")
        now = _now_iso()
        custom = self._load_custom_graph_rule_templates()
        existing = next((item for item in custom if item.get("id") == template_id), {})
        saved = {
            **existing,
            "id": template_id,
            "name": str(template.get("name") or existing.get("name") or "自定义抽取规则").strip(),
            "description": str(template.get("description") or existing.get("description") or "").strip(),
            "entity_types": [str(item).strip() for item in template.get("entity_types", existing.get("entity_types", [])) if str(item).strip()],
            "relation_types": [str(item).strip() for item in template.get("relation_types", existing.get("relation_types", [])) if str(item).strip()],
            "aliases_text": str(template.get("aliases_text", existing.get("aliases_text", ""))),
            "extraction_prompt": str(template.get("extraction_prompt", existing.get("extraction_prompt", ""))),
            "built_in": False,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        custom = [saved if item.get("id") == template_id else item for item in custom]
        if not any(item.get("id") == template_id for item in custom):
            custom.insert(0, saved)
        self._save_custom_graph_rule_templates(custom)
        return saved

    def delete_graph_rule_template(self, template_id: str) -> dict[str, Any]:
        if template_id in {item["id"] for item in BUILTIN_GRAPH_RULE_TEMPLATES}:
            raise ValueError("Built-in graph rule templates cannot be deleted")
        custom = self._load_custom_graph_rule_templates()
        next_custom = [item for item in custom if item.get("id") != template_id]
        if len(next_custom) == len(custom):
            raise KeyError(template_id)
        self._save_custom_graph_rule_templates(next_custom)
        return {"deleted": template_id}

    def apply_graph_rule_template(self, template_id: str) -> dict[str, Any]:
        template = next((item for item in self.list_graph_rule_templates() if item.get("id") == template_id), None)
        if template is None:
            raise KeyError(template_id)
        cfg = self.save_graph_governance(
            {
                "rule_template_id": template["id"],
                "rule_template_name": template["name"],
                "extraction_mode": "assist",
                "allow_other_entity_type": True,
                "entity_types": list(template.get("entity_types") or []),
                "relation_types": list(template.get("relation_types") or []),
                "aliases_text": str(template.get("aliases_text") or ""),
                "extraction_prompt": str(template.get("extraction_prompt") or ""),
            }
        )
        self.append_graph_audit("apply_rule_template", {"template_id": template_id, "template_name": template["name"]})
        return self.load_graph_governance()

    def graph_governance_summary(self) -> dict[str, Any]:
        cfg = self.load_graph_governance()
        prompt = str(cfg.get("extraction_prompt") or "")
        return {
            "rule_template_id": cfg.get("rule_template_id", ""),
            "rule_template_name": cfg.get("rule_template_name", ""),
            "extraction_mode": cfg.get("extraction_mode", "assist"),
            "allow_other_entity_type": bool(cfg.get("allow_other_entity_type", True)),
            "entity_type_count": len(cfg.get("entity_types") or []),
            "relation_type_count": len(cfg.get("relation_types") or []),
            "extraction_prompt_preview": prompt[:180] + ("..." if len(prompt) > 180 else ""),
            "effective_extraction_prompt_preview": str(cfg.get("effective_extraction_prompt") or "")[:240],
            "updated_at": cfg.get("updated_at", ""),
        }

    def add_graph_reference(self, file_name: str, content: str) -> dict[str, Any]:
        ref_id = hashlib.md5(f"{file_name}|{_now_iso()}".encode("utf-8")).hexdigest()[:12]
        safe_name = Path(file_name).name
        self.graph_reference_dir.mkdir(parents=True, exist_ok=True)
        path = self.graph_reference_dir / f"{ref_id}_{safe_name}"
        path.write_text(content, encoding="utf-8")
        item = {
            "id": ref_id,
            "file_name": safe_name,
            "path": str(path),
            "char_count": len(content),
            "created_at": _now_iso(),
        }
        cfg = self.load_graph_governance()
        cfg["reference_files"] = [item, *list(cfg.get("reference_files") or [])]
        self.save_graph_governance(cfg)
        return item

    def delete_graph_reference(self, ref_id: str) -> dict[str, Any]:
        cfg = self.load_graph_governance()
        refs = list(cfg.get("reference_files") or [])
        match = next((item for item in refs if item.get("id") == ref_id), None)
        if match is None:
            raise KeyError(ref_id)
        path = Path(str(match.get("path") or ""))
        try:
            if path.exists() and path.resolve().is_relative_to(self.graph_reference_dir.resolve()):
                path.unlink()
        except Exception as exc:
            logger.warning("Failed to delete graph reference file {}: {}", ref_id, exc)
        cfg["reference_files"] = [item for item in refs if item.get("id") != ref_id]
        self.save_graph_governance(cfg)
        return {"deleted": ref_id}

    def graph_reference_bundle(
        self,
        *,
        max_chars: int = 12000,
        config: dict[str, Any] | None = None,
    ) -> str:
        cfg = config or self.load_graph_governance()
        parts = []
        if cfg.get("aliases_text"):
            parts.append("术语/别名表:\n" + str(cfg.get("aliases_text")))
        for item in cfg.get("reference_files") or []:
            path = Path(str(item.get("path") or ""))
            try:
                if path.exists():
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    parts.append(f"参考文件 {item.get('file_name')}:\n{text}")
            except Exception:
                logger.warning("Failed to read graph reference file {}", item.get("path"))
        bundle = "\n\n".join(parts)
        return bundle[:max_chars].rstrip()

    def graph_extraction_guidance(
        self,
        *,
        max_reference_chars: int = 8000,
        config: dict[str, Any] | None = None,
    ) -> str:
        cfg = config or self.load_graph_governance()
        entity_types = [str(item).strip() for item in cfg.get("entity_types") or [] if str(item).strip()]
        relation_types = [str(item).strip() for item in cfg.get("relation_types") or [] if str(item).strip()]
        refs = self.graph_reference_bundle(max_chars=max_reference_chars, config=cfg)
        mode = str(cfg.get("extraction_mode") or "assist")
        allow_other = bool(cfg.get("allow_other_entity_type", True))
        blocks = [
            (
                "Core graph extraction policy:\n"
                "- First build a useful knowledge graph from the current input text itself.\n"
                "- Extract stable, meaningful entities and real relationships that are explicit or strongly implied by the text.\n"
                "- Project-specific rules below are guidance for classification, normalization and prioritization; they are not the source of truth.\n"
                "- Do not skip an important entity or relationship only because it does not match the configured domain vocabulary, unless strict whitelist mode is enabled.\n"
                "- Do not extract meaningless section numbers, page headers, isolated variables, boilerplate, or generic fragments as entities."
            )
        ]
        if mode == "strict":
            blocks.append(
                "Extraction mode: strict whitelist. Use configured entity and relation types as hard constraints. "
                "If an entity cannot be classified into the configured entity types, skip it unless it is essential to connect two valid entities."
            )
        elif mode == "enhanced":
            blocks.append(
                "Extraction mode: domain enhanced. Prefer configured entity and relation types, but still extract important text-specific entities and relations. "
                "Configured rules should improve precision, not suppress the graph."
            )
        else:
            blocks.append(
                "Extraction mode: assist. Treat configured entity and relation types only as hints. "
                "General LightRAG extraction should remain active for any document domain."
            )
        if allow_other:
            blocks.append("Entity type fallback: if no configured entity type fits, classify the entity as `Other` instead of dropping it.")
        else:
            blocks.append("Entity type fallback: do not use `Other`; choose the closest configured type, or skip only genuinely low-value entities.")
        if entity_types:
            blocks.append("Preferred entity types:\n" + "\n".join(f"- {item}" for item in entity_types))
        if relation_types:
            blocks.append(
                "Preferred relationship categories:\n"
                + "\n".join(f"- {item}" for item in relation_types)
            )
        if cfg.get("extraction_prompt"):
            blocks.append("Additional project extraction guidance:\n" + str(cfg.get("extraction_prompt")))
        if refs:
            blocks.append(
                "Project terminology and reference material. Use this only to normalize names, aliases, entity types, and relationship meaning. "
                "Do not extract entities solely because they appear in this reference material unless they are relevant to the current input text.\n"
                + refs
            )
        return "\n\n".join(blocks).strip()

    def _jsonable(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return str(value)

    def append_graph_audit(self, action: str, payload: dict[str, Any], result: Any = None) -> dict[str, Any]:
        cfg = self.load_graph_governance()
        entry = {
            "id": hashlib.md5(f"{action}|{_now_iso()}".encode("utf-8")).hexdigest()[:12],
            "action": action,
            "payload": self._jsonable(payload),
            "result": self._jsonable(result),
            "created_at": _now_iso(),
        }
        cfg["audit_log"] = [entry, *list(cfg.get("audit_log") or [])][:200]
        self.save_graph_governance(cfg)
        return entry

    async def create_graph_entity(self, entity_name: str, entity_data: dict[str, Any]) -> dict[str, Any]:
        rag = await self.get_rag()
        result = await rag.acreate_entity(entity_name=entity_name, entity_data=entity_data)
        self.append_graph_audit("create_entity", {"entity_name": entity_name, "entity_data": entity_data}, result)
        return self._jsonable(result)

    async def edit_graph_entity(
        self,
        entity_name: str,
        updated_data: dict[str, Any],
        *,
        allow_rename: bool = True,
        allow_merge: bool = False,
    ) -> dict[str, Any]:
        rag = await self.get_rag()
        result = await rag.aedit_entity(
            entity_name=entity_name,
            updated_data=updated_data,
            allow_rename=allow_rename,
            allow_merge=allow_merge,
        )
        self.append_graph_audit(
            "edit_entity",
            {
                "entity_name": entity_name,
                "updated_data": updated_data,
                "allow_rename": allow_rename,
                "allow_merge": allow_merge,
            },
            result,
        )
        return self._jsonable(result)

    async def delete_graph_entity(self, entity_name: str) -> dict[str, Any]:
        rag = await self.get_rag()
        result = await rag.adelete_by_entity(entity_name)
        self.append_graph_audit("delete_entity", {"entity_name": entity_name}, result)
        return self._jsonable(result)

    async def create_graph_relation(
        self,
        source_entity: str,
        target_entity: str,
        relation_data: dict[str, Any],
    ) -> dict[str, Any]:
        rag = await self.get_rag()
        result = await rag.acreate_relation(
            source_entity=source_entity,
            target_entity=target_entity,
            relation_data=relation_data,
        )
        self.append_graph_audit(
            "create_relation",
            {"source_entity": source_entity, "target_entity": target_entity, "relation_data": relation_data},
            result,
        )
        return self._jsonable(result)

    async def edit_graph_relation(
        self,
        source_entity: str,
        target_entity: str,
        updated_data: dict[str, Any],
    ) -> dict[str, Any]:
        rag = await self.get_rag()
        result = await rag.aedit_relation(
            source_entity=source_entity,
            target_entity=target_entity,
            updated_data=updated_data,
        )
        self.append_graph_audit(
            "edit_relation",
            {"source_entity": source_entity, "target_entity": target_entity, "updated_data": updated_data},
            result,
        )
        return self._jsonable(result)

    async def delete_graph_relation(self, source_entity: str, target_entity: str) -> dict[str, Any]:
        rag = await self.get_rag()
        result = await rag.adelete_by_relation(source_entity, target_entity)
        self.append_graph_audit(
            "delete_relation",
            {"source_entity": source_entity, "target_entity": target_entity},
            result,
        )
        return self._jsonable(result)

    async def merge_graph_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        target_entity_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rag = await self.get_rag()
        result = await rag.amerge_entities(
            source_entities=source_entities,
            target_entity=target_entity,
            target_entity_data=target_entity_data,
        )
        self.append_graph_audit(
            "merge_entities",
            {
                "source_entities": source_entities,
                "target_entity": target_entity,
                "target_entity_data": target_entity_data or {},
            },
            result,
        )
        return self._jsonable(result)

    def register_upload(self, doc: Document) -> dict[str, Any]:
        manifest = self._load_manifest()
        doc_id = self._manifest_doc_id(doc, manifest)
        doc.metadata["lightrag_doc_id"] = doc_id
        existing = manifest["documents"].get(doc_id, {})
        item = {
            **existing,
            "doc_id": doc_id,
            "doc_name": doc.file_name,
            "file_type": doc.file_type,
            "file_path": doc.file_path,
            "raw_text_path": doc.metadata.get("raw_text_path", existing.get("raw_text_path", "")),
            "char_count": len(doc.raw_text),
            "indexed": existing.get("indexed", False),
            "status": existing.get("status", "uploaded"),
            "chunk_count": existing.get("chunk_count", 0),
            "graph_rule": existing.get("graph_rule") or self.graph_governance_summary(),
            "updated_at": _now_iso(),
        }
        manifest["documents"][doc_id] = item
        self._save_manifest(manifest)
        return item

    def mark_document_status(
        self,
        doc: Document,
        *,
        status: str,
        indexed: bool | None = None,
        error_msg: str = "",
        chunk_count: int | None = None,
    ) -> dict[str, Any]:
        """Update manifest status for UI-visible indexing progress."""
        manifest = self._load_manifest()
        doc_id = self._manifest_doc_id(doc, manifest)
        doc.metadata["lightrag_doc_id"] = doc_id
        existing = manifest["documents"].get(doc_id, {})
        item = {
            **existing,
            "doc_id": doc_id,
            "doc_name": doc.file_name,
            "file_type": doc.file_type,
            "file_path": doc.file_path,
            "raw_text_path": doc.metadata.get("raw_text_path", existing.get("raw_text_path", "")),
            "char_count": len(doc.raw_text),
            "status": status,
            "updated_at": _now_iso(),
        }
        if indexed is not None:
            item["indexed"] = indexed
        if error_msg:
            item["error_msg"] = error_msg
        elif status not in {"failed"}:
            item["error_msg"] = ""
        if chunk_count is not None:
            item["chunk_count"] = chunk_count
        manifest["documents"][doc_id] = item
        self._save_manifest(manifest)
        return item

    async def index_document(
        self,
        doc: Document,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> dict[str, Any]:
        self.assert_embedding_compatible()
        rag = await self.get_rag()
        if chunk_size is not None:
            rag.chunk_token_size = chunk_size
        if chunk_overlap is not None:
            rag.chunk_overlap_token_size = chunk_overlap
        if separators is not None:
            rag.addon_params = rag.addon_params or {}
            rag.addon_params.setdefault("chunker", {}).setdefault("recursive_character", {})[
                "separators"
            ] = separators
        guidance = self.graph_extraction_guidance()
        if guidance:
            rag.addon_params["entity_types_guidance"] = guidance

        registered = self.register_upload(doc)
        doc_id = registered["doc_id"]
        doc.metadata["lightrag_doc_id"] = doc_id
        manifest = self._load_manifest()
        item = manifest["documents"].get(doc_id, {})
        try:
            existing_status = await self.get_doc_status(doc_id)
            if existing_status is not None:
                deletion = await rag.adelete_by_doc_id(doc_id)
                deletion_status = str(getattr(deletion, "status", "") or "").lower()
                if deletion_status not in {"success", "not_found"}:
                    message = getattr(deletion, "message", "") or str(deletion)
                    raise RuntimeError(
                        f"Existing LightRAG document could not be removed before re-index: {message}"
                    )
                logger.info("Removed existing LightRAG doc before re-index: {}", doc_id)
            collector = install_stage_timing(rag)
            with collector.scope():
                result = await rag.ainsert(
                    doc.raw_text, ids=[doc_id], file_paths=[doc.file_path]
                )
            self._last_stage_timings = collector.to_stages()
            status = await self.get_doc_status(doc_id)
            if status and status.status == "failed":
                item.update(
                    {
                        "indexed": False,
                        "status": "failed",
                        "chunk_count": status.chunk_count,
                    "chunks_list": status.chunks_list or [],
                    "error_msg": status.error_msg,
                    "chunking": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "separators": separators,
                    },
                    "graph_rule": self.graph_governance_summary(),
                    "updated_at": _now_iso(),
                }
                )
                manifest["documents"][doc_id] = item
                self._save_manifest(manifest)
                raise RuntimeError(status.error_msg or "LightRAG document processing failed")

            item.update(
                {
                    "doc_id": doc_id,
                    "doc_name": doc.file_name,
                    "file_type": doc.file_type,
                    "file_path": doc.file_path,
                    "raw_text_path": doc.metadata.get("raw_text_path", item.get("raw_text_path", "")),
                    "char_count": len(doc.raw_text),
                    "indexed": True,
                    "status": status.status if status else "indexed",
                    "chunk_count": status.chunk_count if status else item.get("chunk_count", 0),
                    "chunks_list": status.chunks_list if status else item.get("chunks_list", []),
                    "error_msg": status.error_msg if status else "",
                    "chunking": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "separators": separators,
                    },
                    "graph_rule": self.graph_governance_summary(),
                    "last_insert_result": result,
                    "updated_at": _now_iso(),
                }
            )
            manifest["documents"][doc_id] = item
            self._save_manifest(manifest)
            self.record_embedding_signature(overwrite=True)
            return item
        except Exception:
            item.update({"indexed": False, "status": "failed", "updated_at": _now_iso()})
            manifest["documents"][doc_id] = item
            self._save_manifest(manifest)
            raise

    async def get_doc_status(self, doc_id: str) -> LightRAGDocStatus | None:
        rag = await self.get_rag()
        for status in DocStatus:
            items = await rag.get_docs_by_status(status)
            item = (items or {}).get(doc_id)
            if item is None:
                continue
            status_value = getattr(item, "status", status)
            status_text = getattr(status_value, "value", str(status_value)).lower()
            return LightRAGDocStatus(
                doc_id=doc_id,
                status=status_text,
                chunk_count=getattr(item, "chunks_count", 0) or 0,
                chunks_list=list(getattr(item, "chunks_list", None) or []),
                error_msg=getattr(item, "error_msg", "") or "",
            )
        return None

    async def get_document_chunks(self, doc_name_or_id: str) -> list[dict[str, Any]]:
        docs = await self.list_documents()
        item = next(
            (
                d
                for d in docs
                if d.get("doc_name") == doc_name_or_id or d.get("doc_id") == doc_name_or_id
            ),
            None,
        )
        if item is None:
            return []

        chunk_ids = list(item.get("chunks_list") or [])
        if not chunk_ids:
            return []

        rag = await self.get_rag()
        records = await rag.text_chunks.get_by_ids(chunk_ids)
        chunks = []
        for i, record in enumerate(records or []):
            if not record:
                continue
            text = record.get("content") or record.get("text") or ""
            chunks.append(
                {
                    "chunk_id": chunk_ids[i] if i < len(chunk_ids) else record.get("id", str(i)),
                    "chunk_index": record.get("chunk_order_index", i),
                    "text": text,
                    "char_count": len(text),
                }
            )
        chunks.sort(key=lambda c: c["chunk_index"])
        return chunks

    def read_graph(self, *, limit: int = 200, include_isolated: bool = True) -> dict[str, Any]:
        """Read LightRAG's GraphML as frontend-friendly nodes and edges."""
        path = self.graphml_path
        if not path.exists():
            return {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "path": str(path),
                    "exists": False,
                    "total_nodes": 0,
                    "total_edges": 0,
                    "returned_nodes": 0,
                    "returned_edges": 0,
                    "truncated": False,
                },
            }

        try:
            graph = nx.read_graphml(path)
        except Exception as exc:
            logger.warning("Failed to read LightRAG graphml {}: {}", path, exc)
            return {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "path": str(path),
                    "exists": True,
                    "error": str(exc),
                    "total_nodes": 0,
                    "total_edges": 0,
                    "returned_nodes": 0,
                    "returned_edges": 0,
                    "truncated": False,
                },
            }

        limit = max(1, min(int(limit or 200), 1000))
        degree = dict(graph.degree())

        def clean(value: Any, max_len: int = 300) -> str:
            text = str(value or "").replace("<SEP>", "；")
            text = CONTROL_CHARS_RE.sub("", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_len].rstrip()

        def node_rank(item: tuple[str, dict[str, Any]]) -> tuple[int, int, int, str]:
            node_id, data = item
            return (
                degree.get(node_id, 0),
                len(clean(data.get("description"), 1000)),
                len(clean(data.get("source_id"), 1000)),
                str(node_id),
            )

        all_node_items = list(graph.nodes(data=True))
        if include_isolated:
            selected_items = sorted(all_node_items, key=node_rank, reverse=True)[:limit]
        else:
            selected_items = [
                item for item in sorted(all_node_items, key=node_rank, reverse=True)
                if degree.get(item[0], 0) > 0
            ][:limit]
        selected_ids = {node_id for node_id, _ in selected_items}

        nodes = []
        for node_id, data in selected_items:
            entity_type = clean(data.get("entity_type") or "entity", 80).lower()
            label = clean(data.get("entity_id") or node_id, 80) or str(node_id)
            description = clean(data.get("description"), 500)
            source_id = clean(data.get("source_id"), 300)
            file_path = clean(data.get("file_path"), 180)
            if not description and source_id:
                description = f"来源: {source_id}"
            if file_path:
                description = f"{description}（{file_path}）" if description else file_path
            nodes.append(
                {
                    "id": str(node_id),
                    "label": label,
                    "category": GRAPH_CATEGORY_MAP.get(entity_type, entity_type or "核心系统"),
                    "description": description or "暂无描述",
                    "critical": degree.get(node_id, 0) >= 3,
                    "entity_type": entity_type,
                    "source_id": source_id,
                    "file_path": file_path,
                    "degree": degree.get(node_id, 0),
                }
            )

        edges = []
        for source, target, data in graph.edges(data=True):
            if source not in selected_ids or target not in selected_ids:
                continue
            description = clean(data.get("description"), 300)
            keywords = clean(data.get("keywords"), 160)
            relation = description or keywords or "related"
            edges.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "relation": relation,
                    "description": description,
                    "keywords": keywords,
                    "weight": float(data.get("weight") or 1.0),
                    "source_id": clean(data.get("source_id"), 300),
                    "file_path": clean(data.get("file_path"), 180),
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "path": str(path),
                "exists": True,
                "total_nodes": graph.number_of_nodes(),
                "total_edges": graph.number_of_edges(),
                "returned_nodes": len(nodes),
                "returned_edges": len(edges),
                "truncated": graph.number_of_nodes() > len(nodes),
                "directed": graph.is_directed(),
            },
        }

    def find_graph_references(
        self,
        *,
        doc_id: str,
        doc_name: str,
        sample_limit: int = 8,
    ) -> dict[str, Any]:
        """Check whether graph nodes/edges still reference a deleted document."""
        path = self.graphml_path
        if not path.exists():
            return {
                "checked": True,
                "has_residuals": False,
                "node_count": 0,
                "edge_count": 0,
                "nodes": [],
                "edges": [],
                "graph_exists": False,
            }

        try:
            graph = nx.read_graphml(path)
        except Exception as exc:
            logger.warning("Failed to inspect LightRAG graphml {} after delete: {}", path, exc)
            return {
                "checked": False,
                "has_residuals": False,
                "node_count": 0,
                "edge_count": 0,
                "nodes": [],
                "edges": [],
                "graph_exists": True,
                "error": str(exc),
            }

        def clean(value: Any, max_len: int = 240) -> str:
            text = str(value or "").replace("<SEP>", "；")
            text = CONTROL_CHARS_RE.sub("", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_len].rstrip()

        doc_stem = Path(doc_name).stem if doc_name else ""
        needles = [value for value in {doc_id, doc_name} if value]
        if len(doc_stem) >= 4:
            needles.append(doc_stem)

        def has_reference(data: dict[str, Any]) -> bool:
            values = [
                clean(data.get("source_id"), 2000),
                clean(data.get("file_path"), 2000),
            ]
            return any(needle in value for needle in needles for value in values)

        node_count = 0
        edge_count = 0
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for node_id, data in graph.nodes(data=True):
            if not has_reference(data):
                continue
            node_count += 1
            if len(nodes) < sample_limit:
                nodes.append(
                    {
                        "id": str(node_id),
                        "label": clean(data.get("entity_id") or node_id, 80),
                        "source_id": clean(data.get("source_id")),
                        "file_path": clean(data.get("file_path")),
                    }
                )

        for source, target, data in graph.edges(data=True):
            if not has_reference(data):
                continue
            edge_count += 1
            if len(edges) < sample_limit:
                edges.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "source_id": clean(data.get("source_id")),
                        "file_path": clean(data.get("file_path")),
                    }
                )

        return {
            "checked": True,
            "has_residuals": node_count > 0 or edge_count > 0,
            "node_count": node_count,
            "edge_count": edge_count,
            "nodes": nodes,
            "edges": edges,
            "graph_exists": True,
        }

    async def delete_document(self, doc_name_or_id: str) -> dict[str, Any]:
        await self.get_rag()
        manifest = self._load_manifest()
        docs = manifest["documents"]
        match_id = None
        for doc_id, item in docs.items():
            if doc_id == doc_name_or_id or item.get("doc_name") == doc_name_or_id:
                match_id = doc_id
                break
        if match_id is None:
            raise KeyError(doc_name_or_id)

        item = docs[match_id]
        existing_status = await self.get_doc_status(match_id)
        deletion = None
        if item.get("indexed") or existing_status is not None:
            deletion = await self.rag.adelete_by_doc_id(match_id)
            deletion_status = str(getattr(deletion, "status", "") or "").lower()
            if deletion_status not in {"success", "not_found"}:
                message = getattr(deletion, "message", "") or str(deletion)
                raise RuntimeError(f"LightRAG document deletion was not completed: {message}")
        removed = docs.pop(match_id)
        self._save_manifest(manifest)
        return {
            "doc_id": match_id,
            "doc_name": removed.get("doc_name", doc_name_or_id),
            "file_path": removed.get("file_path", ""),
            "raw_text_path": removed.get("raw_text_path", ""),
            "deletion": str(deletion),
        }

    async def invalidate_document(self, doc_name_or_id: str) -> dict[str, Any]:
        """Remove derived LightRAG data while preserving the uploaded document."""
        await self.get_rag()
        manifest = self._load_manifest()
        docs = manifest["documents"]
        match_id = next(
            (
                doc_id
                for doc_id, item in docs.items()
                if doc_id == doc_name_or_id or item.get("doc_name") == doc_name_or_id
            ),
            None,
        )
        if match_id is None:
            raise KeyError(doc_name_or_id)

        item = docs[match_id]
        existing_status = await self.get_doc_status(match_id)
        deletion = None
        if item.get("indexed") or existing_status is not None:
            deletion = await self.rag.adelete_by_doc_id(match_id)
            deletion_status = str(getattr(deletion, "status", "") or "").lower()
            if deletion_status not in {"success", "not_found"}:
                message = getattr(deletion, "message", "") or str(deletion)
                raise RuntimeError(f"LightRAG document invalidation was not completed: {message}")

        item.update(
            {
                "indexed": False,
                "status": "uploaded",
                "chunk_count": 0,
                "chunks_list": [],
                "error_msg": "",
                "updated_at": _now_iso(),
            }
        )
        docs[match_id] = item
        self._save_manifest(manifest)
        return {"doc_id": match_id, "doc_name": item.get("doc_name", doc_name_or_id), "deletion": str(deletion)}

    def _query_param(
        self,
        *,
        mode: str = DEFAULT_MODE,
        top_k: int = 40,
        chunk_top_k: int = 20,
        enable_rerank: bool = True,
        stream: bool = False,
        only_need_context: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> QueryParam:
        return QueryParam(
            mode=mode or DEFAULT_MODE,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            enable_rerank=enable_rerank,
            stream=stream,
            only_need_context=only_need_context,
            conversation_history=history or [],
            include_references=True,
        )

    def _extract_raw(self, result: dict[str, Any]) -> dict[str, Any]:
        raw_data = result.get("raw_data") or {}
        if raw_data.get("data"):
            return raw_data
        return {
            "status": result.get("status", "success"),
            "message": result.get("message", ""),
            "data": result.get("data", {}),
            "metadata": result.get("metadata", {}),
        }

    def _extract_content(self, result: dict[str, Any]) -> str:
        llm_response = result.get("llm_response") or {}
        content = llm_response.get("content")
        if isinstance(content, str):
            return content
        data = result.get("data") or {}
        if isinstance(data, str):
            return data
        return result.get("content") or ""

    def _extract_iterator(self, result: dict[str, Any]) -> AsyncIterator[str] | None:
        llm_response = result.get("llm_response") or {}
        iterator = llm_response.get("response_iterator")
        if iterator is not None:
            return iterator
        content = llm_response.get("content")
        if hasattr(content, "__aiter__"):
            return content
        return None

    def _citations_from_raw(self, raw_data: dict[str, Any]) -> list[dict[str, Any]]:
        data = raw_data.get("data") or {}
        chunks = data.get("chunks") or []
        citations = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks[:10]:
            file_path = chunk.get("file_path") or ""
            chunk_id = chunk.get("chunk_id") or chunk.get("reference_id") or ""
            key = (file_path, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            content = chunk.get("content") or ""
            citations.append(
                {
                    "index": len(citations) + 1,
                    "doc_name": _basename(file_path) or "LightRAG",
                    "chunk_index": len(citations),
                    "excerpt": content[:240],
                    "chunk_id": chunk_id,
                    "file_path": file_path,
                }
            )
        return citations

    async def query(
        self,
        query: str,
        *,
        mode: str = DEFAULT_MODE,
        top_k: int = 40,
        chunk_top_k: int = 20,
        enable_rerank: bool = True,
        history: list[dict[str, str]] | None = None,
    ) -> LightRAGQueryResult:
        self.assert_embedding_compatible()
        rag = await self.get_rag()
        result = await rag.aquery_llm(
            query,
            param=self._query_param(
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                enable_rerank=enable_rerank,
                history=history,
            ),
        )
        raw = self._extract_raw(result)
        return LightRAGQueryResult(
            content=self._extract_content(result),
            raw_data=raw,
            citations=self._citations_from_raw(raw),
        )

    async def stream_query(
        self,
        query: str,
        *,
        mode: str = DEFAULT_MODE,
        top_k: int = 40,
        chunk_top_k: int = 20,
        enable_rerank: bool = True,
        history: list[dict[str, str]] | None = None,
    ) -> LightRAGStreamResult:
        self.assert_embedding_compatible()
        rag = await self.get_rag()
        result = await rag.aquery_llm(
            query,
            param=self._query_param(
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                enable_rerank=enable_rerank,
                stream=True,
                history=history,
            ),
        )
        raw = self._extract_raw(result)
        iterator = self._extract_iterator(result)
        if iterator is None:
            content = self._extract_content(result)

            async def single_token() -> AsyncIterator[str]:
                yield content

            iterator = single_token()
        return LightRAGStreamResult(
            iterator=iterator,
            raw_data=raw,
            citations=self._citations_from_raw(raw),
        )

    async def preview_context(
        self,
        query: str,
        *,
        mode: str = DEFAULT_MODE,
        top_k: int = 40,
        chunk_top_k: int = 20,
        enable_rerank: bool = True,
    ) -> dict[str, Any]:
        self.assert_embedding_compatible()
        rag = await self.get_rag()
        result = await rag.aquery_llm(
            query,
            param=self._query_param(
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                enable_rerank=enable_rerank,
                only_need_context=True,
            ),
        )
        raw = self._extract_raw(result)
        return {
            "query": query,
            "mode": mode,
            "context": self._extract_content(result),
            "data": raw.get("data") or {},
            "metadata": raw.get("metadata") or {},
        }

    async def text_recall(
        self,
        query: str,
        *,
        top_k: int = 20,
        enable_rerank: bool = True,
    ) -> dict[str, Any]:
        """Inspect LightRAG chunk-vector recall before and after reranking."""
        self.assert_embedding_compatible()
        rag = await self.get_rag()
        raw_hits = await rag.chunks_vdb.query(query, top_k=top_k)
        vector_hits: list[dict[str, Any]] = []
        for rank, item in enumerate(raw_hits, start=1):
            if not isinstance(item, dict) or not item.get("content"):
                continue
            vector_hits.append(
                {
                    "chunk_id": str(item.get("id") or ""),
                    "file_path": str(item.get("file_path") or ""),
                    "content": str(item.get("content") or ""),
                    "vector_score": float(item.get("distance") or 0.0),
                    "vector_rank": rank,
                    "rerank_score": None,
                    "rerank_rank": None,
                }
            )

        rerank_hits = [dict(item) for item in vector_hits]
        rerank_applied = False
        rerank_warning = ""
        rerank_config = self._runtime_models()["rerank"]
        if enable_rerank and vector_hits:
            if not rerank_config.get("enabled", True):
                rerank_warning = "系统绑定的 Rerank 模型已关闭"
            elif not str(rerank_config.get("api_key") or "").strip():
                rerank_warning = "Rerank 连接没有可用的 API Key"
            else:
                results = await self._make_rerank_func()(
                    query,
                    [item["content"] for item in vector_hits],
                    top_n=len(vector_hits),
                )
                score_by_index = {
                    int(item["index"]): float(item.get("relevance_score") or 0.0)
                    for item in results
                    if isinstance(item, dict) and "index" in item
                }
                if score_by_index:
                    rerank_applied = True
                    for index, item in enumerate(rerank_hits):
                        item["rerank_score"] = score_by_index.get(index)
                    rerank_hits.sort(
                        key=lambda item: (
                            item["rerank_score"] is not None,
                            item["rerank_score"] or 0.0,
                        ),
                        reverse=True,
                    )
                    for rank, item in enumerate(rerank_hits, start=1):
                        item["rerank_rank"] = rank
                else:
                    rerank_warning = "Rerank 未返回有效分数，已保留向量排序"

        return {
            "query": query,
            "workspace": self.workspace,
            "top_k": top_k,
            "cosine_threshold": float(
                getattr(rag.chunks_vdb, "cosine_better_than_threshold", 0.0)
            ),
            "rerank_requested": enable_rerank,
            "rerank_applied": rerank_applied,
            "rerank_warning": rerank_warning,
            "vector_hits": vector_hits,
            "rerank_hits": rerank_hits,
        }

    async def replay_graph_audit(self) -> dict[str, Any]:
        """Replay manual graph governance operations after a full rebuild."""
        rag = await self.get_rag()
        config = self.load_graph_governance()
        entries = list(reversed(config.get("audit_log") or []))
        applied = 0
        skipped = 0
        errors: list[dict[str, str]] = []
        replayable = {
            "create_entity",
            "edit_entity",
            "delete_entity",
            "create_relation",
            "edit_relation",
            "delete_relation",
            "merge_entities",
        }
        for entry in entries:
            action = str(entry.get("action") or "")
            if action not in replayable:
                continue
            payload = entry.get("payload") or {}
            try:
                if action == "create_entity":
                    await rag.acreate_entity(
                        entity_name=payload["entity_name"],
                        entity_data=payload.get("entity_data") or {},
                    )
                elif action == "edit_entity":
                    await rag.aedit_entity(
                        entity_name=payload["entity_name"],
                        updated_data=payload.get("updated_data") or {},
                        allow_rename=bool(payload.get("allow_rename", True)),
                        allow_merge=bool(payload.get("allow_merge", False)),
                    )
                elif action == "delete_entity":
                    await rag.adelete_by_entity(payload["entity_name"])
                elif action == "create_relation":
                    await rag.acreate_relation(
                        source_entity=payload["source_entity"],
                        target_entity=payload["target_entity"],
                        relation_data=payload.get("relation_data") or {},
                    )
                elif action == "edit_relation":
                    await rag.aedit_relation(
                        source_entity=payload["source_entity"],
                        target_entity=payload["target_entity"],
                        updated_data=payload.get("updated_data") or {},
                    )
                elif action == "delete_relation":
                    await rag.adelete_by_relation(
                        payload["source_entity"],
                        payload["target_entity"],
                    )
                elif action == "merge_entities":
                    await rag.amerge_entities(
                        source_entities=payload.get("source_entities") or [],
                        target_entity=payload["target_entity"],
                        target_entity_data=payload.get("target_entity_data") or {},
                    )
                applied += 1
            except Exception as exc:
                if action.startswith("delete_") and "not found" in str(exc).lower():
                    skipped += 1
                    continue
                errors.append(
                    {
                        "audit_id": str(entry.get("id") or ""),
                        "action": action,
                        "error": str(exc),
                    }
                )
        return {
            "workspace": self.workspace,
            "total": len([e for e in entries if e.get("action") in replayable]),
            "applied": applied,
            "skipped": skipped,
            "errors": errors,
        }

    async def clear_workspace(self, *, preserve_manifest: bool = False) -> dict[str, Any]:
        """Clear current LightRAG workspace and manifest without deleting uploads."""
        preserved_manifest = self._load_manifest() if preserve_manifest else {"documents": {}}
        if self._rag is not None:
            for method_name in ("finalize_storages", "afinalize_storages", "close"):
                method = getattr(self._rag, method_name, None)
                if method is None:
                    continue
                try:
                    result = method()
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:
                    logger.debug("Ignoring LightRAG storage finalization error: {}", exc)
            self._rag = None

        workspace_path = self.graphml_path.parent.resolve()
        expected_parent = self.working_dir.resolve()
        if expected_parent not in workspace_path.parents:
            raise RuntimeError(f"Refusing to clear unsafe LightRAG path: {workspace_path}")

        removed_workspace = False
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
            removed_workspace = True

        removed_manifest = False
        if self.manifest_path.exists():
            self.manifest_path.unlink()
            removed_manifest = True
        self._save_manifest(preserved_manifest)
        if self.embedding_meta_path.exists():
            self.embedding_meta_path.unlink()

        return {
            "workspace": self.workspace,
            "workspace_path": str(workspace_path),
            "manifest_path": str(self.manifest_path),
            "removed_workspace": removed_workspace,
            "removed_manifest": removed_manifest,
            "preserved_manifest": preserve_manifest,
        }

    async def list_documents(self) -> list[dict[str, Any]]:
        manifest = self._load_manifest()
        doc_status: dict[str, Any] = {}
        try:
            rag = await self.get_rag()
            for status in DocStatus:
                items = await rag.get_docs_by_status(status)
                for doc_id, item in (items or {}).items():
                    doc_status[doc_id] = item
        except Exception as exc:
            logger.debug("LightRAG doc status unavailable: {}", exc)

        docs = []
        for doc_id, item in manifest.get("documents", {}).items():
            status_obj = doc_status.get(doc_id)
            chunk_count = item.get("chunk_count", 0)
            lightrag_status = item.get("status", "uploaded")
            if status_obj is not None:
                chunk_count = getattr(status_obj, "chunks_count", chunk_count) or chunk_count
                chunks_list = getattr(status_obj, "chunks_list", None)
                error_msg = getattr(status_obj, "error_msg", "") or item.get("error_msg", "")
                status_value = getattr(status_obj, "status", None)
                if status_value is not None:
                    lightrag_status = getattr(status_value, "value", str(status_value)).lower()
            else:
                chunks_list = item.get("chunks_list", [])
                error_msg = item.get("error_msg", "")
            indexed = bool(item.get("indexed", False)) and lightrag_status != "failed"
            docs.append(
                {
                    **item,
                    "doc_id": doc_id,
                    "doc_name": item.get("doc_name", doc_id),
                    "file_type": item.get("file_type", ""),
                    "chunk_count": chunk_count,
                    "chunks_list": chunks_list or item.get("chunks_list", []),
                    "status": lightrag_status,
                    "indexed": indexed,
                    "error_msg": error_msg,
                }
            )
        docs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return docs

    async def stats(self) -> dict[str, Any]:
        docs = await self.list_documents()
        indexed = [d for d in docs if d.get("indexed")]
        return {
            "doc_count": len(indexed),
            "uploaded_doc_count": len(docs),
            "chunk_count": sum(int(d.get("chunk_count") or 0) for d in indexed),
            "lightrag_dir": str(self.working_dir),
            "lightrag_dir_size": _dir_size(self.working_dir),
            "workspace": self.workspace,
            "embed_model": self._runtime_models()["embedding"]["model"],
            "embed_dim": self._runtime_models()["embedding"]["embed_dim"],
        }


_SERVICES: dict[str, LightRAGService] = {}


def get_lightrag_service(workspace: str | None = None) -> LightRAGService:
    key = sanitize_workspace(workspace or get_config().get("lightrag", {}).get("workspace", DEFAULT_WORKSPACE))
    service = _SERVICES.get(key)
    if service is None:
        service = LightRAGService(workspace=key)
        _SERVICES[key] = service
    return service


def reset_lightrag_service(workspace: str | None = None) -> None:
    if workspace is None:
        _SERVICES.clear()
        return
    _SERVICES.pop(sanitize_workspace(workspace), None)
