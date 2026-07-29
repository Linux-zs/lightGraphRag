"""FastAPI server for TDX Knowledge Base workbench."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from src.config_loader import get_config, reset_config
from src.doc_processor.chunker import TextChunker
from src.doc_processor.loader import DocumentLoader
from src.doc_processor.parsers.base_parser import Document
from src.llm_backend.siliconflow import SiliconFlowBackend
from src.lightrag_service import (
    DEFAULT_WORKSPACE,
    GRAPH_CATEGORY_MAP,
    get_lightrag_service,
    reset_lightrag_service,
    sanitize_workspace,
)
from src.model_profiles import (
    delete_profile,
    discover_models,
    get_bindings,
    get_profile_with_key,
    get_runtime_model_config,
    list_profiles,
    save_bindings,
    test_chat,
    test_embedding,
    test_rerank,
    upsert_profile,
)

# --- App Setup ---

app = FastAPI(title="TDX KB Workbench", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_API_TOKEN = os.environ.get("TDX_APP_API_TOKEN", "").strip()


def _is_loopback_client(host: str | None) -> bool:
    if not host:
        return True
    if host in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def require_remote_api_token(request: Request, call_next):
    """Keep local desktop use frictionless and protect explicitly remote binds."""
    client_host = request.client.host if request.client else None
    if not _is_loopback_client(client_host):
        provided = request.headers.get("X-App-Token", "").strip()
        if not APP_API_TOKEN or not hmac.compare_digest(provided, APP_API_TOKEN):
            return JSONResponse(
                status_code=403,
                content={"detail": "Remote API access requires a valid X-App-Token"},
            )
    return await call_next(request)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"
UPLOAD_DIR = Path(os.environ.get("TDX_UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RAW_TEXT_DIR = Path(os.environ.get("TDX_RAW_TEXT_DIR", "data/upload_text"))
RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_SETTINGS_DIR = Path(
    os.environ.get("TDX_WORKSPACE_SETTINGS_DIR", "data/workspace_settings")
)
WORKSPACE_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS_DIR = Path("data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _safe_leaf_name(value: str, *, label: str = "File name") -> str:
    """Validate a user-controlled name used directly below an application directory."""
    name = str(value or "").strip()
    if not name:
        raise ValueError(f"{label} cannot be empty")
    if name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"Invalid {label.lower()}")
    if Path(name).name != name:
        raise ValueError(f"Invalid {label.lower()}")
    return name


def _workspace_upload_dir(workspace: str, *, create: bool = False) -> Path:
    workspace_dir = UPLOAD_DIR.resolve() / sanitize_workspace(workspace)
    if workspace_dir.parent != UPLOAD_DIR.resolve():
        raise ValueError("Invalid workspace upload path")
    if create:
        workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


def _workspace_raw_text_dir(workspace: str, *, create: bool = False) -> Path:
    workspace_dir = RAW_TEXT_DIR.resolve() / sanitize_workspace(workspace)
    if workspace_dir.parent != RAW_TEXT_DIR.resolve():
        raise ValueError("Invalid workspace raw-text path")
    if create:
        workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


def _resolve_upload_path(
    file_name: str,
    workspace: str = DEFAULT_WORKSPACE,
    *,
    create_dir: bool = False,
    migrate_legacy: bool = False,
) -> Path:
    """Resolve a workspace-owned upload and optionally copy a legacy source."""
    safe_name = _safe_leaf_name(file_name)
    upload_root = _workspace_upload_dir(workspace, create=create_dir)
    candidate = (upload_root / safe_name).resolve()
    if candidate.parent != upload_root:
        raise ValueError("Invalid file name")
    if migrate_legacy and not candidate.exists():
        legacy = (UPLOAD_DIR.resolve() / safe_name).resolve()
        if legacy.parent == UPLOAD_DIR.resolve() and legacy.is_file():
            upload_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, candidate)
            logger.info("Migrated legacy upload '{}' into workspace '{}'", safe_name, workspace)
    return candidate


def _resolve_raw_text_path(workspace: str, doc_id: str, *, create_dir: bool = False) -> Path:
    safe_doc_id = _safe_leaf_name(doc_id, label="Document id")
    raw_root = _workspace_raw_text_dir(workspace, create=create_dir)
    candidate = (raw_root / f"{safe_doc_id}.txt").resolve()
    if candidate.parent != raw_root:
        raise ValueError("Invalid document id")
    return candidate


def _cache_key(workspace: str, file_name: str) -> tuple[str, str]:
    return sanitize_workspace(workspace), _safe_leaf_name(file_name)


def _clear_workspace_cache(workspace: str) -> None:
    workspace = sanitize_workspace(workspace)
    for key in [key for key in _uploaded_files if key[0] == workspace]:
        _uploaded_files.pop(key, None)
    for key in [key for key in _chunk_cache if key[0] == workspace]:
        _chunk_cache.pop(key, None)


def _remove_workspace_sources(workspace: str) -> int:
    """Remove only source and raw-text directories owned by one workspace."""
    workspace = sanitize_workspace(workspace)
    removed_files = 0
    for root, target in (
        (UPLOAD_DIR.resolve(), _workspace_upload_dir(workspace)),
        (RAW_TEXT_DIR.resolve(), _workspace_raw_text_dir(workspace)),
    ):
        resolved = target.resolve()
        if resolved.parent != root:
            raise RuntimeError(f"Refusing to remove unsafe workspace source path: {resolved}")
        if resolved.exists():
            removed_files += sum(1 for path in resolved.rglob("*") if path.is_file())
            shutil.rmtree(resolved)
    _clear_workspace_cache(workspace)
    return removed_files


def _session_path(session_id: str) -> Path:
    """Return the JSON path for a server-generated session id."""
    value = str(session_id or "").strip().lower()
    if not _SESSION_ID_RE.fullmatch(value):
        raise ValueError("Invalid session id")
    return SESSIONS_DIR / f"{value}.json"


DEFAULT_ANSWER_SYSTEM_PROMPT = (
    "你是严谨的知识库问答助手。必须使用简体中文回答。"
    "只依据给定参考资料回答；资料不足时明确说“知识库上下文不足”。"
    "参考资料只作为依据，不要把原文逐条搬运成答案。"
    "你需要先理解用户问题，再综合多条资料，按原因、链路、排查步骤或结论组织成通顺、有逻辑的说明。"
    "回答要结构清晰，使用正常的 Markdown 标题和有序列表。"
    "不要输出 References/引用文档列表，不要输出 assistant/user 角色名，"
    "不要复述大段原始脚本，不要输出孤立的编号或字母。"
    "引用资料时在相关句子末尾使用 [数字] 标记，数字必须来自参考资料编号。"
)
_LEGACY_TDX_PROMPT_PREFIX = "你是通达信系统技术支持知识库助手"


def _workspace_settings_path(workspace: str) -> Path:
    workspace = sanitize_workspace(workspace)
    root = WORKSPACE_SETTINGS_DIR.resolve()
    path = (root / f"{workspace}.json").resolve()
    if path.parent != root:
        raise ValueError("Invalid workspace settings path")
    return path


def _load_workspace_settings(workspace: str) -> dict[str, Any]:
    workspace = sanitize_workspace(workspace)
    path = _workspace_settings_path(workspace)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {"workspace": workspace, **data}
        except Exception:
            logger.exception("Failed to read workspace settings for {}", workspace)

    prompt = DEFAULT_ANSWER_SYSTEM_PROMPT
    default_workspace = sanitize_workspace(
        get_config().get("lightrag", {}).get("workspace", DEFAULT_WORKSPACE)
    )
    if workspace == default_workspace:
        configured = str(
            get_config().get("answer_generation", {}).get("system_prompt") or ""
        ).strip()
        if configured and not configured.startswith(_LEGACY_TDX_PROMPT_PREFIX):
            prompt = configured
    return {"workspace": workspace, "answer_system_prompt": prompt}


def _save_workspace_settings(workspace: str, settings: dict[str, Any]) -> dict[str, Any]:
    workspace = sanitize_workspace(workspace)
    current = _load_workspace_settings(workspace)
    prompt = str(settings.get("answer_system_prompt") or "").strip()
    current["answer_system_prompt"] = prompt or DEFAULT_ANSWER_SYSTEM_PROMPT
    current["workspace"] = workspace
    current["updated_at"] = _task_now()
    path = _workspace_settings_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)
    return current

# --- Pydantic Models ---

class ChunkPreviewRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    file_name: str
    separators: list[str] = Field(default=["\n\n", "\n", "。", "！", "？", "；", " ", ""])
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)

class ChunkPreviewItem(BaseModel):
    index: int
    text: str
    char_count: int

class IndexRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    file_name: str
    separators: list[str] = Field(default=["\n\n", "\n", "。", "！", "？", "；", " ", ""])
    chunk_size: int = Field(default=512, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)

class RecallTestRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    query: str
    mode: str = "mix"
    top_k: int = Field(default=40, ge=1, le=100)
    chunk_top_k: int = Field(default=20, ge=1, le=100)
    enable_rerank: bool = True

class RecallReference(BaseModel):
    reference_id: str = ""
    chunk_id: str = ""
    file_path: str = ""
    content: str = ""

class RecallTestResponse(BaseModel):
    query: str
    mode: str
    context: str
    chunks: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

class ModelConfig(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    embed_model: str = "BAAI/bge-large-zh-v1.5"
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    chat_model: str = "Qwen/Qwen2.5-7B-Instruct"
    chat_temperature: float = 0.7
    chat_top_p: float = 0.9
    chat_max_tokens: int = 4096
    answer_system_prompt: str = DEFAULT_ANSWER_SYSTEM_PROMPT

class EmbedTestRequest(BaseModel):
    text: str

class EmbedTestResponse(BaseModel):
    dimensions: int
    preview: list[float]

class ModelProfileRequest(BaseModel):
    id: Optional[str] = None
    name: str
    api_base: str
    api_key: str = ""
    api_type: str = "openai_compatible"

class ModelDiscoverRequest(BaseModel):
    api_base: str
    api_key: str = ""

class ModelCapabilityTestRequest(BaseModel):
    profile_id: str
    model: str

class ModelBindingsUpdate(BaseModel):
    bindings: dict[str, Any]

class SearchRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    query: str
    mode: str = "mix"
    top_k: int = 40
    chunk_top_k: int = 20
    enable_rerank: bool = True

# --- Chat Models ---

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = ""
    citations: list["Citation"] = []
    evidence: Optional["EvidenceChain"] = None

class ChatSendRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{12}$")
    workspace: str = DEFAULT_WORKSPACE
    message: str
    mode: str = "mix"
    top_k: int = 40
    chunk_top_k: int = 20
    enable_rerank: bool = True

class Citation(BaseModel):
    index: int
    doc_name: str
    chunk_index: int
    excerpt: str

class EvidenceNode(BaseModel):
    id: str
    label: str
    category: str = "核心系统"
    description: str = ""
    critical: bool = False
    entity_type: str = ""
    source_id: str = ""
    file_path: str = ""

class EvidenceEdge(BaseModel):
    source: str
    target: str
    relation: str = ""
    description: str = ""
    keywords: str = ""
    weight: float = 1.0
    source_id: str = ""
    file_path: str = ""

class EvidenceChain(BaseModel):
    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    chunks: list[Citation] = []

class GraphGovernanceConfig(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    rule_template_id: str = ""
    rule_template_name: str = ""
    extraction_mode: str = "assist"
    allow_other_entity_type: bool = True
    entity_types: list[str] = []
    relation_types: list[str] = []
    aliases_text: str = ""
    extraction_prompt: str = ""
    effective_extraction_prompt: str = ""
    reference_files: list[dict[str, Any]] = []
    updated_at: str = ""
    audit_log: list[dict[str, Any]] = []

class GraphGovernanceUpdate(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    rule_template_id: str = ""
    rule_template_name: str = ""
    extraction_mode: str = "assist"
    allow_other_entity_type: bool = True
    entity_types: list[str] = []
    relation_types: list[str] = []
    aliases_text: str = ""
    extraction_prompt: str = ""

class GraphRuleTemplate(BaseModel):
    id: str = ""
    name: str
    description: str = ""
    entity_types: list[str] = []
    relation_types: list[str] = []
    aliases_text: str = ""
    extraction_prompt: str = ""
    built_in: bool = False
    created_at: str = ""
    updated_at: str = ""

class GraphRuleTemplateApplyRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    template_id: str

class GraphEntityCreateRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    entity_name: str
    entity_type: str = "entity"
    description: str = ""
    source_id: str = ""
    file_path: str = ""

class GraphEntityUpdateRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    entity_name: str
    updated_data: dict[str, Any]
    allow_rename: bool = True
    allow_merge: bool = False

class GraphRelationCreateRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    source_entity: str
    target_entity: str
    description: str = ""
    keywords: str = ""
    weight: float = 1.0
    source_id: str = ""
    file_path: str = ""

class GraphRelationUpdateRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    source_entity: str
    target_entity: str
    updated_data: dict[str, Any]

class GraphRelationDeleteRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    source_entity: str
    target_entity: str

class GraphEntityMergeRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    source_entities: list[str]
    target_entity: str
    target_entity_data: dict[str, Any] = {}

class GraphChange(BaseModel):
    action: str
    reason: str = ""
    entity_name: str = ""
    source_entity: str = ""
    target_entity: str = ""
    source_entities: list[str] = []
    entity_data: dict[str, Any] = {}
    relation_data: dict[str, Any] = {}
    updated_data: dict[str, Any] = {}
    target_entity_data: dict[str, Any] = {}

class GraphSuggestRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    instruction: str
    limit: int = Field(default=120, ge=10, le=500)

class GraphSuggestResponse(BaseModel):
    changes: list[GraphChange] = []
    raw_text: str = ""

class GraphApplyChangesRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    changes: list[GraphChange]

class ChatSendResponse(BaseModel):
    session_id: str
    user_message: ChatMessage
    assistant_message: ChatMessage
    citations: list[Citation] = []
    evidence: Optional[EvidenceChain] = None

class ChatSession(BaseModel):
    id: str
    workspace: str = DEFAULT_WORKSPACE
    title: str
    messages: list[ChatMessage]
    created_at: str
    updated_at: str

class ChatSessionListItem(BaseModel):
    id: str
    workspace: str = DEFAULT_WORKSPACE
    title: str
    message_count: int
    created_at: str
    updated_at: str

# --- Session Persistence ---

def _load_session(session_id: str) -> dict | None:
    """Load a session from disk, returns None if not found."""
    try:
        path = _session_path(session_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        data["workspace"] = sanitize_workspace(data.get("workspace") or DEFAULT_WORKSPACE)
        return data
    except Exception:
        return None

def _save_session(session_id: str, data: dict) -> None:
    """Save a session dict to disk."""
    path = _session_path(session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Per-session locks serializing read-modify-write of a session's JSON file so
# that concurrent requests on the same session cannot overwrite each other.
_session_locks: dict[str, asyncio.Lock] = {}
_rag_query_lock = asyncio.Lock()


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """Get (or lazily create) the asyncio.Lock for a given session id."""
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


def _is_contaminated_text(content: str) -> bool:
    """Detect LLM-generated garbage that pollutes history (escape-sequence loops,
    long binary-looking streams, or repeated noise characters).

    The LLM sometimes echoes back weird patterns from JSON-encoded config snippets in
    retrieved chunks. We filter those out before they end up in history and create
    feedback loops that confuse subsequent turns.
    """
    if not content or len(content) < 80:
        return False

    lines = content.split("\n")

    # Pattern A: many short lines whose ONLY content is noise characters
    # (', ", *, \, /, -, ~, `, |, ., :, ;). Real LLM text always has Chinese/English words.
    noise_chars = set("'\"*\\/-~`|.,:; \t")
    noise_lines = 0
    for ln in lines:
        s = ln.strip()
        if not s or len(s) > 80:
            continue
        if all(c in noise_chars for c in s):
            noise_lines += 1
    if noise_lines >= 5:
        return True

    # Pattern B: long sequence of pure binary/digit characters (no Chinese at all)
    has_cjk = any('一' <= c <= '鿿' for c in content)
    if not has_cjk and len(content) >= 60:
        # If >40% of the content is non-space hex/binary characters
        alnum_run = re.sub(r'\s+', '', content)
        # English-word exemption: a legitimate English/technical answer has real
        # word structure (runs of >=2 letters), whereas pure binary/hex streams
        # do not. If we find enough real words making up a meaningful share of
        # the content, skip the binary-contamination check so normal English
        # answers (which legitimately contain no CJK in this Chinese KB) are not
        # wrongly cleared. True binary streams (no word structure) are still caught.
        english_words = re.findall(r"\b[A-Za-z]{2,}\b", content)
        is_legit_english = (
            len(english_words) >= 5
            and sum(len(w) for w in english_words) / max(len(alnum_run), 1) >= 0.3
        )
        if not is_legit_english:
            if len(alnum_run) >= 40 and sum(c.isalnum() for c in alnum_run) / len(alnum_run) > 0.9:
                # And there are no real words (long alnum runs are at least 50% of length)
                long_runs = re.findall(r'[0-9A-Za-z]{20,}', alnum_run)
                if sum(len(s) for s in long_runs) / max(len(alnum_run), 1) > 0.5:
                    return True

    # Pattern C: excessive backslash-escape quotes (JSON serialized)
    if content.count('\\"') > 30:
        return True

    return False


def _sanitize_history_for_llm(history: list[dict]) -> list[dict]:
    """Filter out contaminated assistant messages that would confuse subsequent LLM calls."""
    out = []
    for m in history:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "assistant" and _is_contaminated_text(content):
            logger.warning(f"Skipping contaminated assistant message ({len(content)} chars) from history")
            continue
        out.append(m)
    return out


_NOISE_CHARS = set("'\"*\\/-~`|.,:; \t")


def _detect_repetition_degenerate(text: str) -> tuple[bool, int]:
    """Detect if the LLM has entered a degenerate repetition loop.

    Returns (is_degenerate, safe_truncation_index).
    If not degenerate, safe_truncation_index is the full length.
    """
    if len(text) < 60:
        return False, len(text)

    lines = text.split("\n")

    # 1) Line-level repetition: 5+ consecutive identical lines
    line_run = 1
    max_line_run = 1
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1] and len(lines[i].strip()) > 0:
            line_run += 1
            max_line_run = max(max_line_run, line_run)
        else:
            line_run = 1
    if max_line_run >= 5:
        # Find where the repetition started and truncate
        run_start = 0
        run_count = 1
        for i in range(1, len(lines)):
            if lines[i] == lines[i - 1] and len(lines[i].strip()) > 0:
                run_count += 1
                if run_count == 3:
                    run_start = i - 2
                    break
            else:
                run_count = 1
        if run_start > 0:
            safe = "\n".join(lines[:run_start])
            return True, len(safe)
        return True, len(text[:len(text)//2])

    # 2) Character-level repetition: the same short pattern (2-10 chars)
    #    repeated >8 times consecutively without meaningful content
    for pat_len in range(2, 11):
        pattern = text[-pat_len:]
        if len(pattern.strip()) == 0:
            continue
        # Count how many times this pattern repeats at the end
        count = 0
        idx = len(text) - pat_len
        while idx >= 0 and text[idx:idx + pat_len] == pattern:
            count += 1
            idx -= pat_len
        if count >= 8:
            safe_end = len(text) - (count - 2) * pat_len
            return True, max(0, safe_end)

    # 3) Single-char repetition: the last 30 chars are >80% the same char
    if len(text) >= 30:
        tail = text[-30:]
        for ch in set(tail):
            if tail.count(ch) >= 25 and ch.strip():
                return True, len(text) - 25

    return False, len(text)


def _sanitize_chunk_text(text: str) -> str:
    """Strip noise from a retrieval chunk before it goes into the LLM prompt.

    Normalizes:
      - Non-breaking spaces (U+00A0) and other invisible whitespace → regular space
      - \\r\\n (Windows line endings) → \\n
      - Multiple consecutive whitespace → single space
      - Multiple consecutive blank lines → at most 2

    Drops:
      - Lines whose ONLY content is noise characters
      - Standalone pure-digit/hex lines (no characters > U+002F other than digits and letters)
    """
    if not text:
        return text

    # 1) Normalize invisible whitespace
    out = text.replace("\xa0", " ").replace("\u2003", " ").replace("\u3000", " ")
    out = out.replace("\r\n", "\n").replace("\r", "\n")

    # 2) Drop noise-only lines
    cleaned = []
    for ln in out.split("\n"):
        s = ln.strip()
        if s and len(s) <= 80 and all(c in _NOISE_CHARS for c in s):
            continue
        cleaned.append(ln)

    # 3) Collapse whitespace
    out = "\n".join(cleaned)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = "\n".join(ln.rstrip() for ln in out.split("\n"))
    return out.strip() or text


def _strip_citation_section(text: str) -> str:
    """Defensively strip a trailing '引用文档' section that the LLM may append
    despite prompt instructions forbidding it.

    Removes everything from the first citation-header line (e.g. "引用文档：",
    "## 引用文档") to the end of the text. As a fallback, also removes a bare
    trailing block of two or more consecutive "[数字] 文档名" lines that extend
    to the end of the text (no significant content after them).

    To avoid nuking legitimate numbered lists in the answer body (e.g.
    "[1] 配置说明"), the bare-entry fallback only treats a "[数字] ..." line as
    a citation entry when it contains a known document extension
    (.docx/.pdf/.md/.doc/.txt/.xlsx/.csv, case-insensitive). A trailing block of
    "[1] 通达信.docx" / "[2] 系统.pdf" is stripped; "[1] 配置说明" is kept.

    This is a safety net — the primary fix is in the prompt template. It only
    trims trailing content, so normal answers are never affected.
    """
    if not text:
        return text

    lines = text.split("\n")
    cut_index = len(lines)

    # Header line: "引用文档：", "引用文档:", "## 引用文档", etc.
    header_re = re.compile(r"^\s*#{0,6}\s*引用文档\s*[:：]?\s*$")
    # Bare citation entry: "[1] 文档名.docx" — requires a document extension so
    # that ordinary numbered list items like "[1] 配置说明" are NOT mistaken for
    # citation entries.
    entry_re = re.compile(
        r"^\s*\[\d+\]\s*\S*(?:\.docx|\.pdf|\.md|\.doc|\.txt|\.xlsx|\.csv)\b.*$",
        re.IGNORECASE,
    )

    for i, ln in enumerate(lines):
        # Pattern 1: explicit header line — cut from here to end
        if header_re.match(ln):
            cut_index = i
            break
        # Pattern 2: trailing block of [数字] entries (≥2 consecutive, to end)
        if entry_re.match(ln):
            run_len = 0
            j = i
            while j < len(lines) and entry_re.match(lines[j]):
                run_len += 1
                j += 1
            # Only cut if nothing meaningful follows this block
            has_trailing_content = any(lines[k].strip() for k in range(j, len(lines)))
            if run_len >= 2 and not has_trailing_content:
                cut_index = i
                break

    if cut_index >= len(lines):
        return text

    kept = lines[:cut_index]
    # Strip trailing blank lines left behind
    while kept and not kept[-1].strip():
        kept.pop()
    result = "\n".join(kept)
    return result if result.strip() else text


def _strip_lightrag_noise(text: str) -> str:
    """Remove common LightRAG/model tail artifacts from generated answers."""
    if not text:
        return ""
    cleaned = text.replace("\ufffd", "")
    cleaned = re.sub(r"\n\s*assistant\s*$.*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\n\s*#+\s*References\s*$.*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\n\s*Reference\s*$.*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _clean_excerpt_for_answer(excerpt: str, max_chars: int = 420) -> str:
    """Prepare retrieved snippets for the final LLM prompt.

    LightRAG can retrieve script/config-heavy chunks. Passing those verbatim to
    smaller chat models often causes copy loops, so we keep the useful prose and
    short inline terms while dropping command blocks and noisy shell lines.
    """
    if not excerpt:
        return ""

    lines: list[str] = []
    in_fence = False
    for raw in excerpt.replace("\ufffd", "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        lower = line.lower()
        if re.match(r"^(@?echo|rem\b|set\s+\"?|goto\b|cls\b|if\b|for\b|chmod\b|ping\b|nc\b|net\b)", lower):
            continue
        if re.match(r"^%[a-z0-9_]+%", lower) or re.match(r"^[a-z]:\\", lower):
            continue
        if lower.startswith(("/cygdrive/", "%command%")):
            continue
        if re.fullmatch(r"[\|\-\s:]+", line):
            continue

        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell and not re.fullmatch(r"-+", cell)]
            line = "；".join(cells)

        line = re.sub(r"`([^`]{1,80})`", r"\1", line)
        line = re.sub(r"[*_#>]+", "", line).strip()
        line = re.sub(r"\s{2,}", " ", line)
        if line:
            lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"(?:\b[1DzZ]\b\s*){6,}", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned[:max_chars].rstrip()


def _is_bad_generated_answer(text: str) -> bool:
    """Detect answer degeneration before it is shown or persisted."""
    if not text or not text.strip():
        return True
    if _is_contaminated_text(text):
        return True

    compact = re.sub(r"\s+", " ", text)
    if re.search(r"(?:^|\s)(?:[1DzZ])(?:\s+(?:[1DzZ])){12,}(?:\s|$)", compact):
        return True

    tokens = re.findall(r"\S+", text)
    isolated_noise = re.findall(r"(?<![\w])(?:D|z|Z|1)(?![\w])", text)
    if len(isolated_noise) >= 18 and len(isolated_noise) / max(len(tokens), 1) > 0.18:
        return True

    if len(re.findall(r"^\s*#{1,6}\s*$", text, flags=re.MULTILINE)) >= 3:
        return True
    if text.count("```") % 2 == 1:
        return True

    return False


def _fallback_answer_from_citations(question: str, citations_data: list[dict]) -> str:
    """Return a deterministic, citation-backed answer when model output degrades."""
    if not citations_data:
        return "未检索到相关文档，知识库上下文不足，无法基于当前资料回答。"

    facts: list[str] = []
    seen: set[str] = set()
    for citation in citations_data[:6]:
        cleaned = _clean_excerpt_for_answer(citation.get("excerpt", ""), max_chars=320)
        if not cleaned:
            continue
        pieces = re.split(r"(?<=[。！？])\s+|\n+", cleaned)
        for piece in pieces:
            piece = piece.strip(" -；;")
            normalized = re.sub(r"\s+", "", piece)
            if 18 <= len(piece) <= 180 and normalized not in seen:
                seen.add(normalized)
                facts.append(f"{piece}[{citation.get('index')}]")
                break

    if not facts:
        return (
            "已检索到相关资料，但回答生成服务未能形成可靠答案。"
            "为避免把原文片段误当成结论，本次不做额外推断，请查看下方引用或稍后重试。"
        )

    joined = "；".join(facts[:4])
    return (
        "回答生成出现异常。根据当前知识库中能够直接确认的资料，"
        f"相关信息为：{joined}。以上仅保留资料明确表达的内容，不补充资料之外的推断。"
    )


_QUERY_STOP_TERMS = {
    "今天", "现在", "一下", "一个", "这个", "那个", "什么", "怎么", "怎样", "如何",
    "为什么", "是否", "可以", "需要", "帮我", "请问", "请", "呢", "吗", "的", "了",
}


def _extract_relevance_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = set(re.findall(r"[a-z0-9_+\-.]{2,}", lowered))

    for word in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if word in _QUERY_STOP_TERMS:
            continue
        if 2 <= len(word) <= 8:
            terms.add(word)
        if len(word) > 4:
            for i in range(len(word) - 1):
                gram = word[i:i + 2]
                if gram not in _QUERY_STOP_TERMS:
                    terms.add(gram)

    return {t for t in terms if t and t not in _QUERY_STOP_TERMS}


def _relevance_context_text(citations_data: list[dict]) -> str:
    parts = []
    for citation in citations_data[:8]:
        parts.append(str(citation.get("doc_name", "")))
        parts.append(str(citation.get("file_path", "")))
        parts.append(str(citation.get("excerpt", "")))
    return "\n".join(parts).lower()


def _citations_are_relevant(question: str, citations_data: list[dict], history: list[dict]) -> bool:
    if not citations_data:
        return False

    q_text = question
    # Short follow-ups such as “具体配置呢” can rely on the previous user turn.
    if len(question.strip()) <= 8:
        previous_users = [
            str(m.get("content", ""))
            for m in history[-6:]
            if m.get("role") == "user" and m.get("content")
        ]
        if previous_users:
            q_text = f"{previous_users[-1]}\n{question}"

    q_terms = _extract_relevance_terms(q_text)
    if not q_terms:
        return False

    ctx = _relevance_context_text(citations_data)
    overlap = {term for term in q_terms if len(term) >= 2 and term in ctx}
    long_question_terms = {term for term in q_terms if len(term) >= 3}
    long_overlap = {term for term in long_question_terms if term in ctx}
    if long_overlap:
        return True

    # Require at least two non-generic overlaps for short or domain-agnostic questions.
    return len(overlap) >= 2


def _build_answer_messages(
    question: str,
    citations_data: list[dict],
    history: list[dict],
    workspace: str = DEFAULT_WORKSPACE,
) -> list[dict]:
    context_parts = []
    for citation in citations_data[:6]:
        excerpt = _clean_excerpt_for_answer(citation.get("excerpt") or "")
        if not excerpt:
            continue
        context_parts.append(
            f"[{citation['index']}] {citation.get('doc_name', '文档')} "
            f"#chunk{citation.get('chunk_index', 0)}\n{excerpt}"
        )
    context = "\n\n---\n\n".join(context_parts) if context_parts else "（未检索到相关文档）"
    system = str(
        _load_workspace_settings(workspace).get("answer_system_prompt")
        or DEFAULT_ANSWER_SYSTEM_PROMPT
    )
    user = (
        f"问题：{question}\n\n"
        f"参考资料：\n{context}\n\n"
        "请综合参考资料回答，不要逐条照抄资料，也不要把命中的文档逐个列成清单。"
        "如果资料只是零散提到相关对象、但不能支撑问题中的因果或判断，请明确说明资料不足。"
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-2:])
    messages.append({"role": "user", "content": user})
    return messages


async def _generate_answer_text(
    question: str,
    citations_data: list[dict],
    history: list[dict],
    workspace: str = DEFAULT_WORKSPACE,
) -> str:
    if not citations_data:
        return "未检索到相关文档，知识库上下文不足，无法基于当前资料回答。"

    cfg = get_config()
    runtime_chat = get_runtime_model_config(cfg)["chat"]
    backend = SiliconFlowBackend(
        {
            "base_url": runtime_chat["base_url"],
            "api_key": runtime_chat["api_key"],
            "chat_model": runtime_chat["model"],
            "timeout": runtime_chat.get("timeout", 90),
        }
    )
    try:
        response = await backend.chat(
            messages=_build_answer_messages(
                question,
                citations_data,
                history,
                workspace,
            ),
            temperature=0.1,
            top_p=min(runtime_chat.get("top_p", 0.9), 0.85),
            max_tokens=min(runtime_chat.get("max_tokens", 4096), 900),
            frequency_penalty=0.5,
            presence_penalty=0.0,
        )
        ai_text = _strip_lightrag_noise(_strip_citation_section(response.content))
        if _is_bad_generated_answer(ai_text) or not re.search(r"\[\d+\]", ai_text):
            logger.warning("Generated answer looked degenerate; using citation fallback")
            return _fallback_answer_from_citations(question, citations_data)
        return ai_text
    except Exception as e:
        logger.warning(f"Answer generation failed; using citation fallback: {e}")
        return _fallback_answer_from_citations(question, citations_data)


def _format_chat_citations(citations_data: list[dict]) -> list[Citation]:
    return [
        Citation(
            index=c["index"],
            doc_name=c["doc_name"],
            chunk_index=c.get("chunk_index", i),
            excerpt=c.get("excerpt", ""),
        )
        for i, c in enumerate(citations_data)
    ]


def _dump_chat_citations(citations: list[Citation]) -> list[dict]:
    return [c.model_dump() for c in citations]


def _clean_evidence_value(value: Any, max_len: int = 300) -> str:
    text = str(value or "").replace("<SEP>", "；")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len].rstrip()


def _evidence_entity_id(entity: dict[str, Any]) -> str:
    return _clean_evidence_value(
        entity.get("entity_name")
        or entity.get("entity_id")
        or entity.get("id")
        or entity.get("name")
        or entity.get("__id__"),
        120,
    )


def _evidence_node_from_entity(entity: dict[str, Any]) -> EvidenceNode | None:
    node_id = _evidence_entity_id(entity)
    if not node_id:
        return None

    raw_type = _clean_evidence_value(entity.get("entity_type") or entity.get("type") or "entity", 80).lower()
    content = _clean_evidence_value(entity.get("content") or entity.get("description"), 600)
    if content.lower().startswith(node_id.lower()):
        content = content[len(node_id):].strip(" \n\t:：-")
    description = _clean_evidence_value(entity.get("description") or content or "暂无描述", 500)
    source_id = _clean_evidence_value(entity.get("source_id"), 300)
    return EvidenceNode(
        id=node_id,
        label=_clean_evidence_value(entity.get("entity_name") or entity.get("entity_id") or node_id, 80),
        category=GRAPH_CATEGORY_MAP.get(raw_type, raw_type or "核心系统"),
        description=description or "暂无描述",
        critical=False,
        entity_type=raw_type,
        source_id=source_id,
        file_path=_clean_evidence_value(entity.get("file_path"), 180),
    )


def _evidence_edge_from_relationship(rel: dict[str, Any]) -> EvidenceEdge | None:
    source = _clean_evidence_value(
        rel.get("src_id") or rel.get("source") or rel.get("source_id_entity") or rel.get("from"),
        120,
    )
    target = _clean_evidence_value(
        rel.get("tgt_id") or rel.get("target") or rel.get("target_id_entity") or rel.get("to"),
        120,
    )
    if not source or not target:
        return None

    content = _clean_evidence_value(rel.get("content"), 800)
    content_parts = [p.strip() for p in re.split(r"[\n\t]+", content) if p.strip()]
    keywords = _clean_evidence_value(rel.get("keywords") or (content_parts[0] if content_parts else ""), 160)
    description = _clean_evidence_value(
        rel.get("description")
        or (content_parts[-1] if len(content_parts) >= 4 else content),
        300,
    )
    relation = description or keywords or "related"
    try:
        weight = float(rel.get("weight") or 1.0)
    except (TypeError, ValueError):
        weight = 1.0
    return EvidenceEdge(
        source=source,
        target=target,
        relation=relation,
        description=description,
        keywords=keywords,
        weight=weight,
        source_id=_clean_evidence_value(rel.get("source_id"), 300),
        file_path=_clean_evidence_value(rel.get("file_path"), 180),
    )


def _build_evidence_chain(data: dict[str, Any], citations: list[Citation]) -> EvidenceChain:
    entities = data.get("entities") if isinstance(data, dict) else []
    relationships = data.get("relationships") if isinstance(data, dict) else []
    nodes_by_id: dict[str, EvidenceNode] = {}

    for entity in (entities or [])[:40]:
        if not isinstance(entity, dict):
            continue
        node = _evidence_node_from_entity(entity)
        if node is not None:
            nodes_by_id[node.id] = node

    edges: list[EvidenceEdge] = []
    for rel in (relationships or [])[:60]:
        if not isinstance(rel, dict):
            continue
        edge = _evidence_edge_from_relationship(rel)
        if edge is None:
            continue
        edges.append(edge)
        if edge.source not in nodes_by_id:
            nodes_by_id[edge.source] = EvidenceNode(id=edge.source, label=edge.source)
        if edge.target not in nodes_by_id:
            nodes_by_id[edge.target] = EvidenceNode(id=edge.target, label=edge.target)

    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1

    nodes = list(nodes_by_id.values())
    for node in nodes:
        node.critical = degree.get(node.id, 0) >= 3

    nodes.sort(key=lambda n: (degree.get(n.id, 0), len(n.description or ""), n.label), reverse=True)
    return EvidenceChain(nodes=nodes[:24], edges=edges[:40], chunks=citations)


async def _empty_async_iter():
    if False:
        yield ""


def _list_sessions(workspace: str | None = None) -> list[dict]:
    """List all sessions sorted by updated_at descending."""
    workspace_filter = sanitize_workspace(workspace) if workspace else None
    sessions = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        s = _load_session(p.stem)
        if s and (workspace_filter is None or s.get("workspace") == workspace_filter):
            sessions.append(s)
    return sessions

# --- State ---

_uploaded_files: dict[tuple[str, str], Document] = {}
_chunk_cache: dict[tuple[str, str], list[dict]] = {}
_index_tasks: dict[str, dict[str, Any]] = {}
_index_task_lock = asyncio.Lock()
_index_write_lock = asyncio.Lock()
INDEX_DOC_TIMEOUT_SECONDS = int(os.environ.get("TDX_INDEX_DOC_TIMEOUT_SECONDS", "180"))


class WorkspaceCreateRequest(BaseModel):
    workspace: str


def _workspace_info(workspace: str) -> dict[str, Any]:
    workspace = sanitize_workspace(workspace)
    service = get_lightrag_service(workspace)
    manifest = service._load_manifest()
    graph = service.read_graph(limit=1, include_isolated=False)
    meta = graph.get("metadata") or {}
    return {
        "workspace": workspace,
        "is_default": workspace == get_config().get("lightrag", {}).get("workspace", DEFAULT_WORKSPACE),
        "doc_count": len([d for d in manifest.get("documents", {}).values() if d.get("indexed")]),
        "uploaded_doc_count": len(manifest.get("documents", {})),
        "graph_nodes": meta.get("total_nodes", 0),
        "graph_edges": meta.get("total_edges", 0),
        "manifest_path": str(service.manifest_path),
        "workspace_path": str(service.workspace_dir),
        "exists": service.workspace_dir.exists() or service.manifest_path.exists(),
    }


def _discover_workspaces() -> list[str]:
    cfg = get_config()
    default = sanitize_workspace(cfg.get("lightrag", {}).get("workspace", DEFAULT_WORKSPACE))
    data_dir = Path(cfg.get("paths", {}).get("data_dir", "./data"))
    lightrag_dir = Path(cfg.get("paths", {}).get("lightrag_dir", data_dir / "lightrag"))
    names = {default}
    if lightrag_dir.exists():
        for child in lightrag_dir.iterdir():
            if child.is_dir():
                try:
                    names.add(sanitize_workspace(child.name))
                except ValueError:
                    continue
    manifests_dir = data_dir / "lightrag_manifests"
    if manifests_dir.exists():
        for path in manifests_dir.glob("*.json"):
            try:
                names.add(sanitize_workspace(path.stem))
            except ValueError:
                continue
    return sorted(names, key=lambda n: (n != default, n))


def _task_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _create_index_task(kind: str, doc_names: list[str], workspace: str) -> dict[str, Any]:
    workspace = sanitize_workspace(workspace)
    task_id = uuid.uuid4().hex[:12]
    now = _task_now()
    task = {
        "task_id": task_id,
        "kind": kind,
        "workspace": workspace,
        "status": "queued",
        "doc_names": doc_names,
        "total": len(doc_names),
        "current": 0,
        "progress": 0,
        "message": "等待索引",
        "current_doc": "",
        "current_doc_started_at": "",
        "timeout_seconds": INDEX_DOC_TIMEOUT_SECONDS,
        "results": [],
        "errors": [],
        "cancel_requested": False,
        "created_at": now,
        "updated_at": now,
    }
    async with _index_task_lock:
        _index_tasks[task_id] = task
    return task


async def _update_index_task(task_id: str, **updates: Any) -> dict[str, Any]:
    async with _index_task_lock:
        task = _index_tasks.get(task_id)
        if task is None:
            return {}
        task.update(updates)
        task["updated_at"] = _task_now()
        total = max(int(task.get("total") or 0), 1)
        task["progress"] = min(100, int((int(task.get("current") or 0) / total) * 100))
        return dict(task)


def _public_index_task(task: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in task.items() if k != "cancel_requested"}


def _load_doc_for_index(doc_name: str, workspace: str) -> Document:
    workspace = sanitize_workspace(workspace)
    doc_name = _safe_leaf_name(doc_name)
    manifest = get_lightrag_service(workspace)._load_manifest()
    match = next(
        (
            (doc_id, item)
            for doc_id, item in manifest.get("documents", {}).items()
            if isinstance(item, dict) and item.get("doc_name") == doc_name
        ),
        None,
    )
    if match is None:
        raise FileNotFoundError(f"文件未登记在当前知识库: {doc_name}")
    doc_id, manifest_item = match
    key = _cache_key(workspace, doc_name)
    doc = _uploaded_files.get(key)
    if doc is not None:
        return doc
    file_path = _resolve_upload_path(doc_name, workspace, migrate_legacy=True)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {doc_name}")
    doc = DocumentLoader().load_document(file_path)
    doc.metadata["lightrag_doc_id"] = doc_id
    raw_text_path = _resolve_raw_text_path(workspace, doc_id)
    if raw_text_path.exists():
        doc.raw_text = raw_text_path.read_text(encoding="utf-8")
        doc.metadata["raw_text_path"] = str(raw_text_path)
    elif manifest_item.get("raw_text_path"):
        legacy_raw_path = Path(str(manifest_item["raw_text_path"]))
        if legacy_raw_path.exists():
            doc.raw_text = legacy_raw_path.read_text(encoding="utf-8")
    _uploaded_files[key] = doc
    return doc


def _workspace_doc_names_for_rebuild(workspace: str) -> list[str]:
    """Return only documents registered in the selected workspace manifest."""
    service = get_lightrag_service(workspace)
    manifest = service._load_manifest()
    loader = DocumentLoader()
    supported_exts = set(loader._ext_to_parser.keys())
    doc_names: list[str] = []
    seen: set[str] = set()
    items = sorted(
        manifest.get("documents", {}).values(),
        key=lambda item: str(item.get("updated_at", "")),
    )
    for item in items:
        doc_name = Path(str(item.get("doc_name") or "")).name
        if not doc_name or doc_name in seen:
            continue
        if Path(doc_name).suffix.lower() not in supported_exts:
            continue
        try:
            source_path = _resolve_upload_path(
                doc_name,
                workspace,
                migrate_legacy=True,
            )
        except ValueError:
            logger.warning("Skipping rebuild source with invalid file name: {}", doc_name)
            continue
        if not source_path.exists():
            logger.warning("Skipping rebuild source missing from upload dir: {}", doc_name)
            continue
        seen.add(doc_name)
        doc_names.append(doc_name)
    return doc_names


def _active_workspace_rebuild(workspace: str) -> dict[str, Any] | None:
    workspace = sanitize_workspace(workspace)
    return next(
        (
            task
            for task in _index_tasks.values()
            if task.get("workspace") == workspace
            and task.get("kind") == "rebuild"
            and task.get("status") in {"queued", "running"}
        ),
        None,
    )


def _ensure_workspace_available(workspace: str) -> None:
    task = _active_workspace_rebuild(workspace)
    if task:
        raise HTTPException(
            409,
            f"Knowledge base is rebuilding ({task['task_id']}): {task.get('message', '')}",
        )


async def _start_workspace_rebuild(
    req: RebuildIndexRequest,
    *,
    reason: str,
    allow_empty: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    req.workspace = sanitize_workspace(req.workspace)
    existing = _active_workspace_rebuild(req.workspace)
    if existing:
        return existing, {"already_running": True}

    doc_names = _workspace_doc_names_for_rebuild(req.workspace)
    if not doc_names and not allow_empty:
        raise HTTPException(400, "No uploaded documents registered in this workspace to rebuild")

    batch_req = BatchIndexRequest(
        workspace=req.workspace,
        doc_names=doc_names,
        separators=req.separators,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    task = await _create_index_task("rebuild", doc_names, req.workspace)
    await _update_index_task(task["task_id"], message=f"准备重建：{reason}")
    try:
        async with _index_write_lock:
            service = get_lightrag_service(req.workspace)
            clear_result = await service.clear_workspace(preserve_manifest=True)
            reset_lightrag_service(req.workspace)
            _clear_workspace_cache(req.workspace)
            logger.info(
                "Cleared LightRAG workspace before rebuild ({}): {}",
                reason,
                clear_result,
            )
    except Exception as exc:
        await _update_index_task(
            task["task_id"],
            status="failed",
            message=f"重建准备失败: {exc}",
            errors=[{"doc_name": "", "status": "error", "error": str(exc)}],
        )
        raise

    if doc_names:
        asyncio.create_task(_run_index_task(task["task_id"], batch_req))
    else:
        service = get_lightrag_service(req.workspace)
        graph_replay = await service.replay_graph_audit()
        replay_errors = graph_replay.get("errors") or []
        await _update_index_task(
            task["task_id"],
            status="failed" if replay_errors else "succeeded",
            current=0,
            message=(
                f"知识库已清空，但人工图谱修改恢复失败: {len(replay_errors)} 项"
                if replay_errors
                else "知识库已清空，无剩余文档需要重建"
            ),
            errors=[
                {
                    "doc_name": "",
                    "status": "error",
                    "error": f"人工图谱修改恢复失败: {replay_errors}",
                }
            ] if replay_errors else [],
            graph_replay=graph_replay,
        )
    return _index_tasks[task["task_id"]], clear_result


async def _run_index_task(task_id: str, req: IndexRequest | BatchIndexRequest) -> None:
    doc_names = [req.file_name] if isinstance(req, IndexRequest) else list(req.doc_names)
    workspace = sanitize_workspace(req.workspace)
    service = get_lightrag_service(workspace)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        await _update_index_task(task_id, message="等待索引写入锁")
        async with _index_write_lock:
            await _update_index_task(task_id, status="running", message="开始索引")
            for idx, doc_name in enumerate(doc_names):
                current_task = _index_tasks.get(task_id, {})
                if current_task.get("cancel_requested"):
                    await _update_index_task(
                        task_id,
                        status="cancelled",
                        current_doc="",
                        current_doc_started_at="",
                        message="任务已取消",
                    )
                    return

                doc: Document | None = None
                try:
                    await _update_index_task(
                        task_id,
                        current=idx,
                        current_doc=doc_name,
                        current_doc_started_at=_task_now(),
                        message=f"正在解析: {doc_name}",
                    )
                    doc = _load_doc_for_index(doc_name, workspace)
                    service.mark_document_status(doc, status="processing", indexed=False)

                    if not doc.raw_text.strip():
                        raise ValueError("Document text is empty")

                    await _update_index_task(
                        task_id,
                        current=idx,
                        message=f"正在写入 LightRAG: {doc_name}，单文档超时 {INDEX_DOC_TIMEOUT_SECONDS}s",
                    )
                    item = await asyncio.wait_for(
                        service.index_document(
                            doc,
                            chunk_size=req.chunk_size,
                            chunk_overlap=req.chunk_overlap,
                            separators=req.separators,
                        ),
                        timeout=INDEX_DOC_TIMEOUT_SECONDS,
                    )
                    result = {
                        "doc_name": doc_name,
                        "doc_id": item["doc_id"],
                        "status": "ok",
                        "chunks": item.get("chunk_count", 0),
                    }
                    results.append(result)
                    logger.info("Indexed '{}' into LightRAG as {}", doc_name, item["doc_id"])
                except asyncio.TimeoutError:
                    error_msg = f"索引超时：单个文档超过 {INDEX_DOC_TIMEOUT_SECONDS} 秒未完成"
                    logger.exception("LightRAG index task timed out for {}", doc_name)
                    if doc is not None:
                        try:
                            service.mark_document_status(doc, status="failed", indexed=False, error_msg=error_msg)
                        except Exception:
                            logger.exception("Failed to mark timeout index status for {}", doc_name)
                    error = {"doc_name": doc_name, "status": "error", "error": error_msg}
                    errors.append(error)
                    results.append(error)
                except Exception as e:
                    error_msg = str(e)
                    logger.exception("LightRAG index task failed for {}", doc_name)
                    try:
                        if doc is None:
                            doc = _uploaded_files.get(_cache_key(workspace, doc_name))
                        if doc is not None:
                            service.mark_document_status(doc, status="failed", indexed=False, error_msg=error_msg)
                    except Exception:
                        logger.exception("Failed to mark failed index status for {}", doc_name)
                    error = {"doc_name": doc_name, "status": "error", "error": error_msg}
                    errors.append(error)
                    results.append(error)

                await _update_index_task(
                    task_id,
                    current=idx + 1,
                    current_doc="",
                    current_doc_started_at="",
                    results=results,
                    errors=errors,
                    message=f"已完成 {idx + 1}/{len(doc_names)}",
                )

        task_kind = _index_tasks.get(task_id, {}).get("kind")
        graph_replay = None
        document_error_count = len(errors)
        if task_kind == "rebuild" and not errors:
            await _update_index_task(task_id, message="正在恢复人工图谱修改")
            graph_replay = await service.replay_graph_audit()
            if graph_replay.get("errors"):
                replay_error = {
                    "doc_name": "",
                    "status": "error",
                    "error": f"人工图谱修改恢复失败: {graph_replay['errors']}",
                }
                errors.append(replay_error)
                results.append(replay_error)

        if document_error_count and document_error_count == len(doc_names):
            status = "failed"
            message = f"索引失败: {document_error_count} 个文档失败"
        elif errors:
            status = "partial"
            message = (
                f"索引部分完成: {len(doc_names) - document_error_count} 个文档成功，"
                f"{document_error_count} 个文档失败，另有 {len(errors) - document_error_count} 个图谱恢复错误"
            )
        else:
            status = "succeeded"
            message = f"索引完成: {len(results)} 个文档"
        await _update_index_task(
            task_id,
            status=status,
            current=len(doc_names),
            current_doc="",
            current_doc_started_at="",
            results=results,
            errors=errors,
            message=message,
            graph_replay=graph_replay,
        )
    except Exception as exc:
        logger.exception("Index task crashed: {}", task_id)
        await _update_index_task(
            task_id,
            status="failed",
            current_doc="",
            current_doc_started_at="",
            results=results,
            errors=errors or [{"doc_name": "", "status": "error", "error": str(exc)}],
            message=f"索引任务异常退出: {exc}",
        )


# --- KB Management Endpoints ---

@app.get("/api/kb/workspaces")
async def list_workspaces():
    """List LightRAG knowledge-base workspaces."""
    return [_workspace_info(name) for name in _discover_workspaces()]


@app.post("/api/kb/workspaces")
async def create_workspace(req: WorkspaceCreateRequest):
    """Create an empty LightRAG knowledge-base workspace."""
    workspace = sanitize_workspace(req.workspace)
    service = get_lightrag_service(workspace)
    result = service.ensure_workspace()
    return {**_workspace_info(workspace), **result}


@app.delete("/api/kb/workspaces/{workspace}")
async def delete_workspace(workspace: str):
    """Delete a non-default workspace index/manifest. Uploaded source files are kept."""
    workspace = sanitize_workspace(workspace)
    default = get_config().get("lightrag", {}).get("workspace", DEFAULT_WORKSPACE)
    if workspace == default:
        raise HTTPException(400, "Default workspace cannot be deleted")
    running = [
        t for t in _index_tasks.values()
        if t.get("workspace") == workspace and t.get("status") in {"queued", "running"}
    ]
    if running:
        raise HTTPException(409, "Index task is running in this workspace")
    service = get_lightrag_service(workspace)
    result = await service.clear_workspace()
    if service.manifest_path.exists():
        service.manifest_path.unlink()
    removed_uploads = _remove_workspace_sources(workspace)
    reset_lightrag_service(workspace)
    return {"deleted": workspace, "removed_uploads": removed_uploads, **result}


@app.post("/api/kb/upload")
async def upload_document(file: UploadFile = File(...), workspace: str = Query(DEFAULT_WORKSPACE)):
    """Upload a document for preview before indexing."""
    workspace = sanitize_workspace(workspace)
    _ensure_workspace_available(workspace)
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    upload_name = str(file.filename).replace("\\", "/").rsplit("/", 1)[-1]
    try:
        safe_name = _safe_leaf_name(upload_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    ext = Path(safe_name).suffix.lower()
    loader = DocumentLoader()
    if ext not in loader._ext_to_parser:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Save to upload dir
    dest = _resolve_upload_path(safe_name, workspace, create_dir=True)
    content = await file.read()
    dest.write_bytes(content)

    # Parse document
    try:
        doc = loader.load_document(dest)
        key = _cache_key(workspace, safe_name)
        service = get_lightrag_service(workspace)
        item = service.register_upload(doc)
        raw_text_path = _resolve_raw_text_path(workspace, item["doc_id"], create_dir=True)
        raw_text_path.write_text(doc.raw_text, encoding="utf-8")
        doc.metadata["lightrag_doc_id"] = item["doc_id"]
        doc.metadata["raw_text_path"] = str(raw_text_path)
        item = service.register_upload(doc)
        _uploaded_files[key] = doc
        return {
            "file_name": safe_name,
            "workspace": workspace,
            "doc_id": item["doc_id"],
            "file_type": doc.file_type,
            "char_count": len(doc.raw_text),
            "preview": doc.raw_text[:500] + ("..." if len(doc.raw_text) > 500 else ""),
        }
    except Exception as e:
        raise HTTPException(500, f"Parse failed: {e}")


@app.post("/api/kb/preview-chunks", response_model=list[ChunkPreviewItem])
async def preview_chunks(req: ChunkPreviewRequest):
    """Preview chunking results without indexing."""
    try:
        req.file_name = _safe_leaf_name(req.file_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    key = _cache_key(req.workspace, req.file_name)
    doc = _uploaded_files.get(key)
    if doc is None:
        try:
            doc = _load_doc_for_index(req.file_name, req.workspace)
        except (FileNotFoundError, ValueError):
            doc = None
    if doc is None:
        raise HTTPException(404, f"File '{req.file_name}' not uploaded yet")

    chunker = TextChunker(
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        separators=req.separators,
    )
    chunks = chunker.chunk_document(doc)

    result = [
        ChunkPreviewItem(index=c.chunk_index, text=c.text, char_count=len(c.text))
        for c in chunks
    ]
    # Cache for indexing
    _chunk_cache[key] = [
        {"index": c.chunk_index, "text": c.text} for c in chunks
    ]
    return result


@app.post("/api/kb/index")
async def index_document(req: IndexRequest):
    """Create a background task to index a document into LightRAG."""
    try:
        req.file_name = _safe_leaf_name(req.file_name)
        file_path = _resolve_upload_path(
            req.file_name,
            req.workspace,
            migrate_legacy=True,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    req.workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(req.workspace)
    manifest = get_lightrag_service(req.workspace)._load_manifest()
    registered = any(
        item.get("doc_name") == req.file_name
        for item in manifest.get("documents", {}).values()
        if isinstance(item, dict)
    )
    if not registered or (_cache_key(req.workspace, req.file_name) not in _uploaded_files and not file_path.exists()):
        raise HTTPException(404, f"File '{req.file_name}' not uploaded yet")

    task = await _create_index_task("single", [req.file_name], req.workspace)
    asyncio.create_task(_run_index_task(task["task_id"], req))
    return _public_index_task(task)


@app.get("/api/kb/documents")
async def list_documents(workspace: str = Query(DEFAULT_WORKSPACE)):
    """List uploaded/indexed documents from the LightRAG manifest and status store."""
    return await get_lightrag_service(workspace).list_documents()


@app.delete("/api/kb/documents/{doc_name}")
async def delete_document(doc_name: str, workspace: str = Query(DEFAULT_WORKSPACE)):
    """Delete a document from LightRAG by doc_id or uploaded file name."""
    workspace = sanitize_workspace(workspace)
    _ensure_workspace_available(workspace)
    service = get_lightrag_service(workspace)
    try:
        result = await service.delete_document(doc_name)
    except KeyError:
        raise HTTPException(404, f"Document '{doc_name}' not found")
    except Exception as e:
        logger.exception("LightRAG delete failed for {}", doc_name)
        raise HTTPException(500, f"LightRAG delete failed: {e}")
    _uploaded_files.pop(_cache_key(workspace, result["doc_name"]), None)
    _chunk_cache.pop(_cache_key(workspace, result["doc_name"]), None)
    for source_path in (
        _resolve_upload_path(result["doc_name"], workspace),
        _resolve_raw_text_path(workspace, result["doc_id"]),
    ):
        try:
            if source_path.is_file():
                source_path.unlink()
        except OSError as exc:
            logger.warning("Failed to remove deleted document source {}: {}", source_path, exc)
    graph_residuals = service.find_graph_references(
        doc_id=result["doc_id"],
        doc_name=result["doc_name"],
    )
    cleanup_task = None
    cleanup_error = ""
    if graph_residuals.get("has_residuals") or not graph_residuals.get("checked", False):
        rebuild_req = RebuildIndexRequest(workspace=workspace)
        try:
            cleanup_task, _clear_result = await _start_workspace_rebuild(
                rebuild_req,
                reason=f"删除文档 {result['doc_name']} 后自动清理图谱",
                allow_empty=True,
            )
        except Exception as exc:
            cleanup_error = str(exc)
            logger.exception("Document deleted but automatic graph cleanup failed to start")
    return {
        "deleted": 1,
        **result,
        "graph_residuals": graph_residuals,
        "cleanup_task": _public_index_task(cleanup_task) if cleanup_task else None,
        "cleanup_error": cleanup_error,
    }


class BatchDeleteRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    doc_names: list[str]

class BatchIndexRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    doc_names: list[str]
    separators: list[str] = ["\n\n", "\n", "。", "！", "？", "；", "  "]
    chunk_size: int = 512
    chunk_overlap: int = 50

class RebuildIndexRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    separators: list[str] = ["\n\n", "\n", "。", "！", "？", "；", "  "]
    chunk_size: int = 512
    chunk_overlap: int = 50

class ClearKnowledgeBaseRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE
    clear_uploads: bool = False

class RawTextUpdateRequest(BaseModel):
    raw_text: str


@app.post("/api/kb/batch-delete")
async def batch_delete(req: BatchDeleteRequest):
    """Delete multiple documents at once."""
    req.workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(req.workspace)
    deleted = 0
    errors = []
    residual_items = []
    service = get_lightrag_service(req.workspace)
    for doc_name in req.doc_names:
        try:
            result = await service.delete_document(doc_name)
            _uploaded_files.pop(_cache_key(req.workspace, result["doc_name"]), None)
            _chunk_cache.pop(_cache_key(req.workspace, result["doc_name"]), None)
            for source_path in (
                _resolve_upload_path(result["doc_name"], req.workspace),
                _resolve_raw_text_path(req.workspace, result["doc_id"]),
            ):
                try:
                    if source_path.is_file():
                        source_path.unlink()
                except OSError as exc:
                    logger.warning("Failed to remove batch-deleted source {}: {}", source_path, exc)
            residual_items.append(
                {
                    "doc_name": result["doc_name"],
                    "doc_id": result["doc_id"],
                    **service.find_graph_references(
                        doc_id=result["doc_id"],
                        doc_name=result["doc_name"],
                    ),
                }
            )
            deleted += 1
            logger.info("Batch-delete: removed LightRAG doc '{}'", doc_name)
        except Exception as e:
            errors.append({"doc_name": doc_name, "error": str(e)})
    cleanup_task = None
    cleanup_error = ""
    needs_cleanup = any(
        item.get("has_residuals") or not item.get("checked", False)
        for item in residual_items
    )
    if needs_cleanup:
        rebuild_req = RebuildIndexRequest(workspace=req.workspace)
        try:
            cleanup_task, _clear_result = await _start_workspace_rebuild(
                rebuild_req,
                reason="批量删除文档后自动清理图谱",
                allow_empty=True,
            )
        except Exception as exc:
            cleanup_error = str(exc)
            logger.exception("Documents deleted but automatic graph cleanup failed to start")
    return {
        "deleted_chunks": deleted,
        "doc_count": len(req.doc_names),
        "errors": errors,
        "graph_residuals": {
            "has_residuals": any(item.get("has_residuals") for item in residual_items),
            "items": residual_items,
        },
        "cleanup_task": _public_index_task(cleanup_task) if cleanup_task else None,
        "cleanup_error": cleanup_error,
    }


@app.post("/api/kb/batch-index")
async def batch_index(req: BatchIndexRequest):
    """Create a background task to index multiple uploaded documents."""
    if not req.doc_names:
        raise HTTPException(400, "No documents selected")
    try:
        req.doc_names = [_safe_leaf_name(name) for name in req.doc_names]
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    req.workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(req.workspace)
    task = await _create_index_task("batch", list(req.doc_names), req.workspace)
    asyncio.create_task(_run_index_task(task["task_id"], req))
    return _public_index_task(task)


@app.post("/api/kb/rebuild")
async def rebuild_index(req: RebuildIndexRequest):
    """Clear current workspace index and rebuild only that workspace's registered documents."""
    task, clear_result = await _start_workspace_rebuild(
        req,
        reason="用户请求",
    )
    return {**_public_index_task(task), "clear_result": clear_result}


@app.post("/api/kb/clear")
async def clear_knowledge_base(req: ClearKnowledgeBaseRequest):
    """Clear LightRAG index/manifest; optionally remove uploaded source files."""
    req.workspace = sanitize_workspace(req.workspace)
    running = [
        t
        for t in _index_tasks.values()
        if t.get("workspace") == req.workspace and t.get("status") in {"queued", "running"}
    ]
    if running:
        raise HTTPException(409, "Index task is running; cancel or wait before clearing")

    async with _index_write_lock:
        service = get_lightrag_service(req.workspace)
        result = await service.clear_workspace()
        reset_lightrag_service(req.workspace)
        _clear_workspace_cache(req.workspace)

        removed_uploads = 0
        if req.clear_uploads:
            removed_uploads = _remove_workspace_sources(req.workspace)

    return {**result, "removed_uploads": removed_uploads}


@app.get("/api/kb/index-tasks")
async def list_index_tasks():
    """List recent in-memory indexing tasks."""
    async with _index_task_lock:
        tasks = sorted(_index_tasks.values(), key=lambda t: t.get("created_at", ""), reverse=True)
        return [_public_index_task(t) for t in tasks[:50]]


@app.get("/api/kb/index-tasks/{task_id}")
async def get_index_task(task_id: str):
    """Get a single indexing task status."""
    async with _index_task_lock:
        task = _index_tasks.get(task_id)
        if not task:
            raise HTTPException(404, f"Index task '{task_id}' not found")
        return _public_index_task(task)


@app.post("/api/kb/index-tasks/{task_id}/cancel")
async def cancel_index_task(task_id: str):
    """Request cancellation for a queued/running indexing task."""
    async with _index_task_lock:
        task = _index_tasks.get(task_id)
        if not task:
            raise HTTPException(404, f"Index task '{task_id}' not found")
        if task.get("status") in {"succeeded", "failed", "partial", "cancelled"}:
            return _public_index_task(task)
        task["cancel_requested"] = True
        task["message"] = f"正在取消；当前文档完成或 {INDEX_DOC_TIMEOUT_SECONDS}s 超时后停止"
        task["updated_at"] = _task_now()
        return _public_index_task(task)


@app.get("/api/kb/documents/{doc_name}/raw-text")
async def get_document_raw_text(
    doc_name: str,
    workspace: str = Query(DEFAULT_WORKSPACE),
):
    """Get the raw parsed text of a document (for preview/editing before chunking)."""
    try:
        doc_name = _safe_leaf_name(doc_name)
        workspace = sanitize_workspace(workspace)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    try:
        key = _cache_key(workspace, doc_name)
        source = "cache" if key in _uploaded_files else "disk"
        doc = _load_doc_for_index(doc_name, workspace)
        return {
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "char_count": len(doc.raw_text),
            "raw_text": doc.raw_text,
            "source": source,
            "workspace": workspace,
        }
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Re-parse failed: {exc}")


@app.put("/api/kb/documents/{doc_name}/raw-text")
async def update_document_raw_text(
    doc_name: str,
    req: RawTextUpdateRequest,
    workspace: str = Query(DEFAULT_WORKSPACE),
):
    """Persist edited parsed text and invalidate derived LightRAG data."""
    try:
        doc_name = _safe_leaf_name(doc_name)
        workspace = sanitize_workspace(workspace)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not req.raw_text.strip():
        raise HTTPException(400, "Raw text cannot be empty")
    _ensure_workspace_available(workspace)
    try:
        doc = _load_doc_for_index(doc_name, workspace)
        service = get_lightrag_service(workspace)
        invalidated = await service.invalidate_document(doc_name)
        raw_text_path = _resolve_raw_text_path(
            workspace,
            invalidated["doc_id"],
            create_dir=True,
        )
        raw_text_path.write_text(req.raw_text, encoding="utf-8")
        doc.raw_text = req.raw_text
        doc.metadata["lightrag_doc_id"] = invalidated["doc_id"]
        doc.metadata["raw_text_path"] = str(raw_text_path)
        service.register_upload(doc)
        _uploaded_files[_cache_key(workspace, doc_name)] = doc
        _chunk_cache.pop(_cache_key(workspace, doc_name), None)
    except KeyError:
        raise HTTPException(404, f"Document '{doc_name}' not found in workspace '{workspace}'")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.exception("Failed to update raw text for {}", doc_name)
        raise HTTPException(500, f"Raw text update failed: {exc}")
    logger.info("Updated raw_text for '{}' in '{}': {} chars", doc_name, workspace, len(req.raw_text))
    return {
        "file_name": doc_name,
        "workspace": workspace,
        "char_count": len(req.raw_text),
        "index_invalidated": True,
        "message": "原始文本已保存，旧索引已移除，请重新预览并索引",
    }


@app.get("/api/kb/documents/{doc_name}/chunks")
async def get_document_chunks(doc_name: str, workspace: str = Query(DEFAULT_WORKSPACE)):
    """Return indexed LightRAG chunks, falling back to local raw-text chunks on failure."""
    try:
        doc_name = _safe_leaf_name(doc_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    workspace = sanitize_workspace(workspace)
    _ensure_workspace_available(workspace)
    service = get_lightrag_service(workspace)
    chunks = await service.get_document_chunks(doc_name)
    if chunks:
        return {"doc_name": doc_name, "total": len(chunks), "chunks": chunks}

    docs = await service.list_documents()
    item = next((d for d in docs if d.get("doc_name") == doc_name or d.get("doc_id") == doc_name), None)
    if item is None:
        return {"doc_name": doc_name, "chunks": [], "total": 0}

    file_path = Path(item.get("file_path") or "")
    if not file_path.exists():
        try:
            file_path = _resolve_upload_path(
                item.get("doc_name", doc_name),
                workspace,
                migrate_legacy=True,
            )
        except ValueError:
            return {"doc_name": item.get("doc_name", doc_name), "total": 0, "chunks": []}
    if not file_path.exists():
        return {"doc_name": item.get("doc_name", doc_name), "total": 0, "chunks": []}

    try:
        doc = _uploaded_files.get(_cache_key(workspace, item.get("doc_name", doc_name)))
        if doc is None:
            doc = _load_doc_for_index(item.get("doc_name", doc_name), workspace)
        chunking = item.get("chunking") or {}
        cfg = get_config()
        default_chunking = cfg.get("chunking", {})
        chunker = TextChunker(
            chunk_size=chunking.get("chunk_size") or default_chunking.get("chunk_size", 512),
            chunk_overlap=chunking.get("chunk_overlap") or default_chunking.get("chunk_overlap", 50),
            separators=chunking.get("separators") or default_chunking.get("separators", ["\n\n", "\n", "。", ""]),
        )
        local_chunks = chunker.chunk_document(doc)
    except Exception as e:
        logger.warning("Fallback chunk preview failed for {}: {}", doc_name, e)
        return {"doc_name": item.get("doc_name", doc_name), "total": 0, "chunks": []}

    chunks = []
    lightrag_ids = list(item.get("chunks_list") or [])
    for i, chunk in enumerate(local_chunks):
        chunks.append(
            {
                "chunk_id": lightrag_ids[i] if i < len(lightrag_ids) else chunk.chunk_id,
                "chunk_index": i,
                "text": chunk.text,
                "char_count": len(chunk.text),
            }
        )
    return {"doc_name": item.get("doc_name", doc_name), "total": len(chunks), "chunks": chunks}


# --- Recall Test Endpoints ---

@app.post("/api/recall/test", response_model=RecallTestResponse)
async def recall_test(req: RecallTestRequest):
    """Preview LightRAG context with QueryParam(only_need_context=True)."""
    req.workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(req.workspace)
    try:
        async with _rag_query_lock:
            result = await get_lightrag_service(req.workspace).preview_context(
                req.query,
                mode=req.mode,
                top_k=req.top_k,
                chunk_top_k=req.chunk_top_k,
                enable_rerank=req.enable_rerank,
            )
    except Exception as e:
        logger.exception("LightRAG context preview failed")
        raise HTTPException(500, f"LightRAG context preview failed: {e}")

    data = result.get("data") or {}
    return RecallTestResponse(
        query=req.query,
        mode=req.mode,
        context=result.get("context", ""),
        chunks=data.get("chunks") or [],
        entities=data.get("entities") or [],
        relationships=data.get("relationships") or [],
        references=data.get("references") or [],
        metadata=result.get("metadata") or {},
    )


# --- Search Endpoint (full pipeline) ---

@app.post("/api/search")
async def search(req: SearchRequest):
    """Full LightRAG search with answer generation."""
    req.workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(req.workspace)
    try:
        answer = await get_lightrag_service(req.workspace).query(
            req.query,
            mode=req.mode,
            top_k=req.top_k,
            chunk_top_k=req.chunk_top_k,
            enable_rerank=req.enable_rerank,
        )
    except Exception as e:
        logger.exception("LightRAG search failed")
        raise HTTPException(500, f"LightRAG search failed: {e}")
    return {
        "question": req.query,
        "content": answer.content,
        "citations": answer.citations,
        "trace": {"mode": req.mode, "raw_data": answer.raw_data},
    }


# --- Model Config Endpoints ---

@app.get("/api/models/config")
async def get_model_config(workspace: str = Query(DEFAULT_WORKSPACE)):
    """Get generation parameters and the selected workspace answer prompt."""
    workspace = sanitize_workspace(workspace)
    cfg = get_config()
    sf = cfg.get("siliconflow", {})
    runtime = get_runtime_model_config(cfg)
    return ModelConfig(
        workspace=workspace,
        embed_model=runtime["embedding"]["model"],
        embed_base_url=runtime["embedding"]["base_url"],
        rerank_model=runtime["rerank"]["model"],
        chat_model=runtime["chat"]["model"],
        chat_temperature=sf.get("chat_temperature", 0.7),
        chat_top_p=sf.get("chat_top_p", 0.9),
        chat_max_tokens=sf.get("chat_max_tokens", 4096),
        answer_system_prompt=_load_workspace_settings(workspace).get(
            "answer_system_prompt",
            DEFAULT_ANSWER_SYSTEM_PROMPT,
        ),
    )


@app.put("/api/models/config")
async def update_model_config(
    config: ModelConfig,
    workspace: str = Query(DEFAULT_WORKSPACE),
):
    """Update global generation parameters and a workspace-specific answer prompt.

    Model endpoints/keys/models are managed by /api/model-profiles and
    /api/model-bindings. This endpoint is kept for compatibility.
    """
    if not CONFIG_PATH.exists():
        raise HTTPException(500, "Config file not found")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)

    sf = yaml_config.setdefault("siliconflow", {})
    sf["chat_temperature"] = config.chat_temperature
    sf["chat_top_p"] = config.chat_top_p
    sf["chat_max_tokens"] = config.chat_max_tokens

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_config, f, allow_unicode=True, default_flow_style=False)

    _save_workspace_settings(
        workspace,
        {"answer_system_prompt": config.answer_system_prompt},
    )
    # Config changed: rebuild the LightRAG adapter on the next request.
    reset_config()
    reset_lightrag_service()

    logger.info("Model config updated and saved")
    return {"status": "ok"}


@app.get("/api/model-profiles")
async def api_list_model_profiles():
    return list_profiles(get_config())


@app.post("/api/model-profiles")
async def api_upsert_model_profile(req: ModelProfileRequest):
    profile = upsert_profile(req.model_dump(exclude_none=True), get_config())
    reset_lightrag_service()
    return profile


@app.put("/api/model-profiles/{profile_id}")
async def api_update_model_profile(profile_id: str, req: ModelProfileRequest):
    payload = req.model_dump(exclude_none=True)
    payload["id"] = profile_id
    profile = upsert_profile(payload, get_config())
    reset_lightrag_service()
    return profile


@app.delete("/api/model-profiles/{profile_id}")
async def api_delete_model_profile(profile_id: str):
    result = delete_profile(profile_id, get_config())
    reset_lightrag_service()
    return result


@app.post("/api/model-profiles/discover")
async def api_discover_models(req: ModelDiscoverRequest):
    try:
        models = await discover_models(req.api_base, req.api_key)
    except Exception as exc:
        logger.exception("Model discovery failed")
        raise HTTPException(400, f"Model discovery failed: {exc}")
    return {"models": models}


@app.post("/api/model-profiles/{profile_id}/discover")
async def api_discover_profile_models(profile_id: str):
    try:
        profile = get_profile_with_key(profile_id, get_config())
        models = await discover_models(profile["api_base"], profile.get("api_key", ""))
        upsert_profile({**profile, "models_cache": models}, get_config())
    except KeyError:
        raise HTTPException(404, "Profile not found")
    except Exception as exc:
        logger.exception("Profile model discovery failed")
        raise HTTPException(400, f"Model discovery failed: {exc}")
    return {"models": models}


@app.post("/api/model-profiles/test-chat")
async def api_test_chat_model(req: ModelCapabilityTestRequest):
    try:
        return await test_chat(req.profile_id, req.model, get_config())
    except Exception as exc:
        logger.exception("Chat model test failed")
        raise HTTPException(400, f"Chat model test failed: {exc}")


@app.post("/api/model-profiles/test-embedding")
async def api_test_embedding_model(req: ModelCapabilityTestRequest):
    try:
        return await test_embedding(req.profile_id, req.model, get_config())
    except Exception as exc:
        logger.exception("Embedding model test failed")
        raise HTTPException(400, f"Embedding model test failed: {exc}")


@app.post("/api/model-profiles/test-rerank")
async def api_test_rerank_model(req: ModelCapabilityTestRequest):
    try:
        return await test_rerank(req.profile_id, req.model, get_config())
    except Exception as exc:
        logger.exception("Rerank model test failed")
        raise HTTPException(400, f"Rerank model test failed: {exc}")


@app.get("/api/model-bindings")
async def api_get_model_bindings():
    return get_bindings(get_config())


@app.put("/api/model-bindings")
async def api_update_model_bindings(req: ModelBindingsUpdate):
    old_runtime = get_runtime_model_config(get_config())
    bindings = save_bindings(req.bindings, get_config())
    new_runtime = get_runtime_model_config(get_config())
    reset_lightrag_service()
    embedding_changed = (
        old_runtime["embedding"]["model"] != new_runtime["embedding"]["model"]
        or int(old_runtime["embedding"]["embed_dim"]) != int(new_runtime["embedding"]["embed_dim"])
    )
    return {"bindings": bindings, "embedding_changed": embedding_changed}


@app.post("/api/models/test-embed", response_model=EmbedTestResponse)
async def test_embed(req: EmbedTestRequest):
    """Test embedding with current model."""
    try:
        rag = await get_lightrag_service().get_rag()
        embeddings = await rag.embedding_func([req.text])
    except Exception as e:
        raise HTTPException(500, f"Embedding failed: {e}")
    if embeddings is None or len(embeddings) == 0:
        raise HTTPException(500, "Embedding failed")
    vec = embeddings[0].tolist() if hasattr(embeddings[0], "tolist") else embeddings[0]
    return EmbedTestResponse(
        dimensions=len(vec),
        preview=[round(v, 6) for v in vec[:10]],
    )


# --- Chat Endpoints ---

def _ensure_session(
    session_id: Optional[str],
    first_message: str,
    workspace: str = DEFAULT_WORKSPACE,
) -> tuple[str, dict]:
    """Get or create a session, return (session_id, session_dict)."""
    workspace = sanitize_workspace(workspace)
    if session_id:
        s = _load_session(session_id)
        if s:
            if s.get("workspace") != workspace:
                raise ValueError("Chat session belongs to a different workspace")
            return session_id, s

    # Create new session
    sid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    title = first_message[:40] + ("..." if len(first_message) > 40 else "")
    session = {
        "id": sid,
        "workspace": workspace,
        "title": title,
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    _save_session(sid, session)
    return sid, session


@app.post("/api/chat/send")
async def chat_send(req: ChatSendRequest):
    """Send a message in a chat session and get AI response with RAG context."""
    now = datetime.now(timezone.utc).isoformat()
    req.workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(req.workspace)
    service = get_lightrag_service(req.workspace)

    # Critical section 1 (per-session lock): ensure session + read history.
    # Serializes against concurrent requests on the same session. For a brand
    # new session (no session_id) there is nothing to race against.
    if req.session_id:
        async with _get_session_lock(req.session_id):
            try:
                sid, session = _ensure_session(req.session_id, req.message, req.workspace)
            except ValueError as exc:
                raise HTTPException(409, str(exc))
            raw_history: list[dict] = [
                {"role": m["role"], "content": m["content"]}
                for m in session.get("messages", [])
            ]
    else:
        sid, session = _ensure_session(req.session_id, req.message, req.workspace)
        raw_history: list[dict] = [
            {"role": m["role"], "content": m["content"]}
            for m in session.get("messages", [])
        ]
    history = _sanitize_history_for_llm(raw_history)

    try:
        async with _rag_query_lock:
            context_result = await service.preview_context(
                req.message,
                mode=req.mode,
                top_k=req.top_k,
                chunk_top_k=req.chunk_top_k,
                enable_rerank=req.enable_rerank,
            )
    except Exception as e:
        context_result = {}
        logger.exception("LightRAG context retrieval failed")
        ai_text = f"[检索失败: {e}]"
        citations_data = []
    else:
        raw_like = {"data": context_result.get("data") or {}}
        citations_data = service._citations_from_raw(raw_like)
        if citations_data and not _citations_are_relevant(req.message, citations_data, history):
            logger.info("Retrieved citations rejected as irrelevant for query: {}", req.message)
            citations_data = []
        if citations_data:
            ai_text = await _generate_answer_text(
                req.message,
                citations_data,
                history,
                req.workspace,
            )

    if not citations_data:
        ai_text = "未检索到相关文档，知识库上下文不足，无法基于当前资料回答。"
    elif not ai_text.strip():
        ai_text = "未检索到相关文档，知识库上下文不足，无法基于当前资料回答。"

    # Save-guard: don't persist contaminated output
    if _is_contaminated_text(ai_text):
        logger.warning(
            f"Contaminated output detected in non-stream chat ({len(ai_text)} chars), "
            f"replacing with placeholder"
        )
        ai_text = "[此前的回答因技术原因丢失，已清理]"

    # Defensively strip any trailing "引用文档" section the LLM may have appended
    ai_text = _strip_lightrag_noise(_strip_citation_section(ai_text))
    citations = _format_chat_citations(citations_data)
    evidence = _build_evidence_chain(context_result.get("data") or {}, citations) if citations_data else EvidenceChain()

    # Build response messages
    user_msg = ChatMessage(role="user", content=req.message, timestamp=now)
    assistant_msg = ChatMessage(
        role="assistant",
        content=ai_text,
        timestamp=now,
        citations=citations,
        evidence=evidence,
    )

    # Critical section 2 (per-session lock): append messages + persist. Re-load
    # the latest session from disk first so concurrent appends on the same
    # session are not clobbered.
    async with _get_session_lock(sid):
        latest = _load_session(sid)
        if latest is not None:
            session = latest
        session["messages"].append({"role": "user", "content": req.message, "timestamp": now})
        session["messages"].append({
            "role": "assistant",
            "content": ai_text,
            "timestamp": now,
            "citations": _dump_chat_citations(citations),
            "evidence": evidence.model_dump(),
        })
        session["updated_at"] = now
        _save_session(sid, session)

    return ChatSendResponse(
        session_id=sid,
        user_message=user_msg,
        assistant_message=assistant_msg,
        citations=citations,
        evidence=evidence,
    )


@app.post("/api/chat/send/stream")
async def chat_send_stream(req: ChatSendRequest):
    """Send a message and stream the AI response via SSE."""
    now = datetime.now(timezone.utc).isoformat()
    req.workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(req.workspace)
    service = get_lightrag_service(req.workspace)

    # Critical section 1 (per-session lock): ensure session + read history.
    if req.session_id:
        async with _get_session_lock(req.session_id):
            try:
                sid, session = _ensure_session(req.session_id, req.message, req.workspace)
            except ValueError as exc:
                raise HTTPException(409, str(exc))
            raw_history: list[dict] = [
                {"role": m["role"], "content": m["content"]}
                for m in session.get("messages", [])
            ]
    else:
        sid, session = _ensure_session(req.session_id, req.message, req.workspace)
        raw_history: list[dict] = [
            {"role": m["role"], "content": m["content"]}
            for m in session.get("messages", [])
        ]
    history = _sanitize_history_for_llm(raw_history)

    try:
        async with _rag_query_lock:
            context_result = await service.preview_context(
                req.message,
                mode=req.mode,
                top_k=req.top_k,
                chunk_top_k=req.chunk_top_k,
                enable_rerank=req.enable_rerank,
            )
    except Exception as e:
        logger.exception("LightRAG context retrieval failed")
        context_result = {}

    raw_like = {"data": context_result.get("data") or {}}
    citations_data = service._citations_from_raw(raw_like)
    if citations_data and not _citations_are_relevant(req.message, citations_data, history):
        logger.info("Retrieved citations rejected as irrelevant for query: {}", req.message)
        citations_data = []
    citations = _format_chat_citations(citations_data)
    evidence = _build_evidence_chain(context_result.get("data") or {}, citations) if citations_data else EvidenceChain()

    async def event_generator():
        full_text = ""

        # First: send citations + graph hit nodes as an event
        citations_payload = {
            "citations": _dump_chat_citations(citations),
            "graph_nodes": [node.id for node in evidence.nodes],
            "graph_edges": [edge.model_dump() for edge in evidence.edges],
            "evidence": evidence.model_dump(),
        }
        yield f"event: citations\ndata: {json.dumps(citations_payload, ensure_ascii=False)}\n\n"

        try:
            try:
                full_text = await _generate_answer_text(
                    req.message,
                    citations_data,
                    history,
                    req.workspace,
                )
                for i in range(0, len(full_text), 80):
                    token = full_text[i:i + 80]
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

                # Final: done with session info
                yield f"event: done\ndata: {json.dumps({'session_id': sid, 'content': full_text}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
                full_text = full_text or f"[回答生成失败: {e}]"
        finally:
            # Persist the session no matter how the stream ends: normal
            # completion, an exception, or a client disconnect (GeneratorExit —
            # a BaseException, NOT caught by `except Exception` above). Saving
            # partial content on disconnect is preferable to losing the turn.
            # Best-effort: never let cleanup mask the propagating
            # GeneratorExit/exception (swallow only Exception, not BaseException).
            try:
                if _is_contaminated_text(full_text):
                    logger.warning(
                        f"Contaminated output detected ({len(full_text)} chars), "
                        f"replacing with placeholder before saving to session"
                    )
                    full_text = "[此前的回答因技术原因丢失，已清理]"

                # Critical section 2 (per-session lock): append + persist. Re-load
                # the latest session so concurrent appends are not clobbered.
                async with _get_session_lock(sid):
                    latest = _load_session(sid)
                    if latest is not None:
                        session = latest
                    session["messages"].append({"role": "user", "content": req.message, "timestamp": now})
                    session["messages"].append({
                        "role": "assistant",
                        "content": full_text,
                        "timestamp": now,
                        "citations": _dump_chat_citations(citations),
                        "evidence": evidence.model_dump(),
                    })
                    session["updated_at"] = now
                    _save_session(sid, session)
            except Exception:
                logger.exception("Failed to persist streaming chat session")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/chat/sessions", response_model=list[ChatSessionListItem])
async def list_chat_sessions(workspace: str = Query(DEFAULT_WORKSPACE)):
    """List chat sessions owned by one workspace."""
    workspace = sanitize_workspace(workspace)
    sessions = _list_sessions(workspace)
    return [
        ChatSessionListItem(
            id=s["id"],
            workspace=s.get("workspace", DEFAULT_WORKSPACE),
            title=s.get("title", ""),
            message_count=len(s.get("messages", [])),
            created_at=s.get("created_at", ""),
            updated_at=s.get("updated_at", ""),
        )
        for s in sessions
    ]


@app.get("/api/chat/sessions/{session_id}", response_model=ChatSession)
async def get_chat_session(
    session_id: str,
    workspace: str = Query(DEFAULT_WORKSPACE),
):
    """Get a single chat session with full messages."""
    if not _SESSION_ID_RE.fullmatch(session_id.lower()):
        raise HTTPException(400, "Invalid session id")
    s = _load_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if s.get("workspace") != sanitize_workspace(workspace):
        raise HTTPException(404, "Session not found in current workspace")
    return ChatSession(
        id=s["id"],
        workspace=s.get("workspace", DEFAULT_WORKSPACE),
        title=s.get("title", ""),
        messages=[ChatMessage(**m) for m in s.get("messages", [])],
        created_at=s.get("created_at", ""),
        updated_at=s.get("updated_at", ""),
    )


@app.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    workspace: str = Query(DEFAULT_WORKSPACE),
):
    """Delete a chat session."""
    try:
        path = _session_path(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    session = _load_session(session_id)
    if session and session.get("workspace") != sanitize_workspace(workspace):
        raise HTTPException(404, "Session not found in current workspace")
    if path.exists():
        path.unlink()
        logger.info(f"Deleted chat session: {session_id}")
    return {"deleted": session_id}


class ChatSessionCreateRequest(BaseModel):
    workspace: str = DEFAULT_WORKSPACE


@app.post("/api/chat/sessions")
async def create_chat_session(req: ChatSessionCreateRequest):
    """Create a new empty chat session."""
    workspace = sanitize_workspace(req.workspace)
    sid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    session = {
        "id": sid,
        "workspace": workspace,
        "title": "新对话",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    _save_session(sid, session)
    return {
        "id": sid,
        "workspace": workspace,
        "title": "新对话",
        "message_count": 0,
        "created_at": now,
        "updated_at": now,
    }


# --- System Stats ---

def _dir_size(path: str | Path) -> int:
    """Recursively compute the total size (bytes) of a directory (cross-platform)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(str(path)):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                continue
    return total


def _format_bytes(num: int) -> str:
    """Format a byte count into a human-readable string (e.g. '12.3 MB')."""
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


@app.get("/api/system/stats")
async def system_stats(workspace: str = Query(DEFAULT_WORKSPACE)):
    """Return LightRAG knowledge base and system statistics."""
    cfg = get_config()
    service = get_lightrag_service(workspace)
    stats = await service.stats()
    graph = service.read_graph(limit=1, include_isolated=False)
    graph_meta = graph.get("metadata") or {}
    embed_model = get_runtime_model_config(cfg)["embedding"]["model"]

    return {
        "doc_count": stats["doc_count"],
        "uploaded_doc_count": stats["uploaded_doc_count"],
        "chunk_count": stats["chunk_count"],
        "graph_nodes": graph_meta.get("total_nodes", 0),
        "graph_edges": graph_meta.get("total_edges", 0),
        "embed_model": embed_model,
        "embed_dim": stats["embed_dim"],
        "workspace": stats["workspace"],
        "lightrag_dir": stats["lightrag_dir"],
        "lightrag_dir_size": _format_bytes(stats["lightrag_dir_size"]),
    }


# --- Health ---

@app.get("/api/graph")
async def get_graph(limit: int = Query(200, ge=1, le=1000), workspace: str = Query(DEFAULT_WORKSPACE)):
    """Return LightRAG entity-relation graph extracted from GraphML."""
    workspace = sanitize_workspace(workspace)
    _ensure_workspace_available(workspace)
    return get_lightrag_service(workspace).read_graph(limit=limit, include_isolated=True)


def _graph_payload_without_empty(data: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        out[key] = value
    return out


async def _read_governance_reference_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".json", ".yaml", ".yml", ".csv"}:
        raise HTTPException(400, "Only .txt/.md/.json/.yaml/.yml/.csv reference files are supported")
    raw = await file.read()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(400, "Reference file text encoding is not supported")


def _extract_json_object(text: str) -> Any:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")
    candidates = [i for i in (start_obj, start_arr) if i >= 0]
    if not candidates:
        raise ValueError("No JSON found in model response")
    start = min(candidates)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        raise ValueError("No complete JSON found in model response")
    return json.loads(cleaned[start:end + 1])


async def _generate_graph_suggestions(req: GraphSuggestRequest) -> GraphSuggestResponse:
    service = get_lightrag_service(req.workspace)
    cfg = service.load_graph_governance()
    graph = service.read_graph(limit=req.limit, include_isolated=True)
    refs = service.graph_reference_bundle(max_chars=12000)
    graph_sample = {
        "nodes": graph.get("nodes", [])[: req.limit],
        "edges": graph.get("edges", [])[: req.limit],
        "metadata": graph.get("metadata") or {},
    }
    system_prompt = (
        "你是知识图谱审校助手。你只输出 JSON，不输出解释性文字。"
        "你的任务是根据抽取规则、术语表、参考文件和当前图谱，提出需要人工确认的图谱变更建议。"
        "允许的 action 只有: create_entity, edit_entity, delete_entity, create_relation, "
        "edit_relation, delete_relation, merge_entities。"
        "不要直接修改图谱。不要提出没有依据的建议。"
        "输出格式必须是 {\"changes\": [...]}。"
    )
    user_prompt = {
        "instruction": req.instruction,
        "allowed_entity_types": cfg.get("entity_types") or [],
        "allowed_relation_types": cfg.get("relation_types") or [],
        "extraction_prompt": cfg.get("extraction_prompt") or "",
        "aliases_text": cfg.get("aliases_text") or "",
        "reference_material": refs,
        "current_graph": graph_sample,
        "change_schema": {
            "action": "create_entity | edit_entity | delete_entity | create_relation | edit_relation | delete_relation | merge_entities",
            "reason": "为什么建议这样改",
            "entity_name": "实体名，实体相关操作使用",
            "source_entity": "关系起点",
            "target_entity": "关系终点或合并目标",
            "source_entities": ["合并源实体列表"],
            "entity_data": {"description": "新实体描述", "entity_type": "实体类型"},
            "relation_data": {"description": "关系描述", "keywords": "关系关键词", "weight": 1.0},
            "updated_data": {"description": "修改后的描述", "entity_type": "修改后的类型", "entity_name": "可选新名称"},
            "target_entity_data": {"description": "合并后实体描述", "entity_type": "合并后类型"},
        },
    }
    cfg = get_config()
    runtime_chat = get_runtime_model_config(cfg)["chat"]
    backend = SiliconFlowBackend(
        {
            "base_url": runtime_chat["base_url"],
            "api_key": runtime_chat["api_key"],
            "chat_model": runtime_chat["model"],
            "timeout": runtime_chat.get("timeout", 90),
        }
    )
    try:
        response = await backend.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
            temperature=0.1,
            top_p=0.8,
            max_tokens=min(int(runtime_chat.get("max_tokens", 4096)), 4096),
        )
    finally:
        await backend.close()

    raw_text = response.content
    try:
        payload = _extract_json_object(raw_text)
    except Exception as exc:
        raise HTTPException(500, f"Failed to parse graph suggestions as JSON: {exc}")
    raw_changes = payload.get("changes") if isinstance(payload, dict) else payload
    if not isinstance(raw_changes, list):
        raw_changes = []
    changes = []
    for item in raw_changes[:80]:
        if not isinstance(item, dict):
            continue
        try:
            changes.append(GraphChange(**item))
        except Exception:
            logger.warning("Skipping invalid graph suggestion: {}", item)
    return GraphSuggestResponse(changes=changes, raw_text=raw_text)


async def _apply_graph_change(service, change: GraphChange) -> dict[str, Any]:
    action = change.action.strip()
    if action == "create_entity":
        return await service.create_graph_entity(change.entity_name, change.entity_data)
    if action == "edit_entity":
        return await service.edit_graph_entity(change.entity_name, change.updated_data)
    if action == "delete_entity":
        return await service.delete_graph_entity(change.entity_name)
    if action == "create_relation":
        return await service.create_graph_relation(change.source_entity, change.target_entity, change.relation_data)
    if action == "edit_relation":
        return await service.edit_graph_relation(change.source_entity, change.target_entity, change.updated_data)
    if action == "delete_relation":
        return await service.delete_graph_relation(change.source_entity, change.target_entity)
    if action == "merge_entities":
        return await service.merge_graph_entities(
            change.source_entities,
            change.target_entity,
            change.target_entity_data,
        )
    raise HTTPException(400, f"Unsupported graph change action: {action}")


@app.get("/api/graph/governance/config", response_model=GraphGovernanceConfig)
async def get_graph_governance_config(workspace: str = Query(DEFAULT_WORKSPACE)):
    workspace = sanitize_workspace(workspace)
    return get_lightrag_service(workspace).load_graph_governance()


@app.get("/api/graph/rule-templates", response_model=list[GraphRuleTemplate])
async def list_graph_rule_templates(workspace: str = Query(DEFAULT_WORKSPACE)):
    workspace = sanitize_workspace(workspace)
    return get_lightrag_service(workspace).list_graph_rule_templates()


@app.post("/api/graph/rule-templates", response_model=GraphRuleTemplate)
async def save_graph_rule_template(req: GraphRuleTemplate):
    try:
        return get_lightrag_service(DEFAULT_WORKSPACE).save_graph_rule_template(req.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/api/graph/rule-templates/{template_id}")
async def delete_graph_rule_template(template_id: str):
    try:
        return get_lightrag_service(DEFAULT_WORKSPACE).delete_graph_rule_template(template_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except KeyError:
        raise HTTPException(404, "Graph rule template not found")


@app.post("/api/graph/governance/apply-template", response_model=GraphGovernanceConfig)
async def apply_graph_rule_template(req: GraphRuleTemplateApplyRequest):
    workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(workspace)
    try:
        return get_lightrag_service(workspace).apply_graph_rule_template(req.template_id)
    except KeyError:
        raise HTTPException(404, "Graph rule template not found")


@app.put("/api/graph/governance/config", response_model=GraphGovernanceConfig)
async def update_graph_governance_config(req: GraphGovernanceUpdate):
    workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(workspace)
    service = get_lightrag_service(workspace)
    return service.save_graph_governance(
        {
            "rule_template_id": req.rule_template_id,
            "rule_template_name": req.rule_template_name,
            "extraction_mode": req.extraction_mode,
            "allow_other_entity_type": req.allow_other_entity_type,
            "entity_types": [item.strip() for item in req.entity_types if item.strip()],
            "relation_types": [item.strip() for item in req.relation_types if item.strip()],
            "aliases_text": req.aliases_text,
            "extraction_prompt": req.extraction_prompt,
        }
    )


@app.post("/api/graph/governance/references")
async def upload_graph_reference(file: UploadFile = File(...), workspace: str = Query(DEFAULT_WORKSPACE)):
    workspace = sanitize_workspace(workspace)
    _ensure_workspace_available(workspace)
    content = await _read_governance_reference_upload(file)
    if not content.strip():
        raise HTTPException(400, "Reference file is empty")
    item = get_lightrag_service(workspace).add_graph_reference(file.filename or "reference.txt", content)
    return item


@app.delete("/api/graph/governance/references/{ref_id}")
async def delete_graph_reference(ref_id: str, workspace: str = Query(DEFAULT_WORKSPACE)):
    workspace = sanitize_workspace(workspace)
    _ensure_workspace_available(workspace)
    try:
        return get_lightrag_service(workspace).delete_graph_reference(ref_id)
    except KeyError:
        raise HTTPException(404, "Reference file not found")


@app.post("/api/graph/governance/suggest", response_model=GraphSuggestResponse)
async def suggest_graph_changes(req: GraphSuggestRequest):
    req.workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(req.workspace)
    async with _rag_query_lock:
        return await _generate_graph_suggestions(req)


@app.post("/api/graph/governance/apply")
async def apply_graph_changes(req: GraphApplyChangesRequest):
    workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(workspace)
    service = get_lightrag_service(workspace)
    results = []
    async with _rag_query_lock:
        for change in req.changes:
            try:
                result = await _apply_graph_change(service, change)
                results.append({"action": change.action, "status": "ok", "result": result})
            except Exception as exc:
                logger.exception("Failed to apply graph change {}", change.model_dump())
                results.append({"action": change.action, "status": "error", "error": str(exc)})
    return {"workspace": workspace, "results": results}


@app.post("/api/graph/entities")
async def create_graph_entity(req: GraphEntityCreateRequest):
    workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(workspace)
    data = _graph_payload_without_empty(
        {
            "description": req.description,
            "entity_type": req.entity_type,
            "source_id": req.source_id,
            "file_path": req.file_path,
        }
    )
    async with _rag_query_lock:
        try:
            result = await get_lightrag_service(workspace).create_graph_entity(req.entity_name, data)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return {"status": "success", "data": result}


@app.put("/api/graph/entities")
async def update_graph_entity(req: GraphEntityUpdateRequest):
    workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(workspace)
    async with _rag_query_lock:
        try:
            result = await get_lightrag_service(workspace).edit_graph_entity(
                req.entity_name,
                _graph_payload_without_empty(req.updated_data),
                allow_rename=req.allow_rename,
                allow_merge=req.allow_merge,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return {"status": "success", "data": result}


@app.delete("/api/graph/entities/{entity_name}")
async def delete_graph_entity(entity_name: str, workspace: str = Query(DEFAULT_WORKSPACE)):
    workspace = sanitize_workspace(workspace)
    _ensure_workspace_available(workspace)
    async with _rag_query_lock:
        try:
            result = await get_lightrag_service(workspace).delete_graph_entity(entity_name)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return {"status": "success", "data": result}


@app.post("/api/graph/relations")
async def create_graph_relation(req: GraphRelationCreateRequest):
    workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(workspace)
    data = _graph_payload_without_empty(
        {
            "description": req.description,
            "keywords": req.keywords,
            "weight": req.weight,
            "source_id": req.source_id,
            "file_path": req.file_path,
        }
    )
    async with _rag_query_lock:
        try:
            result = await get_lightrag_service(workspace).create_graph_relation(
                req.source_entity,
                req.target_entity,
                data,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return {"status": "success", "data": result}


@app.put("/api/graph/relations")
async def update_graph_relation(req: GraphRelationUpdateRequest):
    workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(workspace)
    async with _rag_query_lock:
        try:
            result = await get_lightrag_service(workspace).edit_graph_relation(
                req.source_entity,
                req.target_entity,
                _graph_payload_without_empty(req.updated_data),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return {"status": "success", "data": result}


@app.delete("/api/graph/relations")
async def delete_graph_relation(req: GraphRelationDeleteRequest):
    workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(workspace)
    async with _rag_query_lock:
        try:
            result = await get_lightrag_service(workspace).delete_graph_relation(
                req.source_entity,
                req.target_entity,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return {"status": "success", "data": result}


@app.post("/api/graph/entities/merge")
async def merge_graph_entities(req: GraphEntityMergeRequest):
    workspace = sanitize_workspace(req.workspace)
    _ensure_workspace_available(workspace)
    async with _rag_query_lock:
        try:
            result = await get_lightrag_service(workspace).merge_graph_entities(
                req.source_entities,
                req.target_entity,
                req.target_entity_data,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    return {"status": "success", "data": result}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


def main():
    """Entry point for `python -m src.api.server`."""
    uvicorn.run(
        "src.api.server:app",
        host=os.environ.get("TDX_HOST", "127.0.0.1"),
        port=8101,
        reload=True,
    )


if __name__ == "__main__":
    main()
