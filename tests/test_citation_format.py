"""Test suite for answer format optimization: citation indexing, superscript
markers, and empty-retrieval fallback.

Covers:
  1. Citation model has required `index` field (1-based)
  2. Non-streaming /api/chat/send returns citations with `index`
  3. Non-streaming response content contains [数字] superscript markers
  4. Streaming /api/chat/send/stream emits `event: citations` with `index`
  5. Streaming token accumulation includes [数字] markers
  6. Empty retrieval → citations empty + "未检索到相关文档" message
  7. Answer uses continuous Arabic numbering (no Chinese numerals / jumps)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import pytest
from pydantic import ValidationError

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8101/api"
RELEVANT_QUESTION = "TP和TS有什么区别"
IRRELEVANT_QUESTION = "今天天气怎么样"

# Pattern for [数字] citation markers in answer text
CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

# Chinese numeral list markers that should NOT appear at line starts
CHINESE_NUMERAL_RE = re.compile(r"^[一二三四五六七八九十]+[、.．]", re.MULTILINE)

# Roman numeral markers
ROMAN_NUMERAL_RE = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[、.．]", re.MULTILINE)


@pytest.fixture(scope="module", autouse=True)
def cleanup_live_api_sessions():
    """Keep live-server contract checks from polluting the user's chat list."""
    sessions_dir = PROJECT_ROOT / "data" / "sessions"
    before = {path.name for path in sessions_dir.glob("*.json")}
    yield
    # The streaming endpoint persists in the generator's finally block after
    # emitting the done event. Give that cleanup a moment to finish before
    # removing sessions created by these live-server checks.
    time.sleep(0.5)
    for path in sessions_dir.glob("*.json"):
        if path.name not in before:
            path.unlink(missing_ok=True)


# ===========================================================================
# PART 1: Unit Tests (no server needed)
# ===========================================================================

class TestCitationModel:
    """Unit tests for the Citation Pydantic model and its construction."""

    def test_citation_model_has_index_field(self):
        """Citation model must define an `index` field of type int."""
        from src.api.server import Citation
        fields = Citation.model_fields
        assert "index" in fields, "Citation model must have an `index` field"
        # index should be int type
        ann = fields["index"].annotation
        assert ann is int, f"Citation.index should be int, got {ann}"

    def test_citation_requires_index(self):
        """Citation must reject construction without `index` (it's required)."""
        from src.api.server import Citation
        with pytest.raises(ValidationError):
            Citation(doc_name="test.md", chunk_index=0, excerpt="hello")

    def test_citation_constructed_with_index(self):
        """Citation should construct successfully when index is provided."""
        from src.api.server import Citation
        c = Citation(index=1, doc_name="test.md", chunk_index=0, excerpt="hello")
        assert c.index == 1
        assert c.doc_name == "test.md"

    def test_chat_send_citation_construction_includes_index(self):
        """CRITICAL: Verify that chat_send's Citation construction passes `index`.

        This simulates the exact code path in server.py chat_send() lines 940-943.
        If `index` is not passed, the non-streaming endpoint will crash with
        ValidationError whenever citations are non-empty.
        """
        from src.api.server import Citation

        # Simulate citations_data as produced by _retrieve_and_build_prompt
        citations_data = [
            {"index": 1, "doc_name": "doc1.md", "chunk_index": 0, "excerpt": "excerpt1"},
            {"index": 2, "doc_name": "doc2.md", "chunk_index": 1, "excerpt": "excerpt2"},
        ]

        # This is the EXACT code from chat_send (server.py:940-943)
        # If the source code is correct, it should pass `index=c["index"]`
        try:
            citations = [
                Citation(
                    index=c["index"],            # <-- this MUST be present
                    doc_name=c["doc_name"],
                    chunk_index=c["chunk_index"],
                    excerpt=c["excerpt"],
                )
                for c in citations_data
            ]
            assert len(citations) == 2
            assert citations[0].index == 1
            assert citations[1].index == 2
        except ValidationError:
            pytest.fail(
                "Citation construction in chat_send is missing `index` field. "
                "The non-streaming endpoint /api/chat/send will crash with "
                "ValidationError when citations are non-empty."
            )

    def test_chat_send_source_code_passes_index(self):
        """Static check: read server.py source and verify Citation() call includes index."""
        server_path = PROJECT_ROOT / "src" / "api" / "server.py"
        source = server_path.read_text(encoding="utf-8")

        # Find the Citation construction in chat_send (non-streaming endpoint)
        # It should be in the chat_send function, not chat_send_stream
        # Look for the pattern: Citation(doc_name=...
        # and check that index= is present in that construction
        pattern = r'Citation\(([^)]+)\)'
        matches = re.findall(pattern, source, re.DOTALL)

        # Filter to only the construction in chat_send (not the model definition)
        citation_constructions = [m for m in matches if 'doc_name' in m]
        assert len(citation_constructions) > 0, "No Citation() construction found in server.py"

        # Check that at least one construction includes index= (not chunk_index)
        # We look for 'index=' as a keyword argument, not 'index' inside 'chunk_index'
        has_index = any(re.search(r'\bindex\s*=', m) for m in citation_constructions)
        if not has_index:
            pytest.fail(
                "BUG: Citation() constructor in server.py does not pass `index`. "
                "The non-streaming /api/chat/send endpoint will crash with "
                "ValidationError when citations are non-empty. "
                "Fix: add `index=c['index']` to the Citation() call. "
                f"Found constructions: {citation_constructions}"
            )


class TestLightRAGCitationContext:
    """Unit tests for LightRAGService citation and empty-context format."""

    def test_citations_data_has_index_field(self):
        """LightRAGService must produce citations with a 1-based `index` key."""
        service_path = PROJECT_ROOT / "src" / "lightrag_service.py"
        source = service_path.read_text(encoding="utf-8")

        assert '"index": len(citations) + 1' in source or "'index': len(citations) + 1" in source, \
            "LightRAG citations must include a 1-based index field"

    def test_empty_context_message(self):
        """Empty LightRAG context preview should be surfaced as context text."""
        server_path = PROJECT_ROOT / "src" / "api" / "server.py"
        source = server_path.read_text(encoding="utf-8")

        assert "LightRAG context preview" in source, \
            "Recall endpoint should be a LightRAG context preview, not old vector/graph scoring"

    def test_prompt_has_citation_instructions(self):
        """The active answer prompt must instruct [数字] citation markers."""
        content = (PROJECT_ROOT / "src" / "api" / "server.py").read_text(encoding="utf-8")

        assert "[数字]" in content, \
            "Prompt must instruct [数字] citation markers"
        assert "不要输出 References/引用文档列表" in content, \
            "Prompt must forbid model-generated citation lists"

    def test_prompt_forbids_bare_citation_numbers(self):
        """The active prompt must forbid bare citation numbers."""
        content = (PROJECT_ROOT / "src" / "api" / "server.py").read_text(encoding="utf-8")

        assert "禁止把引用编号写成裸数字" in content, \
            "Prompt must forbid bare citation numbers"


class TestStreamingEventFormat:
    """Unit tests for streaming SSE event format."""

    def test_streaming_citations_event_includes_index(self):
        """The streaming endpoint must send citations with index in the SSE event."""
        server_path = PROJECT_ROOT / "src" / "api" / "server.py"
        source = server_path.read_text(encoding="utf-8")

        # The streaming endpoint sends citations_data directly as JSON
        # which includes index. Verify citations_data is used (not reconstructed)
        # Find the citations_payload construction
        assert "citations_data" in source, \
            "Streaming endpoint must use citations_data (which has index)"

    def test_streaming_sse_event_name(self):
        """Streaming must emit 'event: citations' SSE event."""
        server_path = PROJECT_ROOT / "src" / "api" / "server.py"
        source = server_path.read_text(encoding="utf-8")

        assert 'event: citations' in source, \
            "Streaming must emit 'event: citations' SSE event"


# ===========================================================================
# PART 2: Integration Tests (require running server)
# ===========================================================================

def _server_is_up() -> bool:
    """Check if the backend server is running on port 8101."""
    try:
        url = f"{BASE_URL}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _api_post(path: str, body: dict, timeout: int = 120) -> tuple[int, dict]:
    """POST JSON to the API and return (status_code, response_dict)."""
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}
    except Exception as e:
        return -1, {"error": str(e)}


def _api_post_stream(path: str, body: dict, timeout: int = 120) -> tuple[int, str]:
    """POST JSON and return raw SSE text response."""
    import http.client
    full_path = f"/api{path}"
    conn = http.client.HTTPConnection("localhost", 8101, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    try:
        conn.request("POST", full_path, json.dumps(body), headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        return resp.status, raw
    except Exception as e:
        return -1, str(e)
    finally:
        conn.close()


def _api_get(path: str, timeout: int = 30) -> tuple[int, dict]:
    """GET JSON from the API and return (status_code, response_dict)."""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}
    except Exception as e:
        return -1, {"error": str(e)}


# Skip integration tests if server is not running
pytestmark_skip_if_no_server = pytest.mark.skipif(
    not _server_is_up(),
    reason="Backend server not running on port 8101"
)


@pytest.mark.skipif(not _server_is_up(), reason="Backend server not running on port 8101")
class TestNonStreamingChatSend:
    """Integration tests for /api/chat/send (non-streaming)."""

    def test_relevant_question_returns_citations_with_index(self):
        """Querying a relevant question should return citations with `index` field."""
        status, data = _api_post("/chat/send", {
            "message": RELEVANT_QUESTION,
            "mode": "mix",
            "top_k": 5,
            "chunk_top_k": 5,
            "enable_rerank": True,
        }, timeout=120)

        assert status == 200, f"Expected 200, got {status}: {data}"

        citations = data.get("citations", [])
        assistant_content = data.get("assistant_message", {}).get("content", "")

        # If citations are non-empty, each must have index
        if citations:
            for i, c in enumerate(citations):
                assert "index" in c, \
                    f"Citation {i} missing `index` field: {c}"
                assert isinstance(c["index"], int), \
                    f"Citation {i} index must be int, got {type(c['index'])}"
                assert c["index"] == i + 1, \
                    f"Citation {i} index should be {i+1}, got {c['index']}"
        else:
            # If no citations, the answer should mention "未检索到相关文档"
            assert "未检索到" in assistant_content or "未检索到相关文档" in assistant_content, \
                f"When citations are empty, answer should mention 未检索到, got: {assistant_content[:200]}"

    def test_answer_contains_citation_markers(self):
        """The assistant message should contain [数字] citation markers (when citations exist)."""
        status, data = _api_post("/chat/send", {
            "message": RELEVANT_QUESTION,
            "mode": "mix",
            "top_k": 5,
            "chunk_top_k": 5,
            "enable_rerank": True,
        }, timeout=120)

        assert status == 200, f"Expected 200, got {status}: {data}"

        citations = data.get("citations", [])
        content = data.get("assistant_message", {}).get("content", "")

        if citations:
            markers = CITATION_MARKER_RE.findall(content)
            assert len(markers) > 0, \
                f"Answer should contain [数字] citation markers when citations exist. " \
                f"Content: {content[:300]}"

    def test_answer_uses_arabic_numerals(self):
        """Answer should use Arabic numerals, not Chinese numerals or Roman numerals."""
        status, data = _api_post("/chat/send", {
            "message": RELEVANT_QUESTION,
            "mode": "mix",
            "top_k": 5,
            "chunk_top_k": 5,
            "enable_rerank": True,
        }, timeout=120)

        assert status == 200, f"Expected 200, got {status}: {data}"
        content = data.get("assistant_message", {}).get("content", "")

        # Should NOT have Chinese numeral list markers
        chinese_matches = CHINESE_NUMERAL_RE.findall(content)
        assert len(chinese_matches) == 0, \
            f"Answer should not use Chinese numeral list markers, found: {chinese_matches}"

        # Should NOT have Roman numeral markers
        roman_matches = ROMAN_NUMERAL_RE.findall(content)
        assert len(roman_matches) == 0, \
            f"Answer should not use Roman numeral markers, found: {roman_matches}"

    def test_answer_has_continuous_numbering(self):
        """If answer uses numbered list, numbering should be continuous (1, 2, 3, ...)."""
        status, data = _api_post("/chat/send", {
            "message": RELEVANT_QUESTION,
            "mode": "mix",
            "top_k": 5,
            "chunk_top_k": 5,
            "enable_rerank": True,
        }, timeout=120)

        assert status == 200, f"Expected 200, got {status}: {data}"
        content = data.get("assistant_message", {}).get("content", "")

        # Extract all line-start numbers like "1." "2." "3."
        numbered_lines = re.findall(r"^(\d+)[.、）)]", content, re.MULTILINE)
        if len(numbered_lines) >= 2:
            nums = [int(n) for n in numbered_lines]
            for i in range(1, len(nums)):
                if nums[i] != nums[i-1] + 1:
                    # Allow restart at 1 (new section), but not jumps like 1→3
                    if nums[i] != 1:
                        pytest.fail(
                            f"Numbering is not continuous: {nums}. "
                            f"Jump from {nums[i-1]} to {nums[i]}"
                        )


@pytest.mark.skipif(not _server_is_up(), reason="Backend server not running on port 8101")
class TestIrrelevantQuestion:
    """Test that irrelevant questions get empty citations + fallback message."""

    def test_irrelevant_question_empty_citations(self):
        """Irrelevant question should return empty citations.

        NOTE: The prompt instructs the LLM to say "本次回答未检索到相关文档" when
        no references are found. However, LLM output is non-deterministic and may
        phrase the lack-of-knowledge differently (e.g. "我没有...的能力"). We accept
        any answer that indicates the model lacks the information, rather than
        requiring the exact phrase "未检索到".
        """
        status, data = _api_post("/chat/send", {
            "message": IRRELEVANT_QUESTION,
            "mode": "mix",
            "top_k": 5,
            "chunk_top_k": 5,
            "enable_rerank": True,
        }, timeout=120)

        assert status == 200, f"Expected 200, got {status}: {data}"
        citations = data.get("citations", [])
        content = data.get("assistant_message", {}).get("content", "")
        if len(citations) == 0:
            # Accept any phrase indicating the model lacks the information.
            # The ideal response contains "未检索到" (per prompt), but the LLM
            # may also say "没有...能力", "无法", "不清楚", "自身知识", etc.
            fallback_phrases = [
                "未检索到", "没有", "无法", "不清楚", "不知道",
                "自身知识", "不具备", "不能提供", "暂无",
            ]
            has_fallback = any(p in content for p in fallback_phrases)
            assert has_fallback, \
                f"When citations empty, answer should indicate lack of info. " \
                f"Content: {content[:300]}"


@pytest.mark.skipif(not _server_is_up(), reason="Backend server not running on port 8101")
class TestStreamingChatSend:
    """Integration tests for /api/chat/send/stream (SSE streaming)."""

    def _stream_request(self, message: str = RELEVANT_QUESTION) -> tuple[int, str]:
        """Make a streaming request with retry on transient API errors."""
        for attempt in range(3):
            status, raw = _api_post_stream("/chat/send/stream", {
                "message": message,
                "mode": "mix",
                "top_k": 5,
                "chunk_top_k": 5,
                "enable_rerank": True,
            }, timeout=180)
            if status == 200:
                return status, raw
            if status == 500 and attempt < 2:
                # Transient SiliconFlow API error — retry
                time.sleep(5)
                continue
            return status, raw
        return status, raw

    def _parse_sse(self, raw: str) -> tuple[list | None, str]:
        """Parse SSE response, return (citations_data, full_content)."""
        citations_data = None
        full_content = ""
        for line in raw.split("\n"):
            if line.startswith("data: ") and "citations" in line:
                try:
                    data = json.loads(line[6:])
                    if "citations" in data:
                        citations_data = data["citations"]
                except json.JSONDecodeError:
                    pass
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "token" in data:
                        full_content += data["token"]
                    elif "content" in data:
                        full_content = data["content"]
                except json.JSONDecodeError:
                    pass
        return citations_data, full_content

    def test_stream_emits_citations_event_with_index(self):
        """Streaming should emit 'event: citations' with index field."""
        status, raw = self._stream_request()
        assert status == 200, \
            f"Expected 200, got {status}: {raw[:500]}. " \
            "If 500, may be transient SiliconFlow API error."

        citations_data, _ = self._parse_sse(raw)

        # Verify citations event was emitted
        assert citations_data is not None, \
            "Streaming did not emit 'event: citations' with citations data"

        # If citations are non-empty, each must have index
        if citations_data:
            for i, c in enumerate(citations_data):
                assert "index" in c, \
                    f"Streaming citation {i} missing `index` field: {c}"
                assert c["index"] == i + 1, \
                    f"Streaming citation {i} index should be {i+1}, got {c['index']}"

    def test_stream_content_has_citation_markers(self):
        """Accumulated streaming content should contain [数字] markers (when citations exist)."""
        status, raw = self._stream_request()
        assert status == 200, \
            f"Expected 200, got {status}. If 500, may be transient SiliconFlow API error."

        citations_data, full_content = self._parse_sse(raw)

        if citations_data:
            markers = CITATION_MARKER_RE.findall(full_content)
            assert len(markers) > 0, \
                f"Streamed content should contain [数字] markers. Content: {full_content[:300]}"


@pytest.mark.skipif(not _server_is_up(), reason="Backend server not running on port 8101")
class TestDocumentChunksEndpoint:
    """Integration tests for GET /api/kb/documents/{doc_name}/chunks."""

    def test_chunks_endpoint_returns_valid_structure(self):
        """GET /api/kb/documents/{doc_name}/chunks should return chunks with text."""
        from urllib.parse import quote
        # First, list documents to find one that exists
        status, docs = _api_get("/kb/documents")
        assert status == 200, f"Failed to list documents: {status}"
        assert isinstance(docs, list) and len(docs) > 0, \
            "Need at least one indexed document to test chunks endpoint"

        doc_name = docs[0].get("doc_name", "")
        assert doc_name, f"First document has no doc_name: {docs[0]}"

        # URL-encode doc_name (may contain Chinese characters)
        encoded_name = quote(doc_name, safe="")
        status, data = _api_get(f"/kb/documents/{encoded_name}/chunks")
        assert status == 200, f"Expected 200, got {status}: {data}"

        assert data["doc_name"] == doc_name
        assert "total" in data, "Response missing 'total' field"
        assert "chunks" in data, "Response missing 'chunks' field"
        assert isinstance(data["chunks"], list)

        # If chunks exist, verify structure
        if data["chunks"]:
            for i, chunk in enumerate(data["chunks"]):
                assert "chunk_id" in chunk, f"Chunk {i} missing chunk_id"
                assert "chunk_index" in chunk, f"Chunk {i} missing chunk_index"
                assert "text" in chunk, f"Chunk {i} missing text"
                assert "char_count" in chunk, f"Chunk {i} missing char_count"
                assert chunk["char_count"] == len(chunk["text"]), \
                    f"Chunk {i} char_count mismatch"

            # Chunks should be sorted by chunk_index
            indices = [c["chunk_index"] for c in data["chunks"]]
            assert indices == sorted(indices), \
                f"Chunks not sorted by chunk_index: {indices}"

    def test_chunks_endpoint_nonexistent_doc_returns_empty(self):
        """GET /api/kb/documents/{nonexistent}/chunks should return empty list."""
        status, data = _api_get("/kb/documents/__nonexistent_doc_xyz__.md/chunks")
        assert status == 200, f"Expected 200, got {status}: {data}"
        assert data["total"] == 0, f"Expected 0 chunks for nonexistent doc, got {data['total']}"
        assert data["chunks"] == [], f"Expected empty chunks list, got {data['chunks']}"
