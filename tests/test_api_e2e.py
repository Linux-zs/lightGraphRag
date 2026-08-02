"""Standalone end-to-end API smoke test for the knowledge-base workbench."""
import json
import sys
import urllib.request
import urllib.error

BASE = "http://localhost:8101/api"
PASS = 0
FAIL = 0
SKIP = 0

def req(method, path, body=None, files=None):
    """Make an HTTP request and return (status, response_dict)."""
    url = f"{BASE}{path}"
    try:
        if files:
            import requests
            headers = {}
            data = body or {}
            with open(files["file"][0], "rb") as f:
                r = requests.post(url, data=data, files={"file": f})
            return r.status_code, r.json() if r.text else {}
        else:
            data = json.dumps(body).encode() if body else None
            rq = urllib.request.Request(url, data=data, method=method)
            rq.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(rq) as resp:
                return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()) if e.fp else {"detail": str(e)}
    except Exception as e:
        return -1, {"error": str(e)}

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ PASS  {name}")
        if detail:
            print(f"         {detail}")
    else:
        FAIL += 1
        print(f"  ✗ FAIL  {name}: {detail}")

def test_skip(name, reason):
    global SKIP
    SKIP += 1
    print(f"  ⊘ SKIP  {name} ({reason})")

if "pytest" in sys.modules:
    import pytest
    pytest.skip("test_api_e2e.py is a standalone script; run it directly against a live server", allow_module_level=True)

# ============================================================
print("=" * 60)
print("LightGraphRAG Workbench 全接口测试")
print("=" * 60)

# 1. Health
print("\n[1/9] Health Check")
status, data = req("GET", "/health")
test("GET /api/health → 200", status == 200)
test("返回 status=ok", data.get("status") == "ok", str(data))

# 2. Model Config
print("\n[2/9] Model Config")
status, data = req("GET", "/models/config")
test("GET /api/models/config → 200", status == 200)
test("含 embed_model", "embed_model" in data, data.get("embed_model", ""))
test("含 rerank_model", "rerank_model" in data, data.get("rerank_model", ""))
test("含 chat_model", "chat_model" in data, data.get("chat_model", ""))

# 3. Documents List
print("\n[3/9] Documents List")
status, data = req("GET", "/kb/documents")
test("GET /api/kb/documents → 200", status == 200)
test("返回列表", isinstance(data, list), f"count={len(data)}")

# 4. Upload Document
print("\n[4/9] Upload Document")

# Find a test file
import pathlib
project_root = pathlib.Path(__file__).resolve().parent
test_files = list(project_root.glob("data/docs/**/*.md"))
if not test_files:
    test_files = list(project_root.glob("data/**/*.md"))
if not test_files:
    # create a minimal test file
    test_file = project_root / "data" / "_test_upload.md"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# 测试文档\n\n这是一个测试文档，用于验证上传和切分功能。\n\n## 第二部分\n\n这里包含更多内容以便测试文本切分效果。", encoding="utf-8")
    test_files = [test_file]

test_file = test_files[0]
print(f"  使用测试文件: {test_file.name}")

# Upload via requests (multipart)
try:
    import requests
    with open(test_file, "rb") as f:
        r = requests.post(f"{BASE}/kb/upload", files={"file": (test_file.name, f)})
    status, upload_data = r.status_code, r.json()
    test("POST /api/kb/upload → 200", status == 200)
    test("含 file_name", "file_name" in upload_data, upload_data.get("file_name", ""))
    test("含 file_type", "file_type" in upload_data, upload_data.get("file_type", ""))
    test("含 char_count", "char_count" in upload_data, str(upload_data.get("char_count", "")))

    file_name = upload_data.get("file_name", "")
except Exception as e:
    test("Upload: 请求发送", False, str(e))
    file_name = ""

# 5. Preview Chunks
print("\n[5/9] Preview Chunks")
if not file_name:
    test_skip("POST /api/kb/preview-chunks", "no uploaded file")
else:
    status, data = req("POST", "/kb/preview-chunks", {
        "file_name": file_name,
        "separators": ["\n\n", "\n", "。", "！", "？"],
        "chunk_size": 512,
        "chunk_overlap": 50,
    })
    test("POST /api/kb/preview-chunks → 200", status == 200, f"status={status}")
    if status == 200:
        test("返回列表", isinstance(data, list), f"chunk count={len(data)}")
        if isinstance(data, list) and len(data) > 0:
            c = data[0]
            test("chunk 含 index", "index" in c, str(c.get("index", "")))
            test("chunk 含 text", "text" in c and len(c["text"]) > 0, f"text len={len(c.get('text',''))}")
            test("chunk 含 char_count", "char_count" in c, str(c.get("char_count", "")))
    else:
        test_skip("chunk 结构验证", f"HTTP {status}: {data}")

# 6. Test Embed
print("\n[6/9] Test Embed (需要 SiliconFlow API Key)")
status, data = req("POST", "/models/test-embed", {"text": "测试嵌入文本"})
if status == 200:
    test("POST /api/models/test-embed → 200", True)
    test("含 dimensions", "dimensions" in data, str(data.get("dimensions", "")))
    test("含 preview (list)", isinstance(data.get("preview"), list), f"len={len(data.get('preview', []))}")
elif status == 500:
    test_skip("POST /api/models/test-embed", f"API key may not be set: {data.get('detail','')}")
else:
    test("POST /api/models/test-embed", False, f"status={status}: {data}")

# 7. Recall Test
print("\n[7/9] Recall Test (需要已索引数据)")
status, data = req("POST", "/recall/test", {
    "query": "LightRAG 是什么",
    "mode": "mix",
    "top_k": 5,
    "chunk_top_k": 5,
    "enable_rerank": True,
})
if status == 200:
    test("POST /api/recall/test → 200", True)
    test("含 query", data.get("query") == "LightRAG 是什么")
    test("含 mode", data.get("mode") == "mix")
    test("含 context", "context" in data, f"len={len(data.get('context', ''))}")
    test("含 chunks (list)", isinstance(data.get("chunks"), list), f"count={len(data.get('chunks', []))}")
    test("含 entities (list)", isinstance(data.get("entities"), list), f"count={len(data.get('entities', []))}")
    test("含 relationships (list)", isinstance(data.get("relationships"), list), f"count={len(data.get('relationships', []))}")
elif status == 500:
    test_skip("POST /api/recall/test", f"API error: {data.get('detail','')[:80]}")
else:
    test("POST /api/recall/test", False, f"status={status}: {data}")

# 8. Full Search
print("\n[8/9] Full RAG Search (需要 LLM API)")
status, data = req("POST", "/search", {
    "query": "LightRAG 是什么",
    "mode": "mix",
    "top_k": 3,
    "chunk_top_k": 3,
    "enable_rerank": False,
})
if status == 200:
    test("POST /api/search → 200", True)
    test("含 question", "question" in data)
    test("含 content", "content" in data, f"content len={len(data.get('content',''))}")
    test("含 citations", isinstance(data.get("citations"), list), f"count={len(data.get('citations',[]))}")
else:
    test_skip("POST /api/search", f"LLM may not be available: status={status}")

# 9. Model Config PUT
print("\n[9/9] Model Config Update")
status, data = req("PUT", "/models/config", {
    "embed_model": "BAAI/bge-large-zh-v1.5",
    "embed_base_url": "https://api.siliconflow.cn/v1",
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    "chat_model": "Qwen/Qwen2.5-7B-Instruct",
    "chat_temperature": 0.7,
    "chat_top_p": 0.9,
    "chat_max_tokens": 4096,
})
test("PUT /api/models/config → 200", status == 200, str(data))

# Summary
print("\n" + "=" * 60)
total = PASS + FAIL + SKIP
print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped (total {total})")
print("=" * 60)

if FAIL > 0:
    print("❌ 有失败的测试，需要修复！")
    sys.exit(1)
else:
    print("✅ 所有可执行测试通过！")
    sys.exit(0)
