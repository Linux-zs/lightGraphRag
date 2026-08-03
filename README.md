# LightGraphRAG Workbench

LightGraphRAG Workbench 是一个基于 LightRAG SDK 的本地知识库工作台。后端使用 FastAPI，前端使用 React/Vite，当前路线是保留已有工作台壳子，把问答、索引、召回和知识图谱能力统一接到 LightRAG。

当前项目不接入 LightRAG 官方 Server，不迁移旧 Chroma 索引。知识库数据从 `data/lightrag` 重新建立。

## 一、主要功能

- 多知识库管理：不同知识库使用独立 workspace、上传目录、LightRAG 索引、manifest 和图谱治理配置。
- 文档上传与索引：前端默认支持 `txt`、`md`、`pdf`、`docx`，支持原始文本预览、编辑、切块预览、单文档索引、批量索引、取消任务和进度恢复。
- LightRAG 问答：支持普通返回和 SSE 流式输出，支持多轮会话、自动会话标题、引用文档、证据链和上下文查看。
- 召回调试：支持 LightRAG 生成上下文预览，也支持文本向量召回与 Rerank 前后排序对比。
- 知识图谱：支持图谱浏览、抽取规则模板、抽取参考文件、实体治理、关系治理、合并实体和基于指令生成修正建议。
- 模型设置：支持多个 OpenAI-compatible 模型连接档案，分别绑定大语言模型、嵌入模型和 Rerank 模型。API Key 加密保存到本地数据目录。
- 系统状态：查看当前知识库文档数、chunk 数、图谱节点/关系、LightRAG 目录大小，支持清空索引和重建当前知识库索引。

## 二、技术栈

- 后端：Python 3.10+、FastAPI、LightRAG SDK、Loguru、PyMuPDF、python-docx、NetworkX。
- 前端：Node.js 20+、React 18、Vite、TypeScript、Tailwind CSS、lucide-react。
- 包管理：Python 使用 `uv`，前端使用 `npm`。
- 默认模型接口：OpenAI-compatible API，默认配置指向 SiliconFlow。

## 三、目录结构

```text
.
├── config/
│   └── default.yaml             # 默认配置
├── frontend/
│   ├── src/                     # React 页面、组件和 API 封装
│   └── package.json
├── src/
│   ├── api/server.py            # FastAPI 接口
│   ├── app/cli.py               # CLI 入口
│   ├── doc_processor/           # 文档解析与切块
│   ├── lightrag_service.py      # LightRAG SDK 适配层
│   ├── llm_backend/             # 模型后端封装
│   └── model_profiles.py        # 模型连接档案和密钥存储
├── tests/                       # 后端测试
├── docs/
│   └── USER_MANUAL.md           # 使用手册
├── pyproject.toml
├── uv.lock
└── README.md
```

运行数据默认位于 `data/`，不会提交到 Git。

## 四、环境准备

```powershell
uv sync --locked --all-groups

cd frontend
npm ci
```

需要准备一个可用的 OpenAI-compatible 模型服务。推荐先配置：

- 大语言模型：用于回答生成、会话标题、图谱修正建议。
- 嵌入模型：用于索引和向量召回。
- Rerank 模型：用于召回结果重排，可选。

嵌入模型的维度必须和 LightRAG 索引一致。更换嵌入模型、嵌入地址或维度后，需要重建对应知识库索引。

## 五、启动

启动后端：

```powershell
uv run python -m src.app.cli server --no-reload
```

开发模式可启用后端 reload：

```powershell
uv run python -m src.app.cli server
```

启动前端：

```powershell
cd frontend
npm run dev
```

默认访问地址：

- 前端：`http://127.0.0.1:5173`
- API：`http://127.0.0.1:8101`
- OpenAPI：`http://127.0.0.1:8101/docs`

## 六、CLI

CLI 适合批量导入、重建和简单检索。

```powershell
uv run lightgraphrag ingest --workspace my_kb --docs-dir D:\docs
uv run lightgraphrag search --workspace my_kb "问题"
uv run lightgraphrag rebuild --workspace my_kb --docs-dir D:\docs
```

`rebuild` 只处理目标知识库，不会重建其他 workspace。它会在 shadow workspace
中完成全部索引和人工图谱回放，成功后再切换；失败时保留当前可用索引。
指定 `--docs-dir` 时，目录中的文件集合和内容必须与该知识库已管理的上传文件一致，
否则命令会拒绝执行并提示先运行 `ingest`。

## 七、配置

默认配置文件：

```text
config/default.yaml
```

机器本地覆盖配置：

```text
config/local.yaml
```

加载顺序为：

```text
config/default.yaml -> config/local.yaml -> 环境变量
```

`config/local.yaml` 不进入版本控制，适合保存本机文档目录和本地模型地址。例如：

```yaml
paths:
  docs_dir: D:/local/docs
ollama:
  host: http://127.0.0.1:11434
```

常用环境变量：

```powershell
$env:LIGHTGRAPHRAG_CONFIG_PATH="D:\path\custom.yaml"
$env:LIGHTGRAPHRAG_CONFIG_LOCAL_PATH="D:\path\local.yaml"
$env:LIGHTGRAPHRAG_DOCS_DIR="D:\docs"
$env:LIGHTGRAPHRAG_DATA_DIR="./data"
$env:LIGHTGRAPHRAG_UPLOAD_DIR="data/uploads"
$env:LIGHTGRAPHRAG_RAW_TEXT_DIR="data/upload_text"
$env:LIGHTGRAPHRAG_LOG_DIR="data/logs"
$env:LIGHTGRAPHRAG_INDEX_DOC_TIMEOUT_SECONDS="180"
$env:LIGHTGRAPHRAG_DOCUMENT_UPLOAD_MAX_BYTES="52428800"
$env:LIGHTGRAPHRAG_GRAPH_UPLOAD_MAX_BYTES="2097152"
$env:LIGHTGRAPHRAG_PARSED_TEXT_MAX_CHARS="5000000"
$env:LIGHTGRAPHRAG_CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

部署或通过反向代理访问时必须设置访问令牌：

```powershell
$env:LIGHTGRAPHRAG_APP_API_TOKEN="replace-with-a-random-token"
uv run python -m src.app.cli server --host 0.0.0.0 --no-reload
```

一旦配置令牌，所有受保护 API（包括本机回环请求）都必须带请求头：

```text
X-App-Token: replace-with-a-random-token
```

模型连接建议通过前端“模型设置”维护。API Key 保存后不会回显，密钥文件位于 `data/secrets/`。
网页首次收到 401/403 时会要求输入令牌，可选择保存在浏览器 `localStorage` 或仅保存在当前会话。
反向代理必须保留 `X-App-Token` 请求头，并将浏览器来源加入
`LIGHTGRAPHRAG_CORS_ORIGINS` 白名单。

后端和 CLI 对同一 `data_dir` 使用独占运行锁，只允许一个后端进程或一个 CLI
数据操作进程。不要使用多个 Uvicorn worker。

## 八、数据目录

默认数据布局：

```text
data/
├── uploads/<workspace>/                 # 上传原始文件
├── upload_text/<workspace>/             # 解析后的可编辑文本
├── lightrag/<workspace>/                # LightRAG 工作目录
├── lightrag_manifest.json               # 默认知识库 manifest
├── lightrag_manifests/<workspace>.json  # 非默认知识库 manifest
├── lightrag_embedding_meta/<workspace>.json
├── graph_governance/<workspace>.json
├── graph_governance_refs/<workspace>/
├── graph_rule_templates.json
├── model_profiles.json
├── prompt_templates.json
├── workspace_settings/<workspace>.json
├── sessions/
├── index_tasks/
├── secrets/
└── logs/app.log
```

`data/`、`.env`、`.venv`、`frontend/node_modules`、上传文档、测试数据和截图都被 `.gitignore` 排除。

## 九、接口概览

主要 API 前缀：

```text
/api/kb/*                  知识库、上传、索引、文档、任务
/api/chat/*                会话和问答
/api/recall/*              上下文预览和文本召回
/api/graph/*               图谱浏览和治理
/api/model-profiles/*      模型连接档案
/api/model-bindings        模型绑定
/api/models/config         当前知识库问答提示词
/api/prompt-templates      提示词模板
/api/system/stats          系统状态
/api/system/logs           运行日志
/api/health                健康检查
```

详细参数以 OpenAPI 页面为准：

```text
http://127.0.0.1:8101/docs
```

## 十、验证

后端：

```powershell
uv lock --check
uv run pytest -q
uv run python -m compileall -q src tests
```

前端：

```powershell
cd frontend
npm run build
npm audit --omit=dev --audit-level=high
```

## 十一、常见问题

### 1. 上传成功但问答没有内容

先确认文档状态是否已经完成索引。`uploaded` 只表示文件已保存和解析，不能直接用于 LightRAG 问答；成功索引后才会进入召回和问答。

如果不确定索引是否卡住，进入“系统状态”查看最近索引任务和运行日志。

### 2. 更换嵌入模型后检索失败

嵌入模型、API 地址或维度变化后，旧索引不可复用。需要进入“系统状态”重建当前知识库索引。

### 3. 非知识库问题仍返回文档内容

先用“召回调试”检查查询命中的上下文。如果无关问题仍命中高分文本块，需要调低召回范围、调整提示词，或补充更明确的拒答规则。

### 4. 知识图谱实体关系为 0

检查当前知识库的抽取规则、抽取模式和模型能力。普通知识库建议使用“辅助”或“增强”模式，不建议直接使用严格领域规则。

### 5. 手动推送失败

当前仓库 remote 使用标准 GitHub SSH 地址：

```text
git@github.com:Linux-zs/lightGraphRag.git
```

本机 `.ssh/config` 已将 `github.com` 映射到 `ssh.github.com:443`，普通推送命令为：

```powershell
git push origin main
```

## 十二、使用手册

完整操作说明见：

[docs/USER_MANUAL.md](docs/USER_MANUAL.md)
