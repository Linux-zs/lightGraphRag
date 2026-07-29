# LightGraphRAG Workbench

基于 LightRAG SDK 的多知识库工作台。后端使用 FastAPI，前端使用 React，支持文档上传与索引、知识库问答、上下文预览、引用证据链、知识图谱查看与人工治理。

## 环境

- Python 3.10+
- uv
- Node.js 20+

## 启动

```powershell
uv sync --locked --all-groups
uv run python -m src.app.cli server --no-reload
```

另开终端启动前端：

```powershell
cd frontend
npm ci
npm run dev
```

默认地址：

- 前端：`http://127.0.0.1:5173`
- API：`http://127.0.0.1:8101`
- OpenAPI：`http://127.0.0.1:8101/docs`

## CLI

```powershell
uv run lightgraphrag ingest --workspace my_kb --docs-dir D:\docs
uv run lightgraphrag search --workspace my_kb "问题"
uv run lightgraphrag rebuild --workspace my_kb --docs-dir D:\docs
```

`rebuild` 只清理目标知识库，不会操作其他 workspace。

## 数据与安全

运行数据位于 `data/`，不会提交到 Git。上传文档、测试资料、截图、模型密钥也都被忽略。模型 API Key 使用 Windows DPAPI 加密保存；服务默认只监听回环地址。远程监听时必须设置 `TDX_APP_API_TOKEN`，请求通过 `X-App-Token` 提交令牌。

修改 embedding 地址、模型或维度后，已有知识库会被标记为不兼容，必须先重建索引。

## 验证

```powershell
uv lock --check
uv run pytest -q
uv run python -m compileall -q src tests

cd frontend
npm ci
npm run build
npm audit --omit=dev --audit-level=high
```
