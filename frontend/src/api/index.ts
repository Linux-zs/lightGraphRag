const BASE = '/api'

function formatApiError(payload: unknown, fallback: string): string {
  if (!payload) return fallback
  if (typeof payload === 'string') return payload
  if (payload instanceof Error) return payload.message || fallback

  const maybeRecord = payload as Record<string, unknown>
  const detail = maybeRecord.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === 'string') return item
        if (item && typeof item === 'object') {
          const record = item as Record<string, unknown>
          const loc = Array.isArray(record.loc) ? record.loc.join('.') : ''
          const msg = typeof record.msg === 'string' ? record.msg : JSON.stringify(record)
          return loc ? `${loc}: ${msg}` : msg
        }
        return String(item)
      })
      .filter(Boolean)
    return messages.join('; ') || fallback
  }
  if (detail && typeof detail === 'object') {
    const record = detail as Record<string, unknown>
    if (typeof record.message === 'string') return record.message
    if (typeof record.error === 'string') return record.error
    return JSON.stringify(detail)
  }
  if (typeof maybeRecord.message === 'string') return maybeRecord.message
  if (typeof maybeRecord.error === 'string') return maybeRecord.error

  try {
    return JSON.stringify(payload)
  } catch {
    return fallback
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(formatApiError(err, `HTTP ${res.status}`))
  }
  return res.json()
}

// --- Types ---

export interface ChunkPreviewItem {
  index: number
  text: string
  char_count: number
}

export interface RecallChunk {
  reference_id?: string
  chunk_id?: string
  file_path?: string
  content?: string
}

export interface RecallTestResponse {
  query: string
  mode: string
  context: string
  chunks: RecallChunk[]
  entities: Record<string, unknown>[]
  relationships: Record<string, unknown>[]
  references: Record<string, unknown>[]
  metadata: Record<string, unknown>
}

export interface TextRecallHit {
  chunk_id: string
  file_path: string
  content: string
  vector_score: number
  vector_rank: number
  rerank_score?: number | null
  rerank_rank?: number | null
}

export interface TextRecallResponse {
  query: string
  workspace: string
  top_k: number
  cosine_threshold: number
  rerank_requested: boolean
  rerank_applied: boolean
  rerank_warning: string
  vector_hits: TextRecallHit[]
  rerank_hits: TextRecallHit[]
}

export interface ModelConfig {
  workspace: string
  embed_model: string
  embed_base_url: string
  rerank_model: string
  chat_model: string
  chat_temperature: number
  chat_top_p: number
  chat_max_tokens: number
  frequency_penalty: number
  presence_penalty: number
  answer_prompt_template_id: string
  answer_system_prompt: string
}

export interface PromptTemplate {
  id: string
  name: string
  description: string
  content: string
  built_in: boolean
  created_at: string
  updated_at: string
}

export interface DocInfo {
  doc_id: string
  doc_name: string
  chunk_count: number
  file_type: string
  char_count?: number
  indexed?: boolean
  status?: string
  error_msg?: string
  graph_rule?: GraphRuleSummary
  index_mode?: 'complete' | 'fast'
  kg_status?: 'complete' | 'skipped' | 'filtered_empty' | 'failed' | string
  kg_model?: string
}

export interface WorkspaceInfo {
  workspace: string
  is_default: boolean
  doc_count: number
  uploaded_doc_count: number
  graph_nodes: number
  graph_edges: number
  manifest_path: string
  workspace_path: string
  exists: boolean
}

export function listWorkspaces() {
  return request<WorkspaceInfo[]>('/kb/workspaces')
}

export function createWorkspace(
  workspace: string,
  ruleTemplateId: string,
  extractionMode: 'assist' | 'enhanced' | 'strict' = 'enhanced',
  allowOtherEntityType = false,
) {
  return request<WorkspaceInfo>('/kb/workspaces', {
    method: 'POST',
    body: JSON.stringify({
      workspace,
      rule_template_id: ruleTemplateId,
      extraction_mode: extractionMode,
      allow_other_entity_type: allowOtherEntityType,
    }),
  })
}

export function deleteWorkspace(workspace: string) {
  return request<{ deleted: string }>(`/kb/workspaces/${encodeURIComponent(workspace)}`, {
    method: 'DELETE',
  })
}

// --- KB ---

export interface UploadedDocument {
  file_name: string
  workspace: string
  doc_id: string
  file_type: string
  char_count: number
  preview: string
  index_invalidated: boolean
}

export function uploadDocument(file: File, workspace = 'tdx_default'): Promise<UploadedDocument> {
  const form = new FormData()
  form.append('file', file)
  return fetch(`${BASE}/kb/upload?workspace=${encodeURIComponent(workspace)}`, {
    method: 'POST',
    body: form,
  }).then(async (r) => {
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }))
      throw new Error(formatApiError(err, `HTTP ${r.status}`))
    }
    return r.json() as Promise<UploadedDocument>
  })
}

export function previewChunks(params: {
  workspace?: string
  file_name: string
  separators: string[]
  chunk_size: number
  chunk_overlap: number
}) {
  return request<ChunkPreviewItem[]>('/kb/preview-chunks', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export function indexDocument(params: {
  workspace?: string
  file_name: string
  separators: string[]
  chunk_size: number
  chunk_overlap: number
  index_mode?: 'complete' | 'fast'
}) {
  return request<IndexTask>(
    '/kb/index',
    { method: 'POST', body: JSON.stringify(params) },
  )
}

export function listDocuments(workspace = 'tdx_default') {
  return request<DocInfo[]>(`/kb/documents?workspace=${encodeURIComponent(workspace)}`)
}

export interface GraphDeleteResiduals {
  checked: boolean
  has_residuals: boolean
  node_count: number
  edge_count: number
  nodes: Array<{ id: string; label: string; source_id?: string; file_path?: string }>
  edges: Array<{ source: string; target: string; source_id?: string; file_path?: string }>
  graph_exists?: boolean
  error?: string
}

export interface BatchGraphDeleteResiduals {
  has_residuals: boolean
  items: Array<GraphDeleteResiduals & { doc_name: string; doc_id: string }>
}

export interface DocumentDeleteCleanup {
  cleanup_task?: IndexTask | null
  cleanup_error?: string
}

export function deleteDocument(docName: string, workspace = 'tdx_default') {
  return request<{
    deleted: number
    doc_id: string
    doc_name: string
    graph_residuals: GraphDeleteResiduals
  } & DocumentDeleteCleanup>(
    `/kb/documents/${encodeURIComponent(docName)}?workspace=${encodeURIComponent(workspace)}`,
    { method: 'DELETE' },
  )
}

export function batchDeleteDocuments(docNames: string[], workspace = 'tdx_default') {
  return request<{
    deleted_chunks: number
    doc_count: number
    errors?: Array<{ doc_name: string; error: string }>
    graph_residuals?: BatchGraphDeleteResiduals
    cleanup_task?: IndexTask | null
    cleanup_error?: string
  }>(
    '/kb/batch-delete',
    { method: 'POST', body: JSON.stringify({ workspace, doc_names: docNames }) },
  )
}

export function batchIndexDocuments(params: {
  workspace?: string
  doc_names: string[]
  separators: string[]
  chunk_size: number
  chunk_overlap: number
  index_mode?: 'complete' | 'fast'
}) {
  return request<IndexTask>(
    '/kb/batch-index',
    { method: 'POST', body: JSON.stringify(params) },
  )
}

export interface IndexTaskResult {
  doc_name: string
  doc_id?: string
  status: 'ok' | 'error'
  chunks?: number
  error?: string
  kg_status?: string
  kg_timed_out_chunks?: string[]
  stage_timings?: {
    parse: number
    chunk_vector: number
    kg: number
    merge: number
  } | null
}

export interface IndexTask {
  task_id: string
  kind: 'single' | 'batch' | 'rebuild'
  workspace?: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'partial' | 'cancelled'
  doc_names: string[]
  total: number
  current: number
  progress: number
  message: string
  current_doc?: string
  current_doc_started_at?: string
  current_stage?: 'parse' | 'chunk_vector' | 'kg' | 'merge' | ''
  current_stage_started_at?: string
  timeout_seconds?: number
  stage_timings?: {
    parse: number
    chunk_vector: number
    kg: number
    merge: number
  } | null
  request?: {
    separators?: string[]
    chunk_size?: number
    chunk_overlap?: number
    index_mode?: 'complete' | 'fast'
  }
  results: IndexTaskResult[]
  errors: IndexTaskResult[]
  created_at: string
  updated_at: string
}

export function getIndexTask(taskId: string) {
  return request<IndexTask>(`/kb/index-tasks/${taskId}`)
}

export function listIndexTasks() {
  return request<IndexTask[]>('/kb/index-tasks')
}

export function cancelIndexTask(taskId: string) {
  return request<IndexTask>(`/kb/index-tasks/${taskId}/cancel`, { method: 'POST' })
}

// --- Raw text preview / edit (pre-chunking) ---

export interface RawTextResponse {
  file_name: string
  file_type: string
  char_count: number
  raw_text: string
  source: string
}

export function getDocumentRawText(docName: string, workspace = 'tdx_default') {
  return request<RawTextResponse>(
    `/kb/documents/${encodeURIComponent(docName)}/raw-text?workspace=${encodeURIComponent(workspace)}`,
  )
}

export function updateDocumentRawText(docName: string, raw_text: string, workspace = 'tdx_default') {
  return request<{ file_name: string; char_count: number; message: string }>(
    `/kb/documents/${encodeURIComponent(docName)}/raw-text?workspace=${encodeURIComponent(workspace)}`,
    { method: 'PUT', body: JSON.stringify({ raw_text }) },
  )
}

// --- Indexed chunk viewer ---

export interface DocumentChunkItem {
  chunk_id: string
  chunk_index: number
  text: string
  char_count: number
}

export interface DocumentChunksResponse {
  doc_name: string
  total: number
  chunks: DocumentChunkItem[]
}

export function getDocumentChunks(docName: string, workspace = 'tdx_default') {
  return request<DocumentChunksResponse>(
    `/kb/documents/${encodeURIComponent(docName)}/chunks?workspace=${encodeURIComponent(workspace)}`,
  )
}

// --- Recall ---

export function recallTest(params: {
  workspace?: string
  query: string
  mode: string
  top_k: number
  chunk_top_k: number
  enable_rerank: boolean
}) {
  return request<RecallTestResponse>('/recall/test', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export function textRecallTest(params: {
  workspace?: string
  query: string
  top_k: number
  enable_rerank: boolean
}) {
  return request<TextRecallResponse>('/recall/text', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export function search(params: {
  workspace?: string
  query: string
  mode?: string
  top_k: number
  chunk_top_k?: number
  enable_rerank: boolean
}) {
  return request<{
    question: string
    content: string
    citations: { doc_name: string; chunk_index: number; excerpt: string }[]
    trace: Record<string, unknown>
  }>('/search', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

// --- Models ---

export function getModelConfig(workspace = 'tdx_default') {
  return request<ModelConfig>(`/models/config?workspace=${encodeURIComponent(workspace)}`)
}

export function updateModelConfig(config: ModelConfig, workspace = 'tdx_default') {
  return request<{ status: string }>(
    `/models/config?workspace=${encodeURIComponent(workspace)}`,
    {
    method: 'PUT',
    body: JSON.stringify(config),
    },
  )
}

export function listPromptTemplates() {
  return request<PromptTemplate[]>('/prompt-templates')
}

export function createPromptTemplate(template: Pick<PromptTemplate, 'name' | 'description' | 'content'>) {
  return request<PromptTemplate>('/prompt-templates', {
    method: 'POST',
    body: JSON.stringify(template),
  })
}

export function updatePromptTemplate(
  id: string,
  template: Pick<PromptTemplate, 'name' | 'description' | 'content'>,
) {
  return request<PromptTemplate>(`/prompt-templates/${encodeURIComponent(id)}`, {
    method: 'PUT',
    body: JSON.stringify(template),
  })
}

export function deletePromptTemplate(id: string) {
  return request<{ status: string }>(`/prompt-templates/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  })
}

export function testEmbed(text: string) {
  return request<{ dimensions: number; preview: number[] }>('/models/test-embed', {
    method: 'POST',
    body: JSON.stringify({ text }),
  })
}

export interface ModelProfile {
  id: string
  name: string
  api_base: string
  api_type: string
  models_cache: DiscoveredModel[]
  has_api_key: boolean
  api_key_preview: string
  last_used_at?: string
  last_tested_at?: string
  created_at?: string
  updated_at?: string
}

export interface DiscoveredModel {
  id: string
  type: string
}

export interface ModelBinding {
  profile_id: string
  model: string
  embed_dim?: number
  embed_max_chars?: number
  enabled?: boolean
}

export interface ModelBindings {
  chat: ModelBinding
  kg: ModelBinding
  embedding: ModelBinding
  rerank: ModelBinding
}

export function listModelProfiles() {
  return request<ModelProfile[]>('/model-profiles')
}

export function saveModelProfile(profile: {
  id?: string
  name: string
  api_base: string
  api_key?: string
  api_type?: string
}) {
  const path = profile.id ? `/model-profiles/${encodeURIComponent(profile.id)}` : '/model-profiles'
  return request<ModelProfile>(path, {
    method: profile.id ? 'PUT' : 'POST',
    body: JSON.stringify(profile),
  })
}

export function deleteModelProfile(profileId: string) {
  return request<{ deleted: string }>(`/model-profiles/${encodeURIComponent(profileId)}`, {
    method: 'DELETE',
  })
}

export function discoverModels(api_base: string, api_key = '') {
  return request<{ models: DiscoveredModel[] }>('/model-profiles/discover', {
    method: 'POST',
    body: JSON.stringify({ api_base, api_key }),
  })
}

export function discoverProfileModels(profileId: string) {
  return request<{ models: DiscoveredModel[] }>(`/model-profiles/${encodeURIComponent(profileId)}/discover`, {
    method: 'POST',
  })
}

export function getModelBindings() {
  return request<ModelBindings>('/model-bindings')
}

export function updateModelBindings(bindings: ModelBindings) {
  return request<{
    bindings: ModelBindings
    embedding_changed: boolean
    affected_workspaces: string[]
  }>('/model-bindings', {
    method: 'PUT',
    body: JSON.stringify({ bindings }),
  })
}

export function testChatModel(profile_id: string, model: string) {
  return request<{ ok: boolean; model: string; usage?: Record<string, unknown> }>('/model-profiles/test-chat', {
    method: 'POST',
    body: JSON.stringify({ profile_id, model }),
  })
}

export function testEmbeddingModel(profile_id: string, model: string) {
  return request<{ ok: boolean; model: string; dimensions: number; preview: number[] }>('/model-profiles/test-embedding', {
    method: 'POST',
    body: JSON.stringify({ profile_id, model }),
  })
}

export function testRerankModel(profile_id: string, model: string) {
  return request<{ ok: boolean; model: string; results: unknown[] }>('/model-profiles/test-rerank', {
    method: 'POST',
    body: JSON.stringify({ profile_id, model }),
  })
}

export function healthCheck() {
  return request<{ status: string }>('/health')
}

// --- System Stats ---

export interface SystemStats {
  doc_count: number
  uploaded_doc_count?: number
  chunk_count: number
  graph_nodes: number
  graph_edges: number
  embed_model: string
  embed_dim?: number
  workspace?: string
  lightrag_dir?: string
  lightrag_dir_size?: string
}

export function getSystemStats(workspace = 'tdx_default') {
  return request<SystemStats>(`/system/stats?workspace=${encodeURIComponent(workspace)}`)
}

export interface SystemLogItem {
  line_no: number
  level: string
  text: string
}

export interface SystemLogsResponse {
  path: string
  exists: boolean
  total_matched: number
  items: SystemLogItem[]
}

export function getSystemLogs(params: {
  limit?: number
  level?: string
  contains?: string
} = {}) {
  const query = new URLSearchParams()
  if (params.limit) query.set('limit', String(params.limit))
  if (params.level) query.set('level', params.level)
  if (params.contains) query.set('contains', params.contains)
  const suffix = query.toString() ? `?${query}` : ''
  return request<SystemLogsResponse>(`/system/logs${suffix}`)
}

export interface ClearKnowledgeBaseResult {
  workspace: string
  workspace_path: string
  manifest_path: string
  removed_workspace: boolean
  removed_manifest: boolean
  removed_uploads: number
}

export function clearKnowledgeBase(clearUploads = false, workspace = 'tdx_default') {
  return request<ClearKnowledgeBaseResult>('/kb/clear', {
    method: 'POST',
    body: JSON.stringify({ workspace, clear_uploads: clearUploads }),
  })
}

export function rebuildIndex(params: {
  workspace?: string
  separators: string[]
  chunk_size: number
  chunk_overlap: number
  index_mode?: 'complete' | 'fast'
}) {
  return request<IndexTask & { clear_result?: ClearKnowledgeBaseResult }>('/kb/rebuild', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

// --- Knowledge Graph ---

export interface GraphNode {
  id: string
  label: string
  category: string
  description: string
  critical: boolean
  entity_type?: string
  source_id?: string
  file_path?: string
  degree?: number
}

export interface GraphEdge {
  source: string
  target: string
  relation: string
  description?: string
  keywords?: string
  weight?: number
  source_id?: string
  file_path?: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
  metadata?: {
    exists?: boolean
    total_nodes?: number
    total_edges?: number
    returned_nodes?: number
    returned_edges?: number
    truncated?: boolean
    directed?: boolean
    path?: string
    error?: string
  }
}

export function getGraph(limit = 200, workspace = 'tdx_default') {
  return request<GraphData>(`/graph?limit=${limit}&workspace=${encodeURIComponent(workspace)}`)
}

export interface GraphGovernanceConfig {
  workspace: string
  rule_template_id: string
  rule_template_name: string
  extraction_mode: 'assist' | 'enhanced' | 'strict'
  allow_other_entity_type: boolean
  entity_types: string[]
  relation_types: string[]
  aliases_text: string
  extraction_prompt: string
  effective_extraction_prompt: string
  reference_files: GraphReferenceFile[]
  updated_at: string
  audit_log: GraphAuditEntry[]
}

export interface GraphRuleSummary {
  rule_template_id: string
  rule_template_name: string
  extraction_mode: string
  allow_other_entity_type: boolean
  entity_type_count: number
  relation_type_count: number
  extraction_prompt_preview: string
  effective_extraction_prompt_preview?: string
  updated_at: string
}

export interface GraphRuleTemplate {
  id: string
  name: string
  description: string
  entity_types: string[]
  relation_types: string[]
  aliases_text: string
  extraction_prompt: string
  built_in: boolean
  created_at?: string
  updated_at?: string
}

export interface GraphReferenceFile {
  id: string
  file_name: string
  path: string
  char_count: number
  created_at: string
}

export interface GraphAuditEntry {
  id: string
  action: string
  payload: Record<string, unknown>
  result: unknown
  created_at: string
}

export interface GraphChange {
  action: string
  reason?: string
  entity_name?: string
  source_entity?: string
  target_entity?: string
  source_entities?: string[]
  entity_data?: Record<string, unknown>
  relation_data?: Record<string, unknown>
  updated_data?: Record<string, unknown>
  target_entity_data?: Record<string, unknown>
}

export function getGraphGovernanceConfig(workspace = 'tdx_default') {
  return request<GraphGovernanceConfig>(
    `/graph/governance/config?workspace=${encodeURIComponent(workspace)}`,
  )
}

export function listGraphRuleTemplates(workspace = 'tdx_default') {
  return request<GraphRuleTemplate[]>(
    `/graph/rule-templates?workspace=${encodeURIComponent(workspace)}`,
  )
}

export function saveGraphRuleTemplate(template: GraphRuleTemplate) {
  return request<GraphRuleTemplate>('/graph/rule-templates', {
    method: 'POST',
    body: JSON.stringify(template),
  })
}

export function deleteGraphRuleTemplate(templateId: string) {
  return request<{ deleted: string }>(`/graph/rule-templates/${encodeURIComponent(templateId)}`, {
    method: 'DELETE',
  })
}

export function applyGraphRuleTemplate(workspace: string, templateId: string) {
  return request<GraphGovernanceConfig>('/graph/governance/apply-template', {
    method: 'POST',
    body: JSON.stringify({ workspace, template_id: templateId }),
  })
}

export function updateGraphGovernanceConfig(config: {
  workspace?: string
  rule_template_id?: string
  rule_template_name?: string
  extraction_mode?: 'assist' | 'enhanced' | 'strict'
  allow_other_entity_type?: boolean
  entity_types: string[]
  relation_types: string[]
  aliases_text: string
  extraction_prompt: string
}) {
  return request<GraphGovernanceConfig>('/graph/governance/config', {
    method: 'PUT',
    body: JSON.stringify(config),
  })
}

export function uploadGraphReference(file: File, workspace = 'tdx_default') {
  const form = new FormData()
  form.append('file', file)
  return fetch(`${BASE}/graph/governance/references?workspace=${encodeURIComponent(workspace)}`, {
    method: 'POST',
    body: form,
  }).then(async (r) => {
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }))
      throw new Error(formatApiError(err, `HTTP ${r.status}`))
    }
    return r.json() as Promise<GraphReferenceFile>
  })
}

export function deleteGraphReference(refId: string, workspace = 'tdx_default') {
  return request<{ deleted: string }>(
    `/graph/governance/references/${encodeURIComponent(refId)}?workspace=${encodeURIComponent(workspace)}`,
    { method: 'DELETE' },
  )
}

export function createGraphEntity(params: {
  workspace?: string
  entity_name: string
  entity_type: string
  description: string
  source_id?: string
  file_path?: string
}) {
  return request<{ status: string; data: unknown }>('/graph/entities', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export function updateGraphEntity(params: {
  workspace?: string
  entity_name: string
  updated_data: Record<string, unknown>
  allow_rename?: boolean
  allow_merge?: boolean
}) {
  return request<{ status: string; data: unknown }>('/graph/entities', {
    method: 'PUT',
    body: JSON.stringify(params),
  })
}

export function deleteGraphEntity(entityName: string, workspace = 'tdx_default') {
  return request<{ status: string; data: unknown }>(
    `/graph/entities/${encodeURIComponent(entityName)}?workspace=${encodeURIComponent(workspace)}`,
    { method: 'DELETE' },
  )
}

export function createGraphRelation(params: {
  workspace?: string
  source_entity: string
  target_entity: string
  description: string
  keywords?: string
  weight?: number
  source_id?: string
  file_path?: string
}) {
  return request<{ status: string; data: unknown }>('/graph/relations', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export function updateGraphRelation(params: {
  workspace?: string
  source_entity: string
  target_entity: string
  updated_data: Record<string, unknown>
}) {
  return request<{ status: string; data: unknown }>('/graph/relations', {
    method: 'PUT',
    body: JSON.stringify(params),
  })
}

export function deleteGraphRelation(params: {
  workspace?: string
  source_entity: string
  target_entity: string
}) {
  return request<{ status: string; data: unknown }>('/graph/relations', {
    method: 'DELETE',
    body: JSON.stringify(params),
  })
}

export function mergeGraphEntities(params: {
  workspace?: string
  source_entities: string[]
  target_entity: string
  target_entity_data?: Record<string, unknown>
}) {
  return request<{ status: string; data: unknown }>('/graph/entities/merge', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export function suggestGraphChanges(params: {
  workspace?: string
  instruction: string
  limit?: number
}) {
  return request<{ changes: GraphChange[]; raw_text: string }>('/graph/governance/suggest', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export function applyGraphChanges(workspace: string, changes: GraphChange[]) {
  return request<{ workspace: string; results: { action: string; status: string; result?: unknown; error?: string }[] }>(
    '/graph/governance/apply',
    { method: 'POST', body: JSON.stringify({ workspace, changes }) },
  )
}

export interface EvidenceChain {
  nodes: GraphNode[]
  edges: GraphEdge[]
  chunks: Citation[]
}

// --- Chat ---

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  citations?: Citation[]
  evidence?: EvidenceChain
}

export interface ChatSettings {
  answer_profile_id: string
  answer_model: string
  temperature: number
  top_p: number
  max_tokens: number
  frequency_penalty: number
  presence_penalty: number
  mode: string
  top_k: number
  chunk_top_k: number
  enable_rerank: boolean
}

export interface Citation {
  index: number
  doc_name: string
  chunk_index: number
  excerpt: string
}

export interface ChatSendResponse {
  session_id: string
  title: string
  user_message: ChatMessage
  assistant_message: ChatMessage
  citations: Citation[]
  evidence?: EvidenceChain
}

export interface ChatSession {
  id: string
  workspace: string
  title: string
  settings: ChatSettings
  messages: ChatMessage[]
  created_at: string
  updated_at: string
}

export interface ChatSessionListItem {
  id: string
  workspace: string
  title: string
  message_count: number
  created_at: string
  updated_at: string
}

export function chatSend(params: {
  session_id?: string | null
  workspace?: string
  message: string
  mode?: string
  top_k?: number
  chunk_top_k?: number
  enable_rerank?: boolean
  settings?: ChatSettings
}) {
  return request<ChatSendResponse>('/chat/send', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

/** Stream chat response via SSE. Returns a ReadableStream of SSE events. */
export function chatSendStream(params: {
  session_id?: string | null
  workspace?: string
  message: string
  mode?: string
  top_k?: number
  chunk_top_k?: number
  enable_rerank?: boolean
  settings?: ChatSettings
}): Promise<Response> {
  return fetch(`${BASE}/chat/send/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
}

export function listChatSessions(workspace = 'tdx_default') {
  return request<ChatSessionListItem[]>(`/chat/sessions?workspace=${encodeURIComponent(workspace)}`)
}

export function getChatSession(sessionId: string, workspace = 'tdx_default') {
  return request<ChatSession>(
    `/chat/sessions/${encodeURIComponent(sessionId)}?workspace=${encodeURIComponent(workspace)}`,
  )
}

export function deleteChatSession(sessionId: string, workspace = 'tdx_default') {
  return request<{ deleted: string }>(
    `/chat/sessions/${encodeURIComponent(sessionId)}?workspace=${encodeURIComponent(workspace)}`,
    {
    method: 'DELETE',
    },
  )
}

export function updateChatSessionSettings(
  sessionId: string,
  settings: ChatSettings,
  workspace = 'tdx_default',
) {
  return request<ChatSettings>(
    `/chat/sessions/${encodeURIComponent(sessionId)}/settings?workspace=${encodeURIComponent(workspace)}`,
    {
      method: 'PATCH',
      body: JSON.stringify(settings),
    },
  )
}

export function createChatSession(workspace = 'tdx_default', settings?: ChatSettings) {
  return request<ChatSessionListItem>('/chat/sessions', {
    method: 'POST',
    body: JSON.stringify({ workspace, settings }),
  })
}
