import { useEffect, useRef, useState } from 'react'
import FileUpload from '../components/FileUpload'
import ChunkPreview from '../components/ChunkPreview'
import {
  uploadDocument,
  previewChunks,
  indexDocument,
  listDocuments,
  deleteDocument,
  batchDeleteDocuments,
  batchIndexDocuments,
  getIndexTask,
  listIndexTasks,
  cancelIndexTask,
  getDocumentRawText,
  updateDocumentRawText,
  getDocumentChunks,
  getGraphGovernanceConfig,
  ChunkPreviewItem,
  DocInfo,
  DocumentChunkItem,
  GraphDeleteResiduals,
  GraphGovernanceConfig,
  IndexTask,
  UploadedDocument,
} from '../api'

type UploadedFile = UploadedDocument

interface Props {
  workspace: string
}

export default function KBManagement({ workspace }: Props) {
  // Upload state
  const [uploaded, setUploaded] = useState<UploadedFile | null>(null)
  const [uploading, setUploading] = useState(false)

  // Chunking params
  const [separators, setSeparators] = useState('\\n\\n, \\n, 。, ！, ？, ；,  ')
  const [chunkSize, setChunkSize] = useState(512)
  const [chunkOverlap, setChunkOverlap] = useState(50)

  // Preview & index state
  const [chunks, setChunks] = useState<ChunkPreviewItem[]>([])
  const [previewing, setPreviewing] = useState(false)
  const [indexing, setIndexing] = useState(false)
  const [indexMsg, setIndexMsg] = useState('')
  const [indexTask, setIndexTask] = useState<IndexTask | null>(null)
  const [previewError, setPreviewError] = useState('')

  // Document list
  const [docs, setDocs] = useState<DocInfo[]>([])
  const [deleting, setDeleting] = useState<string | null>(null)

  // Batch operations
  const [checkedDocs, setCheckedDocs] = useState<Set<string>>(new Set())
  const [batchDeleting, setBatchDeleting] = useState(false)
  const [batchIndexing, setBatchIndexing] = useState(false)
  const [batchMsg, setBatchMsg] = useState('')
  const [batchIndexTask, setBatchIndexTask] = useState<IndexTask | null>(null)

  // Raw text viewer / editor (pre-chunking)
  const [rawTextModalOpen, setRawTextModalOpen] = useState(false)
  const [rawTextLoading, setRawTextLoading] = useState(false)
  const [rawTextSaving, setRawTextSaving] = useState(false)
  const [rawTextContent, setRawTextContent] = useState('')
  const [rawTextDocName, setRawTextDocName] = useState('')
  const [rawTextMsg, setRawTextMsg] = useState('')

  // Chunk viewer (post-indexing)
  const [chunkModalOpen, setChunkModalOpen] = useState(false)
  const [chunkModalDocName, setChunkModalDocName] = useState('')
  const [chunkList, setChunkList] = useState<DocumentChunkItem[]>([])
  const [chunkLoading, setChunkLoading] = useState(false)
  const [graphRule, setGraphRule] = useState<GraphGovernanceConfig | null>(null)
  const mountedRef = useRef(true)
  const pollingTaskIdsRef = useRef<Set<string>>(new Set())
  const taskRunIdRef = useRef(0)

  const isTaskTerminal = (task: IndexTask) =>
    ['succeeded', 'failed', 'partial', 'cancelled'].includes(task.status)

  const pollIndexTask = async (
    taskId: string,
    onUpdate: (task: IndexTask) => void,
    onDone?: (task: IndexTask) => void,
    runId = taskRunIdRef.current,
  ) => {
    if (pollingTaskIdsRef.current.has(taskId)) {
      return getIndexTask(taskId)
    }
    pollingTaskIdsRef.current.add(taskId)
    let finalTask = await getIndexTask(taskId)
    try {
      if (!mountedRef.current || taskRunIdRef.current !== runId) return finalTask
      onUpdate(finalTask)
      while (!isTaskTerminal(finalTask)) {
        await new Promise((resolve) => setTimeout(resolve, 1500))
        finalTask = await getIndexTask(taskId)
        if (!mountedRef.current || taskRunIdRef.current !== runId) return finalTask
        onUpdate(finalTask)
        loadDocs()
      }
      if (mountedRef.current && taskRunIdRef.current === runId) {
        onDone?.(finalTask)
        loadDocs()
      }
      return finalTask
    } finally {
      pollingTaskIdsRef.current.delete(taskId)
    }
  }

  const formatTaskMessage = (task: IndexTask) => {
    if (task.status === 'succeeded') return task.message || '索引完成'
    if (task.status === 'partial') return task.message || '索引部分完成'
    if (task.status === 'failed') {
      const firstError = task.errors[0]?.error
      return firstError ? `索引失败: ${firstError}` : (task.message || '索引失败')
    }
    if (task.status === 'cancelled') return '索引任务已取消'
    return task.message || '索引处理中'
  }

  const formatBatchTaskMessage = (task: IndexTask) => {
    const ok = task.results.filter((r) => r.status === 'ok').length
    const fail = task.errors.length
    return `${formatTaskMessage(task)}: ${ok} 成功${fail > 0 ? `, ${fail} 失败` : ''}`
  }

  const formatGraphResidualWarning = (
    docName: string,
    residuals?: GraphDeleteResiduals,
    cleanupTask?: IndexTask | null,
    cleanupError?: string,
  ) => {
    if (!residuals) return ''
    if (cleanupError) {
      return `已删除 ${docName}，但自动图谱清理启动失败：${cleanupError}`
    }
    if (cleanupTask) {
      return `已删除 ${docName}，检测到图谱残留并已启动自动重建任务 ${cleanupTask.task_id}`
    }
    if (!residuals.checked) {
      return `已删除 ${docName}，但图谱残留检查失败：${residuals.error || '未知错误'}`
    }
    if (!residuals.has_residuals) return ''
    return `已删除 ${docName} 的文档索引，但仍检测到 ${residuals.node_count} 个实体 / ${residuals.edge_count} 条关系引用该文档`
  }

  const batchMessageTone = batchMsg.includes('失败') || batchMsg.includes('取消')
    ? 'error'
    : batchMsg.includes('残留') || batchMsg.includes('重建当前知识库')
      ? 'warning'
      : 'success'

  const isRecentTask = (task: IndexTask) => {
    const updatedAt = new Date(task.updated_at).getTime()
    if (!Number.isFinite(updatedAt)) return false
    return Date.now() - updatedAt <= 60 * 60 * 1000
  }

  const restoreIndexTasks = async (runId = taskRunIdRef.current) => {
    try {
      const tasks = await listIndexTasks()
      if (!mountedRef.current || taskRunIdRef.current !== runId) return
      const candidates = tasks.filter(
        (task) => task.workspace === workspace && (task.kind === 'single' || task.kind === 'batch'),
      )
      const pickTask = (kind: 'single' | 'batch') =>
        candidates.find((task) => task.kind === kind && !isTaskTerminal(task)) ||
        candidates.find((task) => task.kind === kind && isRecentTask(task))

      const singleTask = pickTask('single')
      if (singleTask) {
        setIndexTask(singleTask)
        setIndexing(!isTaskTerminal(singleTask))
        setIndexMsg(
          isTaskTerminal(singleTask)
            ? formatTaskMessage(singleTask)
            : `已恢复索引任务: ${singleTask.task_id}`,
        )
        if (!isTaskTerminal(singleTask)) {
          void pollIndexTask(singleTask.task_id, setIndexTask, (finalTask) => {
            setIndexing(false)
            setIndexMsg(formatTaskMessage(finalTask))
          }, runId)
        }
      }

      const batchTask = pickTask('batch')
      if (batchTask) {
        setBatchIndexTask(batchTask)
        setBatchIndexing(!isTaskTerminal(batchTask))
        setBatchMsg(
          isTaskTerminal(batchTask)
            ? formatBatchTaskMessage(batchTask)
            : `已恢复批量索引任务: ${batchTask.task_id}`,
        )
        if (!isTaskTerminal(batchTask)) {
          void pollIndexTask(batchTask.task_id, setBatchIndexTask, (finalTask) => {
            setBatchIndexing(false)
            setBatchMsg(formatBatchTaskMessage(finalTask))
            if (finalTask.status !== 'failed') setCheckedDocs(new Set())
          }, runId)
        }
      }
    } catch {
      // Task restore is best-effort; document list and manual refresh still work.
    }
  }

  const formatDuration = (iso?: string) => {
    if (!iso) return ''
    const start = new Date(iso).getTime()
    if (!Number.isFinite(start)) return ''
    const seconds = Math.max(0, Math.floor((Date.now() - start) / 1000))
    if (seconds < 60) return `${seconds}s`
    const mins = Math.floor(seconds / 60)
    const rest = seconds % 60
    return `${mins}m ${rest}s`
  }

  const formatClock = (iso?: string) => {
    if (!iso) return ''
    try {
      return new Date(iso).toLocaleTimeString('zh-CN', { hour12: false })
    } catch {
      return ''
    }
  }

  const handleUpload = async (file: File) => {
    setUploading(true)
    setChunks([])
    setIndexMsg('')
    try {
      const data = await uploadDocument(file, workspace)
      setUploaded(data)
      if (data.index_invalidated) {
        setIndexMsg('同名文档内容已更新，旧索引已移除，请重新预览并确认索引。')
      }
    } finally {
      setUploading(false)
    }
  }

  /** Upload multiple files sequentially */
  const handleMultiUpload = async (files: File[]) => {
    setUploading(true)
    setBatchMsg('')
    let lastData: UploadedFile | null = null
    const failures: string[] = []
    let invalidatedCount = 0
    const total = files.length
    for (let i = 0; i < total; i++) {
      const file = files[i]
      if (!file) {
        failures.push(`第 ${i + 1} 个文件: 文件对象为空`)
        setBatchMsg(`上传失败 (${i + 1}/${total}) 第 ${i + 1} 个文件: 文件对象为空`)
        continue
      }
      try {
        const data = await uploadDocument(file, workspace)
        lastData = data
        if (data.index_invalidated) invalidatedCount += 1
        setBatchMsg(`已上传 ${i + 1}/${total}: ${data.file_name}`)
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e)
        failures.push(`${file.name}: ${message}`)
        setBatchMsg(`上传失败 (${i + 1}/${total}) ${file.name}: ${message}`)
      }
    }
    if (lastData) setUploaded(lastData)
    if (failures.length > 0) {
      setBatchMsg(`批量上传完成：成功 ${total - failures.length}/${total}，失败 ${failures.length}。${failures.join('；')}`)
    } else {
      setBatchMsg(
        `批量上传完成：成功 ${total}/${total}` +
        (invalidatedCount > 0 ? `，其中 ${invalidatedCount} 个同名文档的旧索引已移除，需重新索引` : ''),
      )
    }
    setUploading(false)
    loadDocs()
  }

  const handlePreview = async () => {
    if (!uploaded) return
    setPreviewing(true)
    setPreviewError('')
    setIndexMsg('')
    try {
      const sepArray = separators
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .map((s) => s.replace(/\\n/g, '\n'))
      const data = await previewChunks({
        file_name: uploaded.file_name,
        workspace,
        separators: sepArray,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
      })
      setChunks(data)
    } catch (e: unknown) {
      setPreviewError((e as Error).message || '切分预览失败')
      setChunks([])
    } finally {
      setPreviewing(false)
    }
  }

  const handleIndex = async () => {
    if (!uploaded) return
    setIndexing(true)
    setIndexMsg('')
    setIndexTask(null)
    setPreviewError('')
    try {
      const sepArray = separators
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .map((s) => s.replace(/\\n/g, '\n'))
      const task = await indexDocument({
        workspace,
        file_name: uploaded.file_name,
        separators: sepArray,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
      })
      setIndexTask(task)
      setIndexMsg(`索引任务已创建: ${task.task_id}`)
      const finalTask = await pollIndexTask(task.task_id, setIndexTask)
      setIndexMsg(formatTaskMessage(finalTask))
    } catch (e: unknown) {
      setIndexMsg(`索引失败: ${(e as Error).message || '未知错误'}`)
    } finally {
      setIndexing(false)
    }
  }

  const loadDocs = async () => {
    try {
      const data = await listDocuments(workspace)
      setDocs(data)
    } catch {/* ignore */}
  }

  const loadGraphRule = async () => {
    try {
      const data = await getGraphGovernanceConfig(workspace)
      setGraphRule(data)
    } catch {
      setGraphRule(null)
    }
  }

  const handleDelete = async (docName: string) => {
    setDeleting(docName)
    setBatchMsg('')
    try {
      const result = await deleteDocument(docName, workspace)
      const residualWarning = formatGraphResidualWarning(
        result.doc_name,
        result.graph_residuals,
        result.cleanup_task,
        result.cleanup_error,
      )
      setBatchMsg(residualWarning || `已删除文档: ${result.doc_name}`)
      if (result.cleanup_task) {
        setBatchIndexTask(result.cleanup_task)
        setBatchIndexing(!isTaskTerminal(result.cleanup_task))
        if (!isTaskTerminal(result.cleanup_task)) {
          void pollIndexTask(result.cleanup_task.task_id, setBatchIndexTask, (finalTask) => {
            setBatchIndexing(false)
            setBatchMsg(`删除后的图谱清理：${formatTaskMessage(finalTask)}`)
          })
        }
      }
      setCheckedDocs((prev) => {
        const next = new Set(prev)
        next.delete(docName)
        return next
      })
      loadDocs()
    } catch (e: unknown) {
      setBatchMsg(`删除失败: ${(e as Error).message || '未知错误'}`)
    } finally {
      setDeleting(null)
    }
  }

  const toggleCheck = (docName: string) => {
    setCheckedDocs((prev) => {
      const next = new Set(prev)
      next.has(docName) ? next.delete(docName) : next.add(docName)
      return next
    })
  }

  const toggleAll = () => {
    if (checkedDocs.size === docs.length) {
      setCheckedDocs(new Set())
    } else {
      setCheckedDocs(new Set(docs.map((d) => d.doc_name)))
    }
  }

  const handleBatchDelete = async () => {
    if (checkedDocs.size === 0) return
    setBatchDeleting(true)
    setBatchMsg('')
    try {
      const result = await batchDeleteDocuments([...checkedDocs], workspace)
      const residualItems = result.graph_residuals?.items.filter((item) => item.has_residuals || !item.checked) || []
      const errorSuffix = result.errors?.length ? `，${result.errors.length} 个失败` : ''
      if (result.cleanup_error) {
        setBatchMsg(
          `批量删除完成: ${result.deleted_chunks} 个文档${errorSuffix}，但自动图谱清理启动失败：${result.cleanup_error}`,
        )
      } else if (result.cleanup_task) {
        setBatchMsg(
          `批量删除完成: ${result.deleted_chunks} 个文档${errorSuffix}，已启动自动图谱重建任务 ${result.cleanup_task.task_id}`,
        )
        setBatchIndexTask(result.cleanup_task)
        setBatchIndexing(!isTaskTerminal(result.cleanup_task))
        if (!isTaskTerminal(result.cleanup_task)) {
          void pollIndexTask(result.cleanup_task.task_id, setBatchIndexTask, (finalTask) => {
            setBatchIndexing(false)
            setBatchMsg(`删除后的图谱清理：${formatTaskMessage(finalTask)}`)
          })
        }
      } else if (residualItems.length > 0) {
        const nodes = residualItems.reduce((sum, item) => sum + (item.node_count || 0), 0)
        const edges = residualItems.reduce((sum, item) => sum + (item.edge_count || 0), 0)
        setBatchMsg(
          `批量删除完成: ${result.deleted_chunks} 个文档，提交 ${result.doc_count} 个${errorSuffix}。其中 ${residualItems.length} 个文档存在图谱残留，共 ${nodes} 个实体 / ${edges} 条关系。`,
        )
      } else {
        setBatchMsg(`批量删除完成: ${result.deleted_chunks} 个文档，提交 ${result.doc_count} 个${errorSuffix}`)
      }
      setCheckedDocs(new Set())
      loadDocs()
    } catch (e: unknown) {
      setBatchMsg(`批量删除失败: ${(e as Error).message}`)
    } finally {
      setBatchDeleting(false)
    }
  }

  const handleBatchIndex = async () => {
    if (checkedDocs.size === 0) return
    setBatchIndexing(true)
    setBatchMsg('')
    setBatchIndexTask(null)
    try {
      const sepArray = separators
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .map((s) => s.replace(/\\n/g, '\n'))
      const task = await batchIndexDocuments({
        workspace,
        doc_names: [...checkedDocs],
        separators: sepArray,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
      })
      setBatchIndexTask(task)
      setBatchMsg(`批量索引任务已创建: ${task.task_id}`)
      const finalTask = await pollIndexTask(task.task_id, setBatchIndexTask)
      const ok = finalTask.results.filter((r) => r.status === 'ok').length
      const fail = finalTask.errors.length
      setBatchMsg(`${formatTaskMessage(finalTask)}: ${ok} 成功${fail > 0 ? `, ${fail} 失败` : ''}`)
      if (finalTask.status !== 'failed') setCheckedDocs(new Set())
    } catch (e: unknown) {
      setBatchMsg(`批量索引失败: ${(e as Error).message}`)
    } finally {
      setBatchIndexing(false)
    }
  }

  const handleCancelTask = async (task: IndexTask, scope: 'single' | 'batch') => {
    try {
      const updated = await cancelIndexTask(task.task_id)
      if (scope === 'single') {
        setIndexTask(updated)
        setIndexMsg(updated.message || '已请求取消')
      } else {
        setBatchIndexTask(updated)
        setBatchMsg(updated.message || '已请求取消')
      }
    } catch (e: unknown) {
      const msg = `取消失败: ${(e as Error).message}`
      scope === 'single' ? setIndexMsg(msg) : setBatchMsg(msg)
    }
  }

  const renderIndexTask = (task: IndexTask | null, scope: 'single' | 'batch') => {
    if (!task) return null
    const terminal = isTaskTerminal(task)
    const failed = task.status === 'failed' || task.status === 'partial'
    const ok = task.results.filter((r) => r.status === 'ok').length
    const fail = task.errors.length
    const currentElapsed = formatDuration(task.current_doc_started_at)
    const updatedClock = formatClock(task.updated_at)

    return (
      <div className={`mt-3 rounded-lg border p-3 text-sm ${
        failed
          ? 'bg-red-50 border-red-200'
          : terminal
            ? 'bg-green-50 border-green-200'
            : 'bg-primary-50 border-primary-200'
      }`}>
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="font-medium text-gray-800">
              任务 {task.task_id}
              <span className="ml-2 text-xs text-gray-500">{task.status}</span>
            </div>
            <div className="text-xs text-gray-500 mt-0.5">
              {task.message}，进度 {task.current}/{task.total}
              {(ok > 0 || fail > 0) && `，成功 ${ok}，失败 ${fail}`}
            </div>
            {!terminal && (
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-gray-400">
                {task.current_doc && <span>当前文档：{task.current_doc}</span>}
                {currentElapsed && <span>本步耗时：{currentElapsed}</span>}
                {task.timeout_seconds && <span>单文档超时：{task.timeout_seconds}s</span>}
                {updatedClock && <span>最后更新：{updatedClock}</span>}
              </div>
            )}
          </div>
          {!terminal && (
            <button
              onClick={() => handleCancelTask(task, scope)}
              className="px-2.5 py-1 text-xs rounded border border-gray-300 text-gray-600 hover:bg-white"
            >
              取消任务
            </button>
          )}
        </div>
        <div className="mt-3 h-2 rounded-full bg-white overflow-hidden border border-gray-100">
          <div
            className={`h-full transition-all ${
              failed ? 'bg-red-500' : terminal ? 'bg-green-500' : 'bg-primary-500'
            }`}
            style={{ width: `${Math.max(2, task.progress)}%` }}
          />
        </div>
        {task.errors.length > 0 && (
          <div className="mt-2 space-y-1">
            {task.errors.slice(0, 3).map((err) => (
              <div key={err.doc_name} className="text-xs text-red-700">
                {err.doc_name}: {err.error}
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  // --- Raw text viewer / editor ---

  const openRawText = async (docName: string) => {
    setRawTextDocName(docName)
    setRawTextModalOpen(true)
    setRawTextLoading(true)
    setRawTextMsg('')
    try {
      const data = await getDocumentRawText(docName, workspace)
      setRawTextContent(data.raw_text)
    } catch (e: unknown) {
      setRawTextMsg(`加载原始文本失败: ${(e as Error).message}`)
      setRawTextContent('')
    } finally {
      setRawTextLoading(false)
    }
  }

  const saveRawText = async () => {
    if (!rawTextDocName) return
    setRawTextSaving(true)
    setRawTextMsg('')
    try {
      const data = await updateDocumentRawText(rawTextDocName, rawTextContent, workspace)
      setRawTextMsg(`已保存 (${data.char_count} 字符)，旧索引已移除，请重新预览并索引`)
      await loadDocs()
    } catch (e: unknown) {
      setRawTextMsg(`保存失败: ${(e as Error).message}`)
    } finally {
      setRawTextSaving(false)
    }
  }

  // --- Chunk viewer ---

  const openChunks = async (docName: string) => {
    setChunkModalDocName(docName)
    setChunkModalOpen(true)
    setChunkLoading(true)
    setChunkList([])
    try {
      const data = await getDocumentChunks(docName, workspace)
      setChunkList(data.chunks)
    } catch {
      setChunkList([])
    } finally {
      setChunkLoading(false)
    }
  }

  useEffect(() => {
    mountedRef.current = true
    const runId = taskRunIdRef.current + 1
    taskRunIdRef.current = runId
    pollingTaskIdsRef.current.clear()
    setIndexTask(null)
    setBatchIndexTask(null)
    setIndexing(false)
    setBatchIndexing(false)
    setIndexMsg('')
    setBatchMsg('')
    loadDocs()
    loadGraphRule()
    restoreIndexTasks(runId)
    return () => {
      mountedRef.current = false
      taskRunIdRef.current += 1
      pollingTaskIdsRef.current.clear()
    }
  }, [workspace])

  return (
    <div className="space-y-8">
      {/* Page header */}
      <div>
        <h2 className="text-xl font-bold text-gray-800">知识库管理</h2>
        <p className="text-sm text-gray-500 mt-1">
          上传文档、预览切分效果、配置参数后索引到 LightRAG。支持批量操作。
        </p>
      </div>

      {/* Upload section */}
      <section className="bg-white border border-gray-200 rounded-xl p-6">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
          上传文档
        </h3>
        <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="font-medium text-amber-900">
                当前图谱抽取规则：{graphRule?.rule_template_name || '加载中'}
              </div>
              <p className="mt-1 text-xs text-amber-800 leading-relaxed">
                索引时 LightRAG 会按这套规则引导实体和关系抽取。切换规则后，已索引文档需要重新索引才会重建图谱。
              </p>
            </div>
            <div className="shrink-0 text-right text-xs text-amber-800">
              <div>实体类型 {graphRule?.entity_types?.length ?? 0}</div>
              <div>关系类型 {graphRule?.relation_types?.length ?? 0}</div>
            </div>
          </div>
          {graphRule?.extraction_prompt && (
            <p className="mt-2 line-clamp-2 text-xs text-amber-700">
              {graphRule.extraction_prompt}
            </p>
          )}
        </div>
        <FileUpload onUpload={handleUpload} onMultiUpload={handleMultiUpload} uploading={uploading} />

        {uploaded && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="h-4 w-3 rounded-[2px] border border-gray-400 bg-white" />
              <span className="font-medium text-gray-800">{uploaded.file_name}</span>
              <span className="text-xs bg-gray-200 text-gray-600 px-2 py-0.5 rounded">
                {uploaded.file_type}
              </span>
              <span className="text-xs text-gray-400">{uploaded.char_count} 字符</span>
              <button
                onClick={() => openRawText(uploaded.file_name)}
                className="ml-auto text-xs text-primary-600 hover:text-primary-800 border border-primary-300 rounded px-2 py-0.5 hover:bg-primary-50 transition-colors"
              >
                查看/编辑原始文本
              </button>
            </div>
            <p className="mt-2 text-sm text-gray-500 bg-white rounded p-2 max-h-24 overflow-y-auto whitespace-pre-wrap">
              {uploaded.preview}
            </p>
          </div>
        )}

        {batchMsg && uploading && (
          <div className="mt-3 p-3 rounded-lg text-sm bg-primary-50 border border-primary-200 text-primary-700">
            {batchMsg}
          </div>
        )}
      </section>

      {/* Chunking config */}
      <section className="bg-white border border-gray-200 rounded-xl p-6">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
          切分配置
        </h3>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              分隔符 (逗号分隔)
            </label>
            <input
              type="text"
              value={separators}
              onChange={(e) => setSeparators(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-200 focus:border-primary-400 outline-none"
              placeholder="\n\n, \n, 。, ！"
            />
            <p className="text-[11px] text-gray-400 mt-1">
              输入 <code className="bg-gray-100 px-1 rounded">\n</code> 表示换行，
              <code className="bg-gray-100 px-1 rounded">\n\n</code> 表示段落
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Chunk Size: <span className="text-primary-600">{chunkSize}</span>
              </label>
              <input
                type="range"
                min={100} max={2000} step={50}
                value={chunkSize}
                onChange={(e) => setChunkSize(parseInt(e.target.value))}
                className="w-full accent-primary-500"
              />
            <p className="text-[11px] text-gray-400">每个文本块的最大字符数</p>
            <p className="text-[11px] text-amber-600 mt-1">LightRAG 实际按 token 处理，此处作为索引参数传入。</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Overlap: <span className="text-primary-600">{chunkOverlap}</span>
              </label>
              <input
                type="range"
                min={0} max={500} step={10}
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(parseInt(e.target.value))}
                className="w-full accent-primary-500"
              />
            <p className="text-[11px] text-gray-400">相邻文本块的重叠字符数</p>
            </div>
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={handlePreview}
            disabled={!uploaded || previewing}
            className="px-5 py-2 text-sm font-medium rounded-lg bg-white border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {previewing ? '切分中...' : '预览切分'}
          </button>
          <button
            onClick={handleIndex}
            disabled={!uploaded || indexing}
            className="px-5 py-2 text-sm font-medium rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {indexing ? '索引中...' : '确认索引'}
          </button>
        </div>

        {indexMsg && (
          <div className={`mt-3 p-3 rounded-lg text-sm ${
            indexMsg.includes('失败') || indexMsg.includes('取消')
              ? 'bg-red-50 border border-red-200 text-red-700'
              : 'bg-green-50 border border-green-200 text-green-700'
          }`}>
            {indexMsg}
          </div>
        )}
        {renderIndexTask(indexTask, 'single')}
      </section>

      {/* Chunk preview */}
      <section className="bg-white border border-gray-200 rounded-xl p-6">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
          切分预览 {chunks.length > 0 && <span className="text-gray-400">({chunks.length} 块)</span>}
        </h3>
        {previewError && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {previewError}
          </div>
        )}
        <ChunkPreview chunks={chunks} loading={previewing} />
      </section>

      {/* Document list */}
      <section className="bg-white border border-gray-200 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">
            文档清单
          </h3>
          <button onClick={loadDocs} className="text-xs text-primary-500 hover:text-primary-700">
            刷新
          </button>
        </div>

        {/* Batch action bar */}
        {checkedDocs.size > 0 && (
          <div className="mb-4 flex items-center gap-3 p-3 bg-primary-50 border border-primary-200 rounded-lg">
            <span className="text-sm text-primary-700 font-medium">
              已选 {checkedDocs.size} 个文档
            </span>
            <button
              onClick={handleBatchIndex}
              disabled={batchIndexing}
              className="px-3 py-1.5 text-xs font-medium rounded bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 transition-colors"
            >
              {batchIndexing ? '索引中...' : '批量索引'}
            </button>
            <button
              onClick={handleBatchDelete}
              disabled={batchDeleting}
              className="px-3 py-1.5 text-xs font-medium rounded bg-red-500 text-white hover:bg-red-600 disabled:opacity-50 transition-colors"
            >
              {batchDeleting ? '删除中...' : '批量删除'}
            </button>
            <button
              onClick={() => setCheckedDocs(new Set())}
              className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-700"
            >
              取消选择
            </button>
          </div>
        )}

        {batchMsg && (
          <div className={`mb-4 p-3 rounded-lg text-sm ${
            batchMessageTone === 'error'
              ? 'bg-red-50 border border-red-200 text-red-700'
              : batchMessageTone === 'warning'
                ? 'bg-amber-50 border border-amber-200 text-amber-800'
              : 'bg-green-50 border border-green-200 text-green-700'
          }`}>
            {batchMsg}
          </div>
        )}
        {renderIndexTask(batchIndexTask, 'batch')}

        {docs.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-6">暂无文档</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 text-left">
                  <th className="pb-2 w-8">
                    <input
                      type="checkbox"
                      checked={checkedDocs.size === docs.length && docs.length > 0}
                      onChange={toggleAll}
                      className="w-4 h-4 rounded accent-primary-500"
                    />
                  </th>
                  <th className="pb-2 font-medium text-gray-500">文档名</th>
                  <th className="pb-2 font-medium text-gray-500">类型</th>
                  <th className="pb-2 font-medium text-gray-500">块数</th>
                  <th className="pb-2 font-medium text-gray-500">抽取规则</th>
                  <th className="pb-2 font-medium text-gray-500">状态</th>
                  <th className="pb-2 font-medium text-gray-500 text-right">操作</th>
                </tr>
              </thead>
              <tbody>
                {docs.map((doc) => (
                  <tr key={doc.doc_id || doc.doc_name} className="border-b border-gray-50">
                    <td className="py-2.5">
                      <input
                        type="checkbox"
                        checked={checkedDocs.has(doc.doc_name)}
                        onChange={() => toggleCheck(doc.doc_name)}
                        className="w-4 h-4 rounded accent-primary-500"
                      />
                    </td>
                    <td className="py-2.5 font-mono text-xs">
                      <button
                        onClick={() => openChunks(doc.doc_name)}
                        className="text-primary-600 hover:text-primary-800 hover:underline text-left cursor-pointer"
                        title="点击查看文本块详情"
                      >
                        {doc.doc_name}
                      </button>
                    </td>
                    <td className="py-2.5">
                      <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
                        {doc.file_type}
                      </span>
                    </td>
                    <td className="py-2.5 text-gray-600">{doc.chunk_count}</td>
                    <td className="py-2.5 text-gray-600">
                      <span
                        title={doc.graph_rule?.extraction_prompt_preview || ''}
                        className="text-xs bg-amber-50 text-amber-700 px-2 py-0.5 rounded"
                      >
                        {doc.graph_rule?.rule_template_name || graphRule?.rule_template_name || '未记录'}
                      </span>
                    </td>
                    <td className="py-2.5">
                      <span
                        title={(doc as DocInfo & { error_msg?: string }).error_msg || ''}
                        className={`text-xs px-2 py-0.5 rounded ${
                          doc.status === 'failed'
                            ? 'bg-red-50 text-red-700'
                            : doc.indexed
                              ? 'bg-green-50 text-green-700'
                              : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {doc.status || (doc.indexed ? 'indexed' : 'uploaded')}
                      </span>
                    </td>
                    <td className="py-2.5 text-right">
                      <button
                        onClick={() => handleDelete(doc.doc_name)}
                        disabled={deleting === doc.doc_name}
                        className="text-xs text-red-500 hover:text-red-700 disabled:opacity-50"
                      >
                        {deleting === doc.doc_name ? '删除中...' : '删除'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Raw Text Editor Modal */}
      {rawTextModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col mx-4">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
              <div>
                <h3 className="text-lg font-semibold text-gray-800">原始文本编辑器</h3>
                <p className="text-xs text-gray-400 mt-0.5">{rawTextDocName}</p>
              </div>
              <button
                onClick={() => setRawTextModalOpen(false)}
                className="text-gray-400 hover:text-gray-600 text-xl leading-none"
              >
                ✕
              </button>
            </div>

            {/* Body */}
            {rawTextLoading ? (
              <div className="flex-1 flex items-center justify-center p-12">
                <div className="flex items-center gap-3 text-gray-400">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  加载中...
                </div>
              </div>
            ) : (
              <>
                <div className="flex-1 overflow-auto p-6">
                  <textarea
                    value={rawTextContent}
                    onChange={(e) => setRawTextContent(e.target.value)}
                    className="w-full h-full min-h-[400px] text-sm font-mono border border-gray-300 rounded-lg p-4 focus:ring-2 focus:ring-primary-200 focus:border-primary-400 outline-none resize-none"
                    placeholder="暂无内容"
                  />
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 bg-gray-50 rounded-b-xl">
                  <div className="flex-1">
                    {rawTextMsg && (
                      <p className={`text-sm ${
                        rawTextMsg.includes('失败') ? 'text-red-600' : 'text-green-600'
                      }`}>
                        {rawTextMsg}
                      </p>
                    )}
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={() => setRawTextModalOpen(false)}
                      className="px-4 py-2 text-sm rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-100 transition-colors"
                    >
                      取消
                    </button>
                    <button
                      onClick={saveRawText}
                      disabled={rawTextSaving}
                      className="px-4 py-2 text-sm font-medium rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 transition-colors"
                    >
                      {rawTextSaving ? '保存中...' : '保存修改'}
                    </button>
                  </div>
                </div>
                <p className="text-xs text-amber-600 px-6 pb-2 bg-gray-50">
                  修改后旧索引会立即移除，需要重新预览并确认索引。
                </p>
              </>
            )}
          </div>
        </div>
      )}

      {/* Chunk Viewer Modal */}
      {chunkModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col mx-4">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
              <div>
                <h3 className="text-lg font-semibold text-gray-800">文本块详情</h3>
                <p className="text-xs text-gray-400 mt-0.5">
                  {chunkModalDocName}
                  {!chunkLoading && <span className="ml-2">— {chunkList.length} 个文本块</span>}
                </p>
              </div>
              <button
                onClick={() => setChunkModalOpen(false)}
                className="text-gray-400 hover:text-gray-600 text-xl leading-none"
              >
                ✕
              </button>
            </div>

            {/* Body */}
            {chunkLoading ? (
              <div className="flex-1 flex items-center justify-center p-12">
                <div className="flex items-center gap-3 text-gray-400">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  加载中...
                </div>
              </div>
            ) : chunkList.length === 0 ? (
              <div className="flex-1 flex items-center justify-center p-12 text-gray-400 text-sm">
                暂无文本块数据
              </div>
            ) : (
              <div className="flex-1 overflow-auto p-6 space-y-4">
                {chunkList.map((chunk) => (
                  <div
                    key={chunk.chunk_id}
                    className="border border-gray-200 rounded-lg overflow-hidden"
                  >
                    <div className="flex items-center gap-3 px-4 py-2 bg-gray-50 border-b border-gray-100">
                      <span className="text-xs font-mono font-semibold text-primary-600 bg-primary-50 px-2 py-0.5 rounded">
                        块 #{chunk.chunk_index}
                      </span>
                      <span className="text-xs text-gray-400">{chunk.char_count} 字符</span>
                      <span className="text-[10px] text-gray-300 font-mono ml-auto truncate max-w-[200px]" title={chunk.chunk_id}>
                        {chunk.chunk_id}
                      </span>
                    </div>
                    <div className="px-4 py-3 text-sm text-gray-700 whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
                      {chunk.text || <span className="text-gray-300 italic">（空文本）</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Footer */}
            <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 rounded-b-xl flex justify-end">
              <button
                onClick={() => setChunkModalOpen(false)}
                className="px-4 py-2 text-sm rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-100 transition-colors"
              >
                关闭
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
