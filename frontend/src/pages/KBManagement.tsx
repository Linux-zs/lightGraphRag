import { useEffect, useRef, useState } from 'react'
import { ChevronDown, ShieldCheck, SlidersHorizontal, Trash2 } from 'lucide-react'
import FileUpload from '../components/FileUpload'
import ChunkPreview from '../components/ChunkPreview'
import { RangeField } from '../components/ui'
import {
  uploadDocument,
  previewChunks,
  indexDocument,
  listDocuments,
  deleteDocument,
  batchDeleteDocuments,
  batchIndexDocuments,
  backfillDocumentGraph,
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
type KGExtractionDensity = 'sparse' | 'standard' | 'dense' | 'custom'

const KG_DENSITY_PRESETS: Record<
  Exclude<KGExtractionDensity, 'custom'>,
  { label: string; entities: number; records: number; hint: string }
> = {
  sparse: { label: '精简', entities: 8, records: 16, hint: '术语少、文本短，优先降噪' },
  standard: { label: '标准', entities: 24, records: 48, hint: '适合大多数技术和业务文档' },
  dense: { label: '密集', entities: 40, records: 80, hint: '实体关系丰富的规范资料' },
}

interface Props {
  workspace: string
  isDefaultWorkspace: boolean
  onDeleteWorkspace: () => Promise<void>
}

export default function KBManagement({
  workspace,
  isDefaultWorkspace,
  onDeleteWorkspace,
}: Props) {
  // Upload state
  const [uploaded, setUploaded] = useState<UploadedFile | null>(null)
  const [uploading, setUploading] = useState(false)

  // Chunking params
  const [separators, setSeparators] = useState('\\n\\n, \\n, 。, ！, ？, ；,  ')
  const [chunkSize, setChunkSize] = useState(1024)
  const [chunkOverlap, setChunkOverlap] = useState(100)
  const [indexMode, setIndexMode] = useState<'complete' | 'fast'>('complete')
  const [kgDensity, setKgDensity] = useState<KGExtractionDensity>('standard')
  const [kgMaxEntities, setKgMaxEntities] = useState(24)
  const [kgMaxRecords, setKgMaxRecords] = useState(48)
  const [advancedIndexOpen, setAdvancedIndexOpen] = useState(false)

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
  const [deletingWorkspace, setDeletingWorkspace] = useState(false)
  const [workspaceDeleteError, setWorkspaceDeleteError] = useState('')

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
        (task) => task.workspace === workspace &&
          (task.kind === 'single' || task.kind === 'batch' || task.kind === 'kg_backfill'),
      )
      const pickTask = (kind: 'single' | 'batch') => {
        const matches = (task: IndexTask) => kind === 'single'
          ? task.kind === 'single'
          : task.kind === 'batch' || task.kind === 'kg_backfill'
        return candidates.find((task) => matches(task) && !isTaskTerminal(task)) ||
          candidates.find((task) => matches(task) && isRecentTask(task))
      }

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

  const formatStageSeconds = (seconds: number) => `${seconds.toFixed(1)}s`

  const kgStatusLabel = (status?: string) => {
    if (status === 'complete') return '已建图谱'
    if (status === 'partial') return '部分图谱'
    if (status === 'skipped') return '跳过KG'
    if (status === 'filtered_empty') return '无有效KG块'
    if (status === 'failed') return 'KG失败'
    return status || '未记录'
  }

  const kgStatusClass = (status?: string) => {
    if (status === 'complete') return 'bg-violet-50 text-violet-700'
    if (status === 'partial') return 'bg-amber-50 text-amber-700'
    if (status === 'skipped') return 'bg-gray-100 text-gray-600'
    if (status === 'filtered_empty') return 'bg-amber-50 text-amber-700'
    if (status === 'failed') return 'bg-red-50 text-red-700'
    return 'bg-gray-100 text-gray-600'
  }

  const selectKgDensity = (density: KGExtractionDensity) => {
    setKgDensity(density)
    if (density === 'custom') return
    const preset = KG_DENSITY_PRESETS[density]
    setKgMaxEntities(preset.entities)
    setKgMaxRecords(preset.records)
  }

  const canBackfillGraph = (doc: DocInfo) =>
    Boolean(doc.indexed) && doc.kg_status !== 'complete'

  const handleDeleteWorkspace = async () => {
    if (isDefaultWorkspace || deletingWorkspace) return
    setDeletingWorkspace(true)
    setWorkspaceDeleteError('')
    try {
      await onDeleteWorkspace()
    } catch (error) {
      setWorkspaceDeleteError((error as Error).message || '删除知识库失败')
    } finally {
      setDeletingWorkspace(false)
    }
  }

  const activeStageSeconds = (startedAt: string | undefined, fallback: number) => {
    if (!startedAt) return fallback
    const start = new Date(startedAt).getTime()
    if (!Number.isFinite(start)) return fallback
    return Math.max(fallback, (Date.now() - start) / 1000)
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
        index_mode: indexMode,
        kg_max_entities: kgMaxEntities,
        kg_max_records: kgMaxRecords,
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
        index_mode: indexMode,
        kg_max_entities: kgMaxEntities,
        kg_max_records: kgMaxRecords,
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

  const handleGraphBackfill = async (docNames: string[]) => {
    const eligible = docNames.filter((name) => {
      const doc = docs.find((item) => item.doc_name === name)
      return doc ? canBackfillGraph(doc) : false
    })
    if (eligible.length === 0) return
    setBatchIndexing(true)
    setBatchMsg('')
    setBatchIndexTask(null)
    try {
      const task = await backfillDocumentGraph({
        workspace,
        doc_names: eligible,
        kg_max_entities: kgMaxEntities,
        kg_max_records: kgMaxRecords,
      })
      setBatchIndexTask(task)
      setBatchMsg(`图谱补建任务已创建: ${task.task_id}`)
      const finalTask = await pollIndexTask(task.task_id, setBatchIndexTask)
      setBatchMsg(formatBatchTaskMessage(finalTask))
      if (finalTask.status !== 'failed') setCheckedDocs(new Set())
    } catch (e: unknown) {
      setBatchMsg(`图谱补建失败: ${(e as Error).message || '未知错误'}`)
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
    const stageTimings = task.stage_timings ?? { parse: 0, chunk_vector: 0, kg: 0, merge: 0 }
    const currentStage = task.current_stage
    const taskMode = task.kind === 'kg_backfill'
      ? '图谱补建'
      : task.request?.index_mode === 'fast'
        ? '快速索引'
        : '完整索引'
    const statusDotClass = failed
      ? 'bg-red-500'
      : terminal
        ? 'bg-green-500'
        : 'bg-primary-500 animate-pulse'
    const ALL_STAGE_CARDS: { key: keyof NonNullable<IndexTask['stage_timings']>; label: string }[] = [
      { key: 'parse', label: '解析' },
      { key: 'chunk_vector', label: 'Chunk向量' },
      { key: 'kg', label: 'KG抽取' },
      { key: 'merge', label: '图谱/落盘' },
    ]
    const STAGE_CARDS = task.kind === 'kg_backfill'
      ? ALL_STAGE_CARDS.filter((stage) => stage.key === 'kg' || stage.key === 'merge')
      : ALL_STAGE_CARDS

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
              <span className={`mr-2 inline-block h-2.5 w-2.5 rounded-full ${statusDotClass}`} />
              任务 {task.task_id}
              <span className="ml-2 text-xs text-gray-500">{task.status}</span>
              <span className="ml-2 rounded bg-white/70 px-1.5 py-0.5 text-[11px] text-gray-500">{taskMode}</span>
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
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
          {STAGE_CARDS.map(({ key, label }) => {
            const v = stageTimings[key]
            const active = !terminal && currentStage === key
            const displaySeconds = active
              ? activeStageSeconds(task.current_stage_started_at, typeof v === 'number' ? v : 0)
              : typeof v === 'number' ? v : 0
            return (
              <div
                key={key}
                className={`rounded-md border bg-white px-2.5 py-2 ${
                  active ? 'border-primary-300 ring-1 ring-primary-100' : 'border-gray-200'
                }`}
              >
                <div className="flex items-center justify-between gap-2 text-[11px] text-gray-400">
                  <span>{label}</span>
                  {active && <span className="text-primary-600">进行中</span>}
                </div>
                <div className="text-sm font-semibold text-gray-800">
                  {formatStageSeconds(displaySeconds)}
                </div>
              </div>
            )
          })}
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
        {task.results.some((result) => result.kg_status === 'partial') && (
          <div className="mt-2 space-y-1 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
            {task.results
              .filter((result) => result.kg_status === 'partial')
              .map((result) => (
                <div key={`${result.doc_name}-kg-partial`}>
                  {result.doc_name}: 已保留文本和向量，跳过{' '}
                  {result.kg_timed_out_chunks?.length || 0} 个超时 KG 块
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
    setWorkspaceDeleteError('')
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
    <div className="space-y-5 sm:space-y-8">
      {/* Page header */}
      <div>
        <h2 className="text-xl font-bold text-gray-800">知识库管理</h2>
        <p className="text-sm text-gray-500 mt-1">
          上传文档、预览切分效果、配置参数后索引到 LightRAG。支持批量操作。
        </p>
      </div>

      {/* Upload section */}
      <section className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
          上传文档
        </h3>
        <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium text-amber-900">
              抽取规则：{graphRule?.rule_template_name || '加载中'}
            </span>
            <span className="rounded bg-white/70 px-2 py-0.5 text-[11px] text-amber-800">
              {graphRule?.entity_types?.length ?? 0} 类实体
            </span>
            <span className="rounded bg-white/70 px-2 py-0.5 text-[11px] text-amber-800">
              {graphRule?.relation_types?.length ?? 0} 类关系
            </span>
          </div>
          <details className="mt-2 text-xs text-amber-800">
            <summary className="cursor-pointer select-none font-medium">查看规则说明</summary>
            <p className="mt-2 leading-relaxed">
              索引时 LightRAG 会按这套规则引导实体和关系抽取。切换规则后，已索引文档需要重新索引才会重建图谱。
            </p>
            {graphRule?.extraction_prompt && (
              <p className="mt-2 line-clamp-3 text-amber-700">{graphRule.extraction_prompt}</p>
            )}
          </details>
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

      {/* Index config */}
      <section className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
          索引配置
        </h3>

        <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
          <div className="mb-2 text-sm font-medium text-gray-700">索引模式</div>
          <div className="grid gap-2 md:grid-cols-2">
            <button
              type="button"
              onClick={() => setIndexMode('complete')}
              className={`rounded-lg border px-3 py-2 text-left transition-colors ${
                indexMode === 'complete'
                  ? 'border-primary-300 bg-white text-gray-900 ring-1 ring-primary-100'
                  : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
              }`}
            >
              <div className="text-sm font-medium">完整索引</div>
              <div className="mt-0.5 text-xs text-gray-500">生成向量并抽取实体关系，问答和图谱都完整。</div>
            </button>
            <button
              type="button"
              onClick={() => setIndexMode('fast')}
              className={`rounded-lg border px-3 py-2 text-left transition-colors ${
                indexMode === 'fast'
                  ? 'border-primary-300 bg-white text-gray-900 ring-1 ring-primary-100'
                  : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300'
              }`}
            >
              <div className="text-sm font-medium">快速索引</div>
              <div className="mt-0.5 text-xs text-gray-500">只写入文本块和向量，跳过 KG 抽取，适合先验证问答。</div>
            </button>
          </div>
        </div>

        <div className="mt-3 rounded-lg border border-gray-200 p-3">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <div className="text-sm font-medium text-gray-700">图谱抽取密度</div>
              <p className="mt-0.5 text-xs text-gray-500">
                数量是每个文本块的上限，不是抽取目标；内容不足时不会强行凑满。
              </p>
            </div>
            <span className="text-xs text-gray-400">用于完整索引和后续图谱补建</span>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {(Object.entries(KG_DENSITY_PRESETS) as [
              Exclude<KGExtractionDensity, 'custom'>,
              (typeof KG_DENSITY_PRESETS)[Exclude<KGExtractionDensity, 'custom'>],
            ][]).map(([key, preset]) => (
              <button
                key={key}
                type="button"
                onClick={() => selectKgDensity(key)}
                className={`rounded-lg border px-3 py-2 text-left transition ${
                  kgDensity === key
                    ? 'border-primary-300 bg-primary-50 ring-1 ring-primary-100'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
              >
                <span className="block text-sm font-medium text-gray-800">{preset.label}</span>
                <span className="mt-0.5 block text-[11px] text-gray-500">
                  实体 {preset.entities} / 记录 {preset.records}
                </span>
                <span className="mt-1 block text-[11px] leading-4 text-gray-400">{preset.hint}</span>
              </button>
            ))}
            <button
              type="button"
              onClick={() => selectKgDensity('custom')}
              className={`rounded-lg border px-3 py-2 text-left transition ${
                kgDensity === 'custom'
                  ? 'border-primary-300 bg-primary-50 ring-1 ring-primary-100'
                  : 'border-gray-200 bg-white hover:border-gray-300'
              }`}
            >
              <span className="block text-sm font-medium text-gray-800">自定义</span>
              <span className="mt-0.5 block text-[11px] text-gray-500">按文档复杂度设置上限</span>
            </button>
          </div>
          <div className="mt-3 grid max-w-xl grid-cols-1 gap-3 sm:grid-cols-2">
            <label>
              <span className="ui-label">每块实体上限</span>
              <input
                type="number"
                min={1}
                max={200}
                value={kgMaxEntities}
                onChange={(e) => {
                  setKgDensity('custom')
                  setKgMaxEntities(Math.max(1, Math.min(200, Number(e.target.value) || 1)))
                }}
                className="ui-control w-full"
              />
            </label>
            <label>
              <span className="ui-label">每块实体与关系统计记录上限</span>
              <input
                type="number"
                min={1}
                max={400}
                value={kgMaxRecords}
                onChange={(e) => {
                  setKgDensity('custom')
                  setKgMaxRecords(Math.max(1, Math.min(400, Number(e.target.value) || 1)))
                }}
                className="ui-control w-full"
              />
            </label>
          </div>
        </div>

        <button
          type="button"
          onClick={() => setAdvancedIndexOpen((open) => !open)}
          className="mt-4 flex w-full items-center gap-3 rounded-lg border border-gray-200 px-3 py-2.5 text-left transition hover:bg-gray-50"
          aria-expanded={advancedIndexOpen}
        >
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-gray-100 text-gray-500">
            <SlidersHorizontal size={16} />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block text-sm font-medium text-gray-800">高级索引设置</span>
            <span className="block truncate text-xs text-gray-500">
              Chunk {chunkSize} · Overlap {chunkOverlap} · 自定义分隔符
            </span>
          </span>
          <ChevronDown
            size={16}
            className={`shrink-0 text-gray-400 transition-transform ${advancedIndexOpen ? 'rotate-180' : ''}`}
          />
        </button>

        {advancedIndexOpen && (
          <div className="mt-3 grid grid-cols-1 gap-6 rounded-lg border border-gray-200 bg-gray-50/60 p-4 lg:grid-cols-[minmax(0,1fr)_minmax(320px,0.85fr)]">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                分隔符（逗号分隔）
              </label>
              <input
                type="text"
                value={separators}
                onChange={(e) => setSeparators(e.target.value)}
                className="ui-control w-full"
                placeholder="\n\n, \n, 。, ！"
              />
              <p className="mt-1 text-xs text-gray-400">
                <code className="rounded bg-gray-100 px-1">\n</code> 表示换行，
                <code className="rounded bg-gray-100 px-1">\n\n</code> 表示段落
              </p>
            </div>

            <div className="space-y-4">
              <div>
                <RangeField
                  label="Chunk Size"
                  value={chunkSize}
                  min={100}
                  max={2000}
                  step={50}
                  onChange={setChunkSize}
                />
                <p className="text-xs text-gray-400">每个文本块的最大 token 数</p>
              </div>
              <div>
                <RangeField
                  label="Overlap"
                  value={chunkOverlap}
                  min={0}
                  max={500}
                  step={10}
                  onChange={setChunkOverlap}
                />
                <p className="text-xs text-gray-400">相邻文本块的重叠 token 数</p>
              </div>
            </div>
          </div>
        )}

        <div className="mt-5 flex flex-wrap gap-3">
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
      <section className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
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
      <section className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
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
            {docs.some((doc) => checkedDocs.has(doc.doc_name) && canBackfillGraph(doc)) && (
              <button
                onClick={() => handleGraphBackfill([...checkedDocs])}
                disabled={batchIndexing}
                className="px-3 py-1.5 text-xs font-medium rounded border border-violet-200 bg-white text-violet-700 hover:bg-violet-50 disabled:opacity-50 transition-colors"
              >
                {batchIndexing ? '处理中...' : '补建图谱'}
              </button>
            )}
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
                  <th className="pb-2 font-medium text-gray-500">KG</th>
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
                    <td className="py-2.5">
                      <span
                        title={[
                          doc.kg_model,
                          doc.kg_extraction_limits?.max_entities_per_chunk
                            ? `每块实体上限 ${doc.kg_extraction_limits.max_entities_per_chunk}，记录上限 ${doc.kg_extraction_limits.max_records_per_chunk}`
                            : '',
                        ].filter(Boolean).join(' · ')}
                        className={`text-xs px-2 py-0.5 rounded ${kgStatusClass(doc.kg_status)}`}
                      >
                        {kgStatusLabel(doc.kg_status)}
                      </span>
                    </td>
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
                    <td className="py-2.5">
                      <div className="flex items-center justify-end gap-3">
                        {canBackfillGraph(doc) && (
                          <button
                            onClick={() => handleGraphBackfill([doc.doc_name])}
                            disabled={batchIndexing}
                            className="text-xs text-violet-600 hover:text-violet-800 disabled:opacity-50"
                            title="复用已有文本块，只补建实体和关系"
                          >
                            补建图谱
                          </button>
                        )}
                        <button
                          onClick={() => handleDelete(doc.doc_name)}
                          disabled={deleting === doc.doc_name}
                          className="text-xs text-red-500 hover:text-red-700 disabled:opacity-50"
                        >
                          {deleting === doc.doc_name ? '删除中...' : '删除'}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Workspace settings */}
      <section className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 items-start gap-3">
            <span
              className={`grid h-9 w-9 shrink-0 place-items-center rounded-md ${
                isDefaultWorkspace
                  ? 'bg-gray-100 text-gray-500'
                  : 'bg-red-50 text-red-600'
              }`}
            >
              {isDefaultWorkspace ? <ShieldCheck size={17} /> : <Trash2 size={17} />}
            </span>
            <div className="min-w-0">
              <h3 className="text-sm font-semibold text-gray-800">知识库设置</h3>
              <p className="mt-1 text-xs leading-5 text-gray-500">
                {isDefaultWorkspace
                  ? '当前为默认知识库，系统已保护其工作区，不能删除。'
                  : `删除“${workspace}”会同时移除上传文档、索引、图谱和会话入口。`}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleDeleteWorkspace}
            disabled={isDefaultWorkspace || deletingWorkspace}
            className="ui-button shrink-0 self-start border border-red-200 bg-white text-red-600 hover:border-red-300 hover:bg-red-50 disabled:cursor-not-allowed disabled:border-gray-200 disabled:text-gray-300 disabled:hover:bg-white sm:self-auto"
          >
            <Trash2 size={15} />
            {deletingWorkspace ? '删除中...' : '删除知识库'}
          </button>
        </div>
        {workspaceDeleteError && (
          <p className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {workspaceDeleteError}
          </p>
        )}
      </section>

      {/* Raw Text Editor Modal */}
      {rawTextModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="mx-4 flex max-h-[85vh] w-full max-w-4xl flex-col rounded-lg bg-white shadow-2xl">
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
          <div className="mx-4 flex max-h-[85vh] w-full max-w-4xl flex-col rounded-lg bg-white shadow-2xl">
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
