import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ChevronDown,
  FileText,
  GitBranch,
  Layers3,
  ListChecks,
  Network,
  RefreshCw,
  ScrollText,
} from 'lucide-react'
import { useConfirm } from '../components/ConfirmDialog'
import {
  clearKnowledgeBase,
  getIndexTask,
  getSystemLogs,
  getSystemStats,
  listIndexTasks,
  rebuildIndex,
  IndexTask,
  SystemLogItem,
  SystemStats,
} from '../api'

interface Props {
  workspace: string
  onWorkspaceChanged?: () => void
}

export default function Dashboard({ workspace, onWorkspaceChanged }: Props) {
  const confirm = useConfirm()
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [opMsg, setOpMsg] = useState('')
  const [operating, setOperating] = useState(false)
  const [rebuildTask, setRebuildTask] = useState<IndexTask | null>(null)
  const [recentTasks, setRecentTasks] = useState<IndexTask[]>([])
  const [logs, setLogs] = useState<SystemLogItem[]>([])
  const [logLevel, setLogLevel] = useState('')
  const [logContains, setLogContains] = useState('')
  const [logMsg, setLogMsg] = useState('')
  const [logsLoading, setLogsLoading] = useState(false)
  const [recentTasksOpen, setRecentTasksOpen] = useState(false)
  const pollingTaskIdRef = useRef<string | null>(null)
  const logViewportRef = useRef<HTMLDivElement | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    loadStats()
    restoreRebuildTask()
    loadObservability()
    return () => {
      mountedRef.current = false
      pollingTaskIdRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace])

  const loadStats = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const data = await getSystemStats(workspace)
      if (!mountedRef.current) return
      setStats(data)
    } catch (e: unknown) {
      if (!mountedRef.current) return
      setError((e as Error).message || '加载失败')
    } finally {
      if (mountedRef.current) setLoading(false)
    }
  }, [workspace])

  const isTaskTerminal = (task: IndexTask) =>
    ['succeeded', 'failed', 'partial', 'cancelled'].includes(task.status)

  const taskTone = (task: IndexTask) => {
    if (task.status === 'failed' || task.status === 'partial') return 'red'
    if (task.status === 'cancelled') return 'gray'
    if (isTaskTerminal(task)) return 'green'
    return 'primary'
  }

  const formatTaskTime = (iso?: string) => {
    if (!iso) return ''
    try {
      return new Date(iso).toLocaleTimeString('zh-CN', { hour12: false })
    } catch {
      return ''
    }
  }

  const formatStageSeconds = (seconds: number) => `${seconds.toFixed(1)}s`

  const activeStageSeconds = (startedAt: string | undefined, fallback: number) => {
    if (!startedAt) return fallback
    const start = new Date(startedAt).getTime()
    if (!Number.isFinite(start)) return fallback
    return Math.max(fallback, (Date.now() - start) / 1000)
  }

  const formatTaskKind = (kind: IndexTask['kind']) => {
    if (kind === 'single') return '单文档索引'
    if (kind === 'batch') return '批量索引'
    return '重建索引'
  }

  const formatIndexStage = (stage: NonNullable<IndexTask['current_stage']>) => {
    if (stage === 'parse') return '解析'
    if (stage === 'chunk_vector') return 'Chunk 向量'
    if (stage === 'kg') return 'KG 抽取'
    if (stage === 'merge') return '图谱/落盘'
    return ''
  }

  const formatTaskStatus = (status: IndexTask['status']) => {
    if (status === 'queued') return '等待中'
    if (status === 'running') return '进行中'
    if (status === 'succeeded') return '已完成'
    if (status === 'partial') return '部分完成'
    if (status === 'failed') return '失败'
    return '已取消'
  }

  const taskDotClass = (task: IndexTask) => {
    if (task.status === 'failed' || task.status === 'partial') return 'bg-red-500'
    if (task.status === 'cancelled') return 'bg-gray-400'
    if (isTaskTerminal(task)) return 'bg-emerald-500'
    return 'bg-blue-500 animate-pulse'
  }

  const loadObservability = useCallback(async () => {
    setLogsLoading(true)
    setLogMsg('')
    try {
      const [tasks, logData] = await Promise.all([
        listIndexTasks(),
        getSystemLogs({
          limit: 200,
          level: logLevel,
          contains: logContains.trim(),
        }),
      ])
      if (!mountedRef.current) return
      setRecentTasks(tasks.filter((task) => task.workspace === workspace).slice(0, 8))
      setLogs(logData.items)
      setLogMsg(
        logData.exists
          ? `匹配 ${logData.total_matched} 行，显示最近 ${logData.items.length} 行`
          : '日志文件尚未生成',
      )
    } catch (e: unknown) {
      if (!mountedRef.current) return
      setLogMsg(`加载日志失败: ${(e as Error).message}`)
    } finally {
      if (mountedRef.current) setLogsLoading(false)
    }
  }, [logContains, logLevel, workspace])

  useEffect(() => {
    if (logsLoading || logs.length === 0) return
    const frame = window.requestAnimationFrame(() => {
      const viewport = logViewportRef.current
      if (viewport) viewport.scrollTop = viewport.scrollHeight
    })
    return () => window.cancelAnimationFrame(frame)
  }, [logs, logsLoading])

  useEffect(() => {
    const hasActiveTask = recentTasks.some((task) => !isTaskTerminal(task))
    if (!hasActiveTask) return
    const timer = window.setInterval(() => {
      void loadObservability()
    }, 3000)
    return () => window.clearInterval(timer)
  }, [loadObservability, recentTasks])

  const pollTask = useCallback(async (taskId: string) => {
    if (pollingTaskIdRef.current === taskId) return
    pollingTaskIdRef.current = taskId
    let task = await getIndexTask(taskId)
    if (!mountedRef.current || pollingTaskIdRef.current !== taskId) return
    setRebuildTask(task)
    setOperating(!isTaskTerminal(task))
    while (!isTaskTerminal(task)) {
      await new Promise((resolve) => setTimeout(resolve, 1500))
      task = await getIndexTask(taskId)
      if (!mountedRef.current || pollingTaskIdRef.current !== taskId) return
      setRebuildTask(task)
    }
    if (!mountedRef.current || pollingTaskIdRef.current !== taskId) return
    setOperating(false)
    setOpMsg(task.message)
    await loadStats()
    await loadObservability()
  }, [loadObservability, loadStats])

  const restoreRebuildTask = useCallback(async () => {
    try {
      const tasks = await listIndexTasks()
      if (!mountedRef.current) return
      setRecentTasks(tasks.filter((item) => item.workspace === workspace).slice(0, 8))
      const task = tasks.find((item) => item.workspace === workspace && item.kind === 'rebuild')
      if (!task) {
        setRebuildTask(null)
        setOperating(false)
        return
      }
      setRebuildTask(task)
      setOperating(!isTaskTerminal(task))
      if (isTaskTerminal(task)) {
        setOpMsg(task.message || '')
      } else {
        setOpMsg(`已恢复重建任务: ${task.task_id}`)
        void pollTask(task.task_id)
      }
    } catch {
      // Task recovery is best-effort; stats still load independently.
    }
  }, [workspace, pollTask])

  const handleClear = async () => {
    const confirmed = await confirm({
      title: '清空当前知识库索引',
      message: `将清空知识库“${workspace}”的 LightRAG 索引和知识图谱，上传的原始文档会保留。`,
      confirmLabel: '清空索引',
      tone: 'danger',
    })
    if (!confirmed) return
    setOperating(true)
    setOpMsg('')
    setRebuildTask(null)
    pollingTaskIdRef.current = null
    try {
      const result = await clearKnowledgeBase(false, workspace)
      setOpMsg(`已清空索引: ${result.workspace}`)
      await onWorkspaceChanged?.()
      await loadStats()
      await loadObservability()
    } catch (e: unknown) {
      setOpMsg(`清空失败: ${(e as Error).message}`)
    } finally {
      setOperating(false)
    }
  }

  const handleRebuild = async () => {
    const confirmed = await confirm({
      title: '重建当前知识库索引',
      message: `将先清空知识库“${workspace}”的现有索引，再从该知识库的上传目录重新索引全部文档。`,
      confirmLabel: '开始重建',
      tone: 'danger',
    })
    if (!confirmed) return
    setOperating(true)
    setOpMsg('')
    setRebuildTask(null)
    pollingTaskIdRef.current = null
    try {
      const task = await rebuildIndex({
        workspace,
        separators: ['\n\n', '\n', '。', '！', '？', '；', '  '],
        chunk_size: 1024,
        chunk_overlap: 100,
        index_mode: 'complete',
      })
      setRebuildTask(task)
      setOpMsg(`重建任务已创建: ${task.task_id}`)
      await pollTask(task.task_id)
    } catch (e: unknown) {
      setOpMsg(`重建失败: ${(e as Error).message}`)
    } finally {
      setOperating(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-gray-400">
        <div className="animate-spin w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full mr-2" />
        <span className="text-sm">加载系统状态...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-sm text-red-700">
        加载失败: {error}
        <button onClick={loadStats} className="ml-3 underline">重试</button>
      </div>
    )
  }

  if (!stats) return null

  const cards = [
    { label: '文档总数', value: stats.doc_count, icon: FileText, tone: 'bg-blue-50 text-blue-700' },
    { label: 'Chunk 总数', value: stats.chunk_count, icon: Layers3, tone: 'bg-emerald-50 text-emerald-700' },
    { label: '图谱节点', value: stats.graph_nodes, icon: Network, tone: 'bg-violet-50 text-violet-700' },
    { label: '图谱关系', value: stats.graph_edges, icon: GitBranch, tone: 'bg-amber-50 text-amber-700' },
  ]

  return (
    <div className="space-y-5 sm:space-y-8">
      <div>
        <h2 className="text-xl font-bold text-gray-800">系统状态</h2>
        <p className="text-sm text-gray-500 mt-1">知识库运行状态与统计概览。</p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((card) => (
          <div key={card.label} className="rounded-lg border border-gray-200 bg-white p-5">
            <div className="flex items-center justify-between">
              <span className={`grid h-9 w-9 place-items-center rounded-md ${card.tone}`}>
                <card.icon size={18} strokeWidth={1.8} />
              </span>
              <span className="text-3xl font-semibold tabular-nums text-gray-900">{card.value.toLocaleString()}</span>
            </div>
            <p className="mt-3 text-sm font-medium text-gray-500">{card.label}</p>
          </div>
        ))}
      </div>

      {/* Config info */}
      <div className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
          配置信息
        </h3>
        <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <dt className="text-xs text-gray-400 mb-1">嵌入模型</dt>
            <dd className="text-sm font-mono text-gray-800">{stats.embed_model}</dd>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <dt className="text-xs text-gray-400 mb-1">LightRAG Workspace</dt>
            <dd className="text-sm font-mono text-gray-800">{stats.workspace}</dd>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <dt className="text-xs text-gray-400 mb-1">LightRAG 目录大小</dt>
            <dd className="text-sm font-mono text-gray-800">{stats.lightrag_dir_size}</dd>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <dt className="text-xs text-gray-400 mb-1">嵌入维度</dt>
            <dd className="text-sm font-mono text-gray-800">{stats.embed_dim || 1024}</dd>
          </div>
        </dl>
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
        <div className="flex flex-col items-stretch gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">
              知识库运维
            </h3>
            <p className="text-sm text-gray-500 mt-1">
              清空或重建 LightRAG 索引。默认保留上传目录里的原始文档。
            </p>
          </div>
          <div className="flex shrink-0 gap-2">
            <button
              onClick={handleRebuild}
              disabled={operating}
              className="px-3 py-2 text-sm rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50"
            >
              {operating && rebuildTask ? '重建中...' : '重建索引'}
            </button>
            <button
              onClick={handleClear}
              disabled={operating}
              className="px-3 py-2 text-sm rounded-lg border border-red-200 text-red-600 hover:bg-red-50 disabled:opacity-50"
            >
              清空索引
            </button>
          </div>
        </div>

        {rebuildTask && (
          <div className={`mt-4 rounded-lg border p-3 ${
            taskTone(rebuildTask) === 'red'
              ? 'border-red-200 bg-red-50'
              : taskTone(rebuildTask) === 'green'
                ? 'border-green-200 bg-green-50'
                : taskTone(rebuildTask) === 'gray'
                  ? 'border-gray-200 bg-gray-50'
                  : 'border-primary-200 bg-primary-50'
          }`}>
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-gray-800">重建任务 {rebuildTask.task_id}</span>
              <span className="text-xs text-gray-500">{rebuildTask.status}</span>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {rebuildTask.message}，文档 {rebuildTask.current}/{rebuildTask.total}
            </div>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-gray-400">
              {rebuildTask.current_doc && <span>当前文档：{rebuildTask.current_doc}</span>}
              {rebuildTask.timeout_seconds && <span>单文档超时：{rebuildTask.timeout_seconds}s</span>}
              {rebuildTask.updated_at && <span>最后更新：{formatTaskTime(rebuildTask.updated_at)}</span>}
              {isTaskTerminal(rebuildTask) && rebuildTask.updated_at && <span>完成时间：{formatTaskTime(rebuildTask.updated_at)}</span>}
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {(['parse', 'chunk_vector', 'kg', 'merge'] as const).map((key) => {
                const label =
                  key === 'parse'
                    ? '解析'
                    : key === 'chunk_vector'
                      ? 'Chunk向量'
                      : key === 'kg'
                        ? 'KG抽取'
                        : '图谱/落盘'
                const v = rebuildTask.stage_timings?.[key] ?? 0
                const active = !isTaskTerminal(rebuildTask) && rebuildTask.current_stage === key
                const displaySeconds = active
                  ? activeStageSeconds(rebuildTask.current_stage_started_at, typeof v === 'number' ? v : 0)
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
            {rebuildTask.errors.length > 0 && (
              <div className="mt-2 space-y-1">
                {rebuildTask.errors.slice(0, 3).map((err) => (
                  <div key={`${err.doc_name}-${err.error}`} className="text-xs text-red-700">
                    {err.doc_name || '任务'}: {err.error}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {opMsg && (
          <div className={`mt-4 p-3 rounded-lg text-sm ${
            opMsg.includes('失败') ? 'bg-red-50 text-red-700 border border-red-200' : 'bg-green-50 text-green-700 border border-green-200'
          }`}>
            {opMsg}
          </div>
        )}
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
        <div className="flex items-center justify-between gap-4">
          <button
            type="button"
            onClick={() => setRecentTasksOpen((open) => !open)}
            className="group flex min-w-0 flex-1 items-center gap-3 text-left"
            aria-expanded={recentTasksOpen}
          >
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-gray-50 text-gray-500 ring-1 ring-gray-200">
              <ListChecks size={17} />
            </span>
            <span className="min-w-0">
              <span className="flex items-center gap-2 text-sm font-semibold text-gray-800">
                最近索引任务
                <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[11px] font-medium text-gray-500">
                  {recentTasks.length}
                </span>
              </span>
              <span className="mt-0.5 block truncate text-xs text-gray-500">
                {recentTasks.some((task) => !isTaskTerminal(task))
                  ? '有索引任务正在运行，状态将自动刷新'
                  : '当前知识库最近的索引记录'}
              </span>
            </span>
            <ChevronDown
              size={16}
              className={`ml-auto shrink-0 text-gray-400 transition-transform ${
                recentTasksOpen ? 'rotate-180' : ''
              }`}
            />
          </button>
          <button
            onClick={loadObservability}
            className="ui-button-secondary inline-flex shrink-0 items-center gap-2"
            title="刷新最近任务"
          >
            <RefreshCw size={14} />
            刷新
          </button>
        </div>

        {recentTasksOpen && (
          <div className="mt-5 border-t border-gray-100 pt-2">
            {recentTasks.length === 0 ? (
              <div className="py-8 text-center text-sm text-gray-400">
                当前知识库暂无索引任务记录
              </div>
            ) : (
              <div className="divide-y divide-gray-100">
                {recentTasks.map((task) => (
                  <details key={task.task_id} className="group/task">
                    <summary className="flex cursor-pointer list-none items-center gap-3 py-3 text-sm [&::-webkit-details-marker]:hidden">
                      <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${taskDotClass(task)}`} />
                      <span className="min-w-0 flex-1">
                        <span className="flex min-w-0 items-center gap-2">
                          <span className="font-medium text-gray-900">{formatTaskKind(task.kind)}</span>
                          <span className="truncate font-mono text-[11px] text-gray-400">{task.task_id}</span>
                        </span>
                        <span className="mt-0.5 block truncate text-xs text-gray-500">
                          {task.current_doc || task.message}
                        </span>
                      </span>
                      <span className="shrink-0 text-xs text-gray-500">
                        {formatTaskStatus(task.status)}
                      </span>
                      <span className="w-14 shrink-0 text-right text-xs text-gray-400">
                        {formatTaskTime(task.updated_at)}
                      </span>
                      <ChevronDown
                        size={15}
                        className="shrink-0 text-gray-400 transition-transform group-open/task:rotate-180"
                      />
                    </summary>
                    <div className="mb-3 ml-5 border-l border-gray-200 pl-4 text-xs text-gray-500">
                      <div className="flex flex-wrap gap-x-4 gap-y-1">
                        <span>文档 {task.current}/{task.total}</span>
                        {task.request?.index_mode && (
                          <span>{task.request.index_mode === 'fast' ? '快速索引' : '完整索引'}</span>
                        )}
                        {task.current_stage && <span>阶段：{formatIndexStage(task.current_stage)}</span>}
                        <span>更新于 {formatTaskTime(task.updated_at)}</span>
                      </div>
                      <p className="mt-1.5 break-words text-gray-600">{task.message}</p>
                      {task.results.some((result) => result.kg_status === 'partial') && (
                        <div className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-amber-800">
                          {task.results
                            .filter((result) => result.kg_status === 'partial')
                            .map((result) => (
                              <div key={`${task.task_id}-${result.doc_name}-kg-partial`}>
                                {result.doc_name}: 已保留文本和向量，跳过{' '}
                                {result.kg_timed_out_chunks?.length || 0} 个超时 KG 块
                              </div>
                            ))}
                        </div>
                      )}
                      {task.errors.length > 0 && (
                        <div className="mt-2 space-y-1 rounded-md bg-red-50 px-3 py-2 text-red-700">
                          {task.errors.slice(0, 4).map((err) => (
                            <div key={`${task.task_id}-${err.doc_name}-${err.error}`}>
                              {err.doc_name || '任务'}: {err.error}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </details>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-4 sm:p-6">
        <div className="mb-4 flex flex-col items-stretch gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-gray-600">
              <ScrollText size={16} />
              运行日志
            </h3>
            <p className="mt-1 text-sm text-gray-500">查看索引、模型测试、答案质量检查和异常日志。</p>
          </div>
          <button
            onClick={loadObservability}
            disabled={logsLoading}
            className="ui-button-secondary inline-flex items-center gap-2 disabled:opacity-50"
          >
            <RefreshCw size={14} className={logsLoading ? 'animate-spin' : ''} />
            刷新日志
          </button>
        </div>
        <div className="grid gap-3 border-b border-gray-100 pb-4 md:grid-cols-[160px_minmax(0,1fr)]">
          <label className="block">
            <span className="ui-label">级别</span>
            <select value={logLevel} onChange={(event) => setLogLevel(event.target.value)} className="ui-control w-full">
              <option value="">全部</option>
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
              <option value="CRITICAL">CRITICAL</option>
            </select>
          </label>
          <label className="block">
            <span className="ui-label">关键词</span>
            <input
              value={logContains}
              onChange={(event) => setLogContains(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') void loadObservability()
              }}
              className="ui-control w-full"
              placeholder="例如 answer_quality_check、model、索引失败"
            />
          </label>
        </div>
        {logMsg && (
          <div className={`mt-3 text-xs ${logMsg.includes('失败') ? 'text-red-600' : 'text-gray-400'}`}>
            {logMsg}
          </div>
        )}
        <div
          ref={logViewportRef}
          className="mt-3 max-h-[420px] overflow-auto rounded-lg bg-gray-950 p-3 font-mono text-xs leading-relaxed text-gray-200"
        >
          {logs.length === 0 ? (
            <div className="py-8 text-center text-gray-500">暂无可显示日志</div>
          ) : (
            logs.map((item) => (
              <div key={`${item.line_no}-${item.text}`} className="flex gap-3 border-b border-white/5 py-1 last:border-0">
                <span className="w-12 shrink-0 text-right text-gray-600">{item.line_no}</span>
                <span className={`w-16 shrink-0 ${
                  item.level === 'ERROR' || item.level === 'CRITICAL'
                    ? 'text-red-300'
                    : item.level === 'WARNING'
                      ? 'text-amber-300'
                      : 'text-emerald-300'
                }`}>
                  {item.level || 'LOG'}
                </span>
                <span className="min-w-0 whitespace-pre-wrap break-words">{item.text}</span>
              </div>
            ))
          )}
        </div>
      </div>

      <button
        onClick={() => {
          void loadStats()
          void loadObservability()
        }}
        className="text-xs text-primary-500 hover:text-primary-700 transition-colors"
      >
        刷新数据
      </button>
    </div>
  )
}
