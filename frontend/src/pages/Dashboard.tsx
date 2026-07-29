import { useCallback, useEffect, useRef, useState } from 'react'
import {
  clearKnowledgeBase,
  getIndexTask,
  getSystemStats,
  listIndexTasks,
  rebuildIndex,
  IndexTask,
  SystemStats,
} from '../api'

interface Props {
  workspace: string
  onWorkspaceChanged?: () => void
}

export default function Dashboard({ workspace, onWorkspaceChanged }: Props) {
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [opMsg, setOpMsg] = useState('')
  const [operating, setOperating] = useState(false)
  const [rebuildTask, setRebuildTask] = useState<IndexTask | null>(null)
  const pollingTaskIdRef = useRef<string | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    loadStats()
    restoreRebuildTask()
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
  }, [loadStats])

  const restoreRebuildTask = useCallback(async () => {
    try {
      const tasks = await listIndexTasks()
      if (!mountedRef.current) return
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
    const confirmed = window.confirm('确认清空 LightRAG 索引？上传的原始文档会保留。')
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
    } catch (e: unknown) {
      setOpMsg(`清空失败: ${(e as Error).message}`)
    } finally {
      setOperating(false)
    }
  }

  const handleRebuild = async () => {
    const confirmed = window.confirm('确认清空当前 LightRAG 索引，并从上传目录重新索引全部文档？')
    if (!confirmed) return
    setOperating(true)
    setOpMsg('')
    setRebuildTask(null)
    pollingTaskIdRef.current = null
    try {
      const task = await rebuildIndex({
        workspace,
        separators: ['\n\n', '\n', '。', '！', '？', '；', '  '],
        chunk_size: 512,
        chunk_overlap: 50,
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
      <div className="p-6 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
        加载失败: {error}
        <button onClick={loadStats} className="ml-3 underline">重试</button>
      </div>
    )
  }

  if (!stats) return null

  const cards = [
    { label: '文档总数', value: stats.doc_count, shape: 'rounded-[3px]', color: 'bg-blue-50 border-blue-200 text-blue-800' },
    { label: 'Chunk 总数', value: stats.chunk_count, shape: 'rounded-full', color: 'bg-green-50 border-green-200 text-green-800' },
    { label: '图谱节点', value: stats.graph_nodes, shape: 'rotate-45 rounded-[2px]', color: 'bg-purple-50 border-purple-200 text-purple-800' },
    { label: '图谱关系', value: stats.graph_edges, shape: 'rounded-none', color: 'bg-orange-50 border-orange-200 text-orange-800' },
  ]

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-bold text-gray-800">系统状态</h2>
        <p className="text-sm text-gray-500 mt-1">知识库运行状态与统计概览。当前知识库: {workspace}</p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((card) => (
          <div key={card.label} className={`rounded-xl border p-5 ${card.color}`}>
            <div className="flex items-center justify-between">
              <span className={`inline-block h-5 w-5 border-2 border-current ${card.shape}`} />
              <span className="text-3xl font-bold">{card.value.toLocaleString()}</span>
            </div>
            <p className="text-sm font-medium mt-3 opacity-70">{card.label}</p>
          </div>
        ))}
      </div>

      {/* Config info */}
      <div className="bg-white border border-gray-200 rounded-xl p-6">
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

      <div className="bg-white border border-gray-200 rounded-xl p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">
              知识库运维
            </h3>
            <p className="text-sm text-gray-500 mt-1">
              清空或重建 LightRAG 索引。默认保留上传目录里的原始文档。
            </p>
          </div>
          <div className="flex gap-2">
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
              {rebuildTask.message}，进度 {rebuildTask.current}/{rebuildTask.total}
            </div>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-gray-400">
              {rebuildTask.current_doc && <span>当前文档：{rebuildTask.current_doc}</span>}
              {rebuildTask.timeout_seconds && <span>单文档超时：{rebuildTask.timeout_seconds}s</span>}
              {rebuildTask.updated_at && <span>最后更新：{formatTaskTime(rebuildTask.updated_at)}</span>}
              {isTaskTerminal(rebuildTask) && rebuildTask.updated_at && <span>完成时间：{formatTaskTime(rebuildTask.updated_at)}</span>}
            </div>
            <div className="mt-3 h-2 rounded-full bg-white overflow-hidden border border-gray-100">
              <div
                className={`h-full transition-all ${
                  rebuildTask.status === 'failed' || rebuildTask.status === 'partial'
                    ? 'bg-red-500'
                    : isTaskTerminal(rebuildTask)
                      ? 'bg-green-500'
                      : 'bg-primary-500'
                }`}
                style={{ width: `${Math.max(2, rebuildTask.progress)}%` }}
              />
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

      <button
        onClick={loadStats}
        className="text-xs text-primary-500 hover:text-primary-700 transition-colors"
      >
        刷新数据
      </button>
    </div>
  )
}
