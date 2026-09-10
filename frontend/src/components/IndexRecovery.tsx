import { useEffect, useRef, useState } from 'react'
import { listIndexTasks, recoverIndexTask, type IndexTask } from '../api'

export default function IndexRecovery({ workspace, onRecovered }: {
  workspace: string
  onRecovered?: (task: IndexTask) => Promise<void>
}) {
  const [tasks, setTasks] = useState<IndexTask[]>([])
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const controller = useRef<AbortController | null>(null)
  useEffect(() => {
    const current = new AbortController()
    controller.current = current
    setTasks([]); setMessage(''); setBusy(false)
    listIndexTasks(current.signal).then(items => {
      if (!current.signal.aborted) setTasks(items.filter(task => task.workspace === workspace && task.phase === 'recovery_required'))
    }).catch(error => {
      if (!current.signal.aborted) setMessage(`恢复状态检查失败：${error.message}`)
    })
    return () => current.abort()
  }, [workspace])
  const recover = async (task: IndexTask) => {
    const current = controller.current
    if (!current || current.signal.aborted || busy) return
    setBusy(true); setMessage('正在校验并恢复索引，请勿关闭后端…')
    try {
      const result = await recoverIndexTask(task.task_id, workspace, current.signal)
      if (current.signal.aborted) return
      setTasks(items => items.filter(item => item.task_id !== task.task_id))
      setMessage(result.message)
      try {
        await onRecovered?.(result)
      } catch (error) {
        if (!current.signal.aborted) setMessage(`${result.message}；页面数据刷新失败，请刷新页面：${(error as Error).message}`)
      }
    } catch (error) {
      if (!current.signal.aborted) setMessage(`恢复失败，索引仍隔离：${(error as Error).message}`)
    } finally {
      if (!current.signal.aborted) setBusy(false)
    }
  }
  if (!tasks.length && !message) return null
  return <section aria-label="索引恢复" className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950">
    {!!tasks.length && <><h3 className="font-semibold">索引需要恢复</h3><p className="mt-1">发布曾中断，当前索引已隔离。恢复会校验已发布版本，或恢复原索引；不会重新抽取，也不会删除候选文件。</p></>}
    {tasks.map(task => <div key={task.task_id} className="mt-3 flex flex-wrap items-center gap-3">
      <span>任务 {task.task_id}</span><button disabled={busy} onClick={() => void recover(task)} className="rounded border border-amber-800 bg-white px-3 py-2 font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 disabled:opacity-50">{busy ? '恢复中…' : '校验并尝试恢复'}</button>
    </div>)}
    {message && <p role="status" className="mt-2">{message}</p>}
  </section>
}
