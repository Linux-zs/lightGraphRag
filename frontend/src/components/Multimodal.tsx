import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api'

export async function mmRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await apiFetch(path, options)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(typeof body.detail === 'string' ? body.detail : `请求失败 (${response.status})`)
  }
  return response.json()
}

export async function uploadQueryImage(file: File, workspace: string) {
  const form = new FormData()
  form.append('file', file)
  return mmRequest<{ image_id: string }>(`/images/attachments?workspace=${encodeURIComponent(workspace)}`, { method: 'POST', body: form })
}

export function StoredImage({ path, alt, imageClassName = 'max-h-56 max-w-full rounded-lg object-contain' }: { path: string; alt: string; imageClassName?: string }) {
  const [url, setUrl] = useState('')
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    let objectUrl = ''
    setUrl(''); setError('')
    apiFetch(path, { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error(response.status === 404 ? '图片来源已删除或不可用' : '图片加载失败')
      const blob = await response.blob()
      if (controller.signal.aborted) return
      objectUrl = URL.createObjectURL(blob)
      setUrl(objectUrl)
    }).catch(e => { if (!controller.signal.aborted) setError((e as Error).message) })
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [path])
  if (error) return <p role="status" className="text-xs text-amber-700">{error}</p>
  return url ? <div className="space-y-2"><a href={url} target="_blank" rel="noreferrer" aria-label={`放大预览：${alt}`}><img src={url} alt={alt} className={imageClassName} /></a>
    {path.startsWith('/kb/images/') && <button className="text-xs text-gray-500 underline" onClick={async () => {
      try {
        const response = await apiFetch(path + (path.includes('?') ? '&' : '?') + 'kind=original')
        if (!response.ok) throw new Error('原图下载失败，来源可能已删除')
        const blob = await response.blob(); const original = URL.createObjectURL(blob)
        const link = document.createElement('a'); link.href = original
        const ext = blob.type === 'image/png' ? 'png' : blob.type === 'image/webp' ? 'webp' : 'jpg'
        link.download = `image.${ext}`; link.click(); setTimeout(() => URL.revokeObjectURL(original), 1000)
      } catch (e) { setError((e as Error).message) }
    }}>下载原图</button>}
  </div> : <p className="text-xs text-gray-400">图片加载中…</p>
}

export function ImageInput({ workspace, value, onChange, disabled = false }: {
  workspace: string; value: string | null; onChange: (id: string | null) => void; disabled?: boolean
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const generation = useRef(0)
  useEffect(() => { generation.current++; setBusy(false); setError(''); return () => { generation.current++ } }, [workspace])
  return <div className="space-y-2 text-xs">
    <label className="inline-flex cursor-pointer items-center rounded-lg border border-gray-300 bg-white px-3 py-2 focus-within:ring-2 focus-within:ring-blue-500">
      {busy ? '正在上传图片…' : '添加图片'}
      <input aria-label="添加查询图片" type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" disabled={disabled || busy}
        onChange={async e => {
          const file = e.target.files?.[0]; e.target.value = ''; if (!file) return
          const current = ++generation.current
          setBusy(true); setError('')
          try { const result = await uploadQueryImage(file, workspace); if (generation.current === current) onChange(result.image_id) }
          catch (err) { if (generation.current === current) setError((err as Error).message) }
          finally { if (generation.current === current) setBusy(false) }
        }} />
    </label>
    {value && <div className="space-y-2"><StoredImage path={`/images/attachments/${value}?workspace=${encodeURIComponent(workspace)}`} alt="查询图片" /><button className="text-gray-600 underline" disabled={disabled} onClick={() => onChange(null)}>移除图片</button></div>}
    {error && <p role="alert" className="text-red-600">{error}</p>}
  </div>
}

type Source = { doc_id: string; doc_name: string; location: Record<string, unknown>; context: string }
type Hit = { image_id: string; score: number; description: string; ocr: string; sources: Source[] }

export function ImageRecall({ workspace }: { workspace: string }) {
  const [query, setQuery] = useState('')
  const [image, setImage] = useState<string | null>(null)
  const [hits, setHits] = useState<Hit[] | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  return <section className="space-y-4 rounded-lg border border-gray-200 bg-white p-5">
    <h3 className="font-semibold text-gray-800">图片检索</h3>
    <p className="text-sm text-gray-500">用文字、图片或两者组合检索。结果按融合排名展示，分数不代表正确率。</p>
    <ImageInput workspace={workspace} value={image} onChange={setImage} disabled={busy} />
    <div className="flex gap-3"><input aria-label="图片检索文字" value={query} onChange={e => setQuery(e.target.value)} placeholder="描述要找的图片…" className="min-w-0 flex-1 rounded-lg border px-3 py-2 text-sm" />
      <button disabled={busy || (!query.trim() && !image)} className="rounded-lg bg-gray-900 px-4 py-2 text-sm text-white disabled:opacity-40" onClick={async () => {
        controller.current?.abort(); const abort = new AbortController(); controller.current = abort
        setBusy(true); setError(''); setHits(null)
        try { const data = await mmRequest<{ hits: Hit[] }>('/recall/images', { method: 'POST', signal: abort.signal, body: JSON.stringify({ workspace, query, image_id: image }) }); if (!abort.signal.aborted) setHits(data.hits) }
        catch (e) { if (!abort.signal.aborted) setError((e as Error).message) }
        finally { if (!abort.signal.aborted) setBusy(false) }
      }}>{busy ? '检索中…' : '检索图片'}</button></div>
    {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
    {hits?.length === 0 && <p className="text-sm text-gray-500">未找到图片，请先在知识库管理中建立图片索引。</p>}
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">{hits?.map((hit, index) => <article key={hit.image_id} className="space-y-3 rounded-lg border p-3">
      <p className="text-xs text-gray-500">排名 {index+1} · 融合分数 {hit.score.toFixed(4)}</p>
      <StoredImage path={`/kb/images/${hit.image_id}?workspace=${encodeURIComponent(workspace)}`} alt={hit.description || '检索图片'} />
      <p className="text-sm text-gray-700">{hit.description}</p>
      <details className="text-xs text-gray-600"><summary className="cursor-pointer">来源与关联正文（{hit.sources.length}）</summary>{hit.sources.map((source, i) => <div key={i} className="mt-2 space-y-1"><p className="font-medium">{source.doc_name} · {source.location.page ? `第 ${source.location.page} 页` : source.location.paragraph ? `段落 ${source.location.paragraph}` : '独立图片'}</p><p className="whitespace-pre-wrap">{source.context || '无关联正文'}</p></div>)}<p className="mt-2 whitespace-pre-wrap">识别文字：{hit.ocr || '无'}</p></details>
    </article>)}</div>
  </section>
}

export function ImageIndexPanel({ workspace, names }: { workspace: string; names: string[] }) {
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const [taskId, setTaskId] = useState('')
  const [rows, setRows] = useState<{ id: string; name: string; status: string; error: string }[]>([])
  useEffect(() => {
    const abort = new AbortController()
    let refreshing = false
    const refresh = async () => {
      if (refreshing) return
      refreshing = true
      try {
        const documents = await mmRequest<typeof rows>(`/kb/image-documents?workspace=${encodeURIComponent(workspace)}`, { signal: abort.signal })
        if (!abort.signal.aborted) setRows(documents)
        if (!taskId) {
          const tasks = await mmRequest<{ task_id: string; workspace: string; kind: string; status: string; message: string }[]>('/kb/index-tasks', { signal: abort.signal })
          if (abort.signal.aborted) return
          const running = tasks.find(task => task.workspace === workspace && task.kind === 'images' && ['queued', 'running'].includes(task.status))
          if (running) { setTaskId(running.task_id); setBusy(true); setMessage(running.message) }
        } else {
          const task = await mmRequest<{ status: string; message: string }>(`/kb/index-tasks/${taskId}`, { signal: abort.signal })
          if (abort.signal.aborted) return
          setMessage(task.message)
          setBusy(['queued', 'running'].includes(task.status))
        }
      } catch (e) { if (!abort.signal.aborted) setMessage((e as Error).message) }
      finally { refreshing = false }
    }
    void refresh(); const timer = setInterval(refresh, 2500)
    return () => { abort.abort(); clearInterval(timer) }
  }, [workspace, taskId])
  return <section className="space-y-3 rounded-lg border border-gray-200 bg-white p-5">
    <h3 className="font-semibold">图片索引</h3>
    <p className="text-sm text-gray-500">为已上传的图片、PDF、DOCX 建立或重建图片索引；原有文字索引独立保留。未选择文档时处理全部支持的文档。</p>
    <div className="flex gap-3"><button disabled={busy || !names.length} className="rounded-lg bg-gray-900 px-4 py-2 text-sm text-white disabled:opacity-40" onClick={async () => {
      setBusy(true); setMessage('正在创建图片索引任务…')
      try { const task = await mmRequest<{ task_id: string }>('/kb/index-images', { method: 'POST', body: JSON.stringify({ workspace, doc_names: names }) }); setTaskId(task.task_id) }
      catch (e) { setBusy(false); setMessage((e as Error).message) }
    }}>建立 / 重建图片索引（{names.length} 份）</button>
    {busy && taskId && <button className="text-sm underline" onClick={async () => { try { await mmRequest(`/kb/index-tasks/${taskId}/cancel`, { method: 'POST' }); setMessage('已请求取消') } catch (e) { setMessage((e as Error).message) } }}>取消任务</button>}</div>
    {message && <p role="status" className="text-sm text-gray-600">{message}</p>}
    {rows.map(row => <div key={row.id} className="border-t pt-2 text-xs"><span className="font-medium">{row.name}</span> · {({ indexed: '已索引', processing: '处理中', partial: '部分完成，请查看详情', failed: '失败', cancelled: '已取消', stale: '来源已更新，需重建' } as Record<string, string>)[row.status] || row.status}{row.error && <p className="mt-1 text-amber-700">{row.error}</p>}</div>)}
  </section>
}
