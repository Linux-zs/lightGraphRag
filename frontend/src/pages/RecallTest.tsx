import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowDown, ArrowRight, ArrowUp, FileSearch, LoaderCircle, Route, Search } from 'lucide-react'
import {
  recallTest,
  RecallChunk,
  RecallTestResponse,
  textRecallTest,
  TextRecallHit,
  TextRecallResponse,
} from '../api'
import { Panel, RangeField, SelectField, Toggle } from '../components/ui'

const MODES = ['mix', 'hybrid', 'local', 'global', 'naive']

function shortFileName(path?: string) {
  if (!path) return 'LightRAG'
  return path.split(/[\\/]/).pop() || path
}

function keywordList(metadata: Record<string, unknown>, key: 'high_level' | 'low_level') {
  const keywords = metadata.keywords
  if (!keywords || typeof keywords !== 'object') return []
  const values = (keywords as Record<string, unknown>)[key]
  return Array.isArray(values) ? values.map(String).filter(Boolean) : []
}

function scoreText(value?: number | null) {
  return typeof value === 'number' ? value.toFixed(4) : '—'
}

interface Props {
  workspace: string
}

export default function RecallTest({ workspace }: Props) {
  const [tab, setTab] = useState<'context' | 'text'>('context')
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState('mix')
  const [topK, setTopK] = useState(40)
  const [chunkTopK, setChunkTopK] = useState(20)
  const [enableRerank, setEnableRerank] = useState(true)
  const [loading, setLoading] = useState(false)
  const [contextResult, setContextResult] = useState<RecallTestResponse | null>(null)
  const [textResult, setTextResult] = useState<TextRecallResponse | null>(null)
  const [error, setError] = useState('')
  const [activeQuery, setActiveQuery] = useState('')
  const [lastRun, setLastRun] = useState<{
    query: string
    tab: 'context' | 'text'
    durationMs: number
    completedAt: string
    sequence: number
  } | null>(null)
  const [runSequence, setRunSequence] = useState(0)
  const requestRef = useRef(0)
  const requestAbortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    requestAbortRef.current?.abort()
    requestRef.current += 1
    setContextResult(null)
    setTextResult(null)
    setError('')
    setLoading(false)
    setActiveQuery('')
    setLastRun(null)
    return () => requestAbortRef.current?.abort()
  }, [workspace])

  const handleRun = async () => {
    if (!query.trim()) return
    const submittedQuery = query.trim()
    const submittedTab = tab
    const startedAt = performance.now()
    const sequence = runSequence + 1
    const requestId = ++requestRef.current
    requestAbortRef.current?.abort()
    const controller = new AbortController()
    requestAbortRef.current = controller
    setRunSequence(sequence)
    setActiveQuery(submittedQuery)
    setLoading(true)
    setError('')
    if (submittedTab === 'context') {
      setContextResult(null)
    } else {
      setTextResult(null)
    }
    try {
      if (submittedTab === 'context') {
        const result = await recallTest({
          workspace,
          query: submittedQuery,
          mode,
          top_k: topK,
          chunk_top_k: chunkTopK,
          enable_rerank: enableRerank,
        }, controller.signal)
        if (requestId !== requestRef.current) return
        setContextResult(result)
      } else {
        const result = await textRecallTest({
          workspace,
          query: submittedQuery,
          top_k: chunkTopK,
          enable_rerank: enableRerank,
        }, controller.signal)
        if (requestId !== requestRef.current) return
        setTextResult(result)
      }
      setLastRun({
        query: submittedQuery,
        tab: submittedTab,
        durationMs: Math.round(performance.now() - startedAt),
        completedAt: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
        sequence,
      })
    } catch (caught: unknown) {
      if (requestId !== requestRef.current) return
      if (controller.signal.aborted) return
      setError((caught as Error).message || '召回测试失败')
    } finally {
      if (requestId !== requestRef.current) return
      setLoading(false)
      setActiveQuery('')
    }
  }

  const renderChunk = (chunk: RecallChunk, index: number) => (
    <div key={chunk.chunk_id || chunk.reference_id || index} className="rounded-lg border border-gray-200 bg-white p-3.5">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="font-mono text-[10px] text-gray-400">#{index + 1}</span>
        <span className="rounded bg-primary-50 px-1.5 py-0.5 text-[10px] text-primary-700">
          {shortFileName(chunk.file_path)}
        </span>
        {(chunk.chunk_id || chunk.reference_id) && (
          <span className="max-w-[280px] truncate font-mono text-[10px] text-gray-300">
            {chunk.chunk_id || chunk.reference_id}
          </span>
        )}
      </div>
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-gray-700">
        {chunk.content || '无内容'}
      </p>
    </div>
  )

  const renderTextHit = (hit: TextRecallHit, reranked: boolean) => {
    const rank = reranked ? (hit.rerank_rank || hit.vector_rank) : hit.vector_rank
    const delta = hit.rerank_rank ? hit.vector_rank - hit.rerank_rank : 0
    return (
      <div key={`${reranked ? 'r' : 'v'}-${hit.chunk_id}`} className="rounded-lg border border-gray-200 bg-white p-3.5">
        <div className="flex items-center gap-2 text-xs">
          <span className="grid h-6 w-6 place-items-center rounded bg-gray-100 font-semibold tabular-nums text-gray-700">
            {rank}
          </span>
          <span className="min-w-0 flex-1 truncate font-medium text-gray-800">
            {shortFileName(hit.file_path)}
          </span>
          {reranked && hit.rerank_rank && (
            <span className={`inline-flex items-center gap-1 tabular-nums ${
              delta > 0 ? 'text-emerald-600' : delta < 0 ? 'text-amber-600' : 'text-gray-400'
            }`}>
              {delta > 0 ? <ArrowUp size={13} /> : delta < 0 ? <ArrowDown size={13} /> : <ArrowRight size={13} />}
              {delta === 0 ? '持平' : Math.abs(delta)}
            </span>
          )}
        </div>
        <p className="mt-2 line-clamp-4 whitespace-pre-wrap text-sm leading-relaxed text-gray-700">
          {hit.content}
        </p>
        <div className="mt-3 flex items-center gap-4 border-t border-gray-100 pt-2 font-mono text-[11px] text-gray-500">
          <span>向量 {scoreText(hit.vector_score)}</span>
          {reranked && <span>Rerank {scoreText(hit.rerank_score)}</span>}
          <span className="ml-auto truncate text-gray-400" title={hit.chunk_id}>{hit.chunk_id}</span>
        </div>
      </div>
    )
  }

  const contextKeywords = useMemo(() => {
    if (!contextResult) return { high: [], low: [] }
    return {
      high: keywordList(contextResult.metadata, 'high_level'),
      low: keywordList(contextResult.metadata, 'low_level'),
    }
  }, [contextResult])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900">召回调试</h2>
        <p className="mt-1 text-sm text-gray-500">
          检查 LightRAG 实际上下文，并对比文本向量召回与 Rerank。
        </p>
      </div>

      <div className="flex w-fit rounded-lg bg-gray-100 p-1" role="tablist">
        <button
          onClick={() => setTab('context')}
          className={`flex h-8 items-center gap-2 rounded-md px-3 text-sm ${
            tab === 'context' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-800'
          }`}
        >
          <Route size={15} />
          生成上下文
        </button>
        <button
          onClick={() => setTab('text')}
          className={`flex h-8 items-center gap-2 rounded-md px-3 text-sm ${
            tab === 'text' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-800'
          }`}
        >
          <FileSearch size={15} />
          文本召回
        </button>
      </div>

      <Panel>
        <div className="flex max-w-3xl gap-3">
          <label className="flex h-10 min-w-0 flex-1 items-center gap-2 rounded-md border border-gray-300 px-3 focus-within:border-primary-500 focus-within:ring-2 focus-within:ring-primary-100">
            <Search size={16} className="text-gray-400" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => event.key === 'Enter' && handleRun()}
              placeholder="输入测试查询"
              className="min-w-0 flex-1 bg-transparent text-sm outline-none"
            />
          </label>
          <button
            onClick={handleRun}
            disabled={loading || !query.trim()}
            className="ui-button-primary min-w-24"
          >
            {loading ? '检索中' : '开始测试'}
          </button>
        </div>

        <div className={`mt-5 grid gap-5 border-t border-gray-100 pt-5 ${
          tab === 'context' ? 'grid-cols-1 lg:grid-cols-4' : 'grid-cols-1 lg:grid-cols-2'
        }`}>
          {tab === 'context' && (
            <>
              <SelectField label="检索模式" value={mode} onChange={(event) => setMode(event.target.value)}>
                {MODES.map((item) => <option key={item}>{item}</option>)}
              </SelectField>
              <RangeField label="图谱 Top-K" value={topK} min={1} max={100} onChange={setTopK} />
            </>
          )}
          <RangeField label="文本块 Top-K" value={chunkTopK} min={1} max={100} onChange={setChunkTopK} />
          <div className="flex items-end pb-2">
            <Toggle checked={enableRerank} onChange={setEnableRerank} label="启用 Rerank" />
          </div>
        </div>
      </Panel>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {loading && (
        <div className="flex min-h-44 items-center justify-center rounded-lg border border-primary-100 bg-primary-50/40">
          <div className="text-center">
            <LoaderCircle size={24} className="mx-auto animate-spin text-primary-600" />
            <p className="mt-3 text-sm font-medium text-gray-800">正在重新召回</p>
            <p className="mt-1 max-w-xl truncate px-6 text-xs text-gray-500">“{activeQuery}”</p>
          </div>
        </div>
      )}

      {!loading && lastRun && lastRun.tab === tab && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-500">
          <span className="font-medium text-gray-700">第 {lastRun.sequence} 次测试</span>
          <span>查询：{lastRun.query}</span>
          <span>完成：{lastRun.completedAt}</span>
          <span>耗时：{lastRun.durationMs} ms</span>
        </div>
      )}

      {!loading && tab === 'context' && contextResult && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[
              ['文本块', contextResult.chunks.length],
              ['实体', contextResult.entities.length],
              ['关系', contextResult.relationships.length],
              ['引用', contextResult.references.length],
            ].map(([label, value]) => (
              <div key={String(label)} className="rounded-lg border border-gray-200 bg-white px-4 py-3">
                <p className="text-xs text-gray-500">{label}</p>
                <p className="mt-1 text-xl font-semibold tabular-nums text-gray-900">{value}</p>
              </div>
            ))}
          </div>

          {(contextKeywords.high.length > 0 || contextKeywords.low.length > 0) && (
            <Panel>
              <h3 className="text-sm font-semibold text-gray-800">LightRAG 查询关键词</h3>
              <div className="mt-3 space-y-3 text-xs">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="w-16 text-gray-500">高层语义</span>
                  {contextKeywords.high.map((item) => (
                    <span key={item} className="rounded bg-violet-50 px-2 py-1 text-violet-700">{item}</span>
                  ))}
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="w-16 text-gray-500">具体实体</span>
                  {contextKeywords.low.map((item) => (
                    <span key={item} className="rounded bg-blue-50 px-2 py-1 text-blue-700">{item}</span>
                  ))}
                </div>
              </div>
            </Panel>
          )}

          <Panel>
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-800">发送给回答模型的上下文</h3>
              <span className="font-mono text-xs text-gray-400">mode {contextResult.mode}</span>
            </div>
            <pre className="mt-3 max-h-[420px] overflow-auto rounded-lg border border-gray-200 bg-gray-50 p-4 font-sans text-sm leading-relaxed text-gray-700 whitespace-pre-wrap">
              {contextResult.context || '未返回上下文'}
            </pre>
          </Panel>

          {contextResult.chunks.length > 0 && (
            <div>
              <h3 className="mb-3 text-sm font-semibold text-gray-800">命中文本块</h3>
              <div className="space-y-2">{contextResult.chunks.map(renderChunk)}</div>
            </div>
          )}
        </div>
      )}

      {!loading && tab === 'text' && textResult && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500">
            <span>向量阈值 {textResult.cosine_threshold.toFixed(2)}</span>
            <span>命中 {textResult.vector_hits.length}</span>
            <span className={textResult.rerank_applied ? 'text-emerald-600' : 'text-amber-600'}>
              {textResult.rerank_applied ? '已完成 Rerank' : (textResult.rerank_warning || '未启用 Rerank')}
            </span>
          </div>
          <div className="grid gap-5 xl:grid-cols-2">
            <div>
              <h3 className="mb-3 text-sm font-semibold text-gray-800">向量原始排序</h3>
              <div className="space-y-2">
                {textResult.vector_hits.map((hit) => renderTextHit(hit, false))}
                {textResult.vector_hits.length === 0 && (
                  <div className="rounded-lg border border-dashed border-gray-300 p-8 text-center text-sm text-gray-400">
                    没有超过向量阈值的文本块
                  </div>
                )}
              </div>
            </div>
            <div>
              <h3 className="mb-3 text-sm font-semibold text-gray-800">Rerank 排序</h3>
              <div className="space-y-2">
                {textResult.rerank_hits.map((hit) => renderTextHit(hit, textResult.rerank_applied))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
