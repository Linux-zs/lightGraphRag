import { useState } from 'react'
import { recallTest, RecallTestResponse, RecallChunk } from '../api'

const MODES = ['mix', 'hybrid', 'local', 'global', 'naive']

function shortFileName(path?: string) {
  if (!path) return 'LightRAG'
  return path.split(/[\\/]/).pop() || path
}

interface Props {
  workspace: string
}

export default function RecallTest({ workspace }: Props) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState('mix')
  const [topK, setTopK] = useState(40)
  const [chunkTopK, setChunkTopK] = useState(20)
  const [enableRerank, setEnableRerank] = useState(true)

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<RecallTestResponse | null>(null)
  const [error, setError] = useState('')

  const handlePreview = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError('')
    try {
      const data = await recallTest({
        workspace,
        query: query.trim(),
        mode,
        top_k: topK,
        chunk_top_k: chunkTopK,
        enable_rerank: enableRerank,
      })
      setResult(data)
    } catch (e: unknown) {
      setError((e as Error).message || '上下文预览失败')
    } finally {
      setLoading(false)
    }
  }

  const renderChunk = (chunk: RecallChunk, i: number) => (
    <div key={chunk.chunk_id || chunk.reference_id || i} className="border border-gray-200 rounded-lg p-3.5 bg-white">
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <span className="text-[10px] text-gray-400 font-mono">#{i + 1}</span>
        <span className="text-[10px] bg-primary-50 text-primary-700 px-1.5 py-0.5 rounded">
          {shortFileName(chunk.file_path)}
        </span>
        {(chunk.chunk_id || chunk.reference_id) && (
          <span className="text-[10px] text-gray-300 font-mono truncate max-w-[280px]">
            {chunk.chunk_id || chunk.reference_id}
          </span>
        )}
      </div>
      <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
        {chunk.content || '无内容'}
      </p>
    </div>
  )

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-bold text-gray-800">上下文预览</h2>
        <p className="text-sm text-gray-500 mt-1">
          使用 LightRAG 的 only_need_context 模式查看本次问题会送入模型的上下文。当前知识库: {workspace}
        </p>
      </div>

      <section className="bg-white border border-gray-200 rounded-xl p-6">
        <div className="flex gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handlePreview()}
            placeholder="输入测试查询..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary-200 focus:border-primary-400 outline-none"
          />
          <button
            onClick={handlePreview}
            disabled={loading || !query.trim()}
            className="px-6 py-2.5 text-sm font-medium rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? '生成中...' : '预览'}
          </button>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 mt-5 pt-5 border-t border-gray-100">
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">Mode</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white"
            >
              {MODES.map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">
              Top-K: <span className="text-primary-600">{topK}</span>
            </label>
            <input
              type="range"
              min={1}
              max={100}
              step={1}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full accent-primary-500"
            />
          </div>

          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">
              Chunk Top-K: <span className="text-primary-600">{chunkTopK}</span>
            </label>
            <input
              type="range"
              min={1}
              max={100}
              step={1}
              value={chunkTopK}
              onChange={(e) => setChunkTopK(parseInt(e.target.value))}
              className="w-full accent-primary-500"
            />
          </div>

          <label className="flex items-center gap-2.5 cursor-pointer pt-7">
            <input
              type="checkbox"
              checked={enableRerank}
              onChange={(e) => setEnableRerank(e.target.checked)}
              className="w-4 h-4 rounded accent-primary-500 cursor-pointer"
            />
            <span className="text-sm text-gray-700 font-medium">启用 Rerank</span>
          </label>
        </div>
      </section>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {error}
        </div>
      )}

      {result && (
        <section className="bg-white border border-gray-200 rounded-xl p-6 space-y-5">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">
              LightRAG Context
            </h3>
            <div className="flex items-center gap-3 text-xs text-gray-400">
              <span>mode: <span className="text-primary-600 font-mono">{result.mode}</span></span>
              <span>chunks: <span className="text-primary-600 font-mono">{result.chunks.length}</span></span>
            </div>
          </div>

          <pre className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed bg-gray-50 border border-gray-200 rounded-lg p-4 max-h-[420px] overflow-auto font-sans">
            {result.context || '未返回上下文'}
          </pre>

          {result.chunks.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-gray-600 mb-3">Chunks</h4>
              <div className="space-y-2 max-h-[520px] overflow-y-auto pr-1">
                {result.chunks.map(renderChunk)}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="font-semibold text-gray-600 mb-1">Entities</p>
              <p className="text-gray-500">{result.entities.length} items</p>
            </div>
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="font-semibold text-gray-600 mb-1">Relationships</p>
              <p className="text-gray-500">{result.relationships.length} items</p>
            </div>
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <p className="font-semibold text-gray-600 mb-1">References</p>
              <p className="text-gray-500">{result.references.length} items</p>
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
