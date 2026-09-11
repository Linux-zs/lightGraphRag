import type { GraphExtractionPreview } from '../api'

interface Props {
  preview: GraphExtractionPreview | null
  loading: boolean
  error: string
}

export default function GraphExtractionPreviewPanel({ preview, loading, error }: Props) {
  if (loading) {
    return (
      <div className="rounded-lg border border-sky-200 bg-sky-50/70 p-4 text-sm text-sky-800">
        正在调用 KG 模型抽取采样文本块，这通常需要几十秒…
      </div>
    )
  }
  if (error) {
    return <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</div>
  }
  if (!preview) {
    return (
      <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50/60 px-4 py-8 text-center text-sm text-gray-500">
        上传文档后点击“预览实体关系”，可在正式索引前检查抽取噪声；预览不会写入图谱。
      </div>
    )
  }

  const rejections = preview.filter_stats?.policy_rejections || {}
  const rejectedCount = Object.values(rejections).reduce((sum, value) => sum + Number(value || 0), 0)
  const sampledIndexes = preview.sampled_chunks.map((chunk) => chunk.index + 1).join('、')

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded-full bg-slate-900 px-3 py-1 text-white">实体 {preview.entity_count}</span>
        <span className="rounded-full bg-sky-100 px-3 py-1 text-sky-800">关系 {preview.relation_count}</span>
        <span className="rounded-full bg-amber-100 px-3 py-1 text-amber-800">规则过滤 {rejectedCount}</span>
        {!!preview.filter_stats?.sanitized && (
          <span className="rounded-full bg-emerald-100 px-3 py-1 text-emerald-800">
            输入净化 {preview.filter_stats.sanitized} 块 / {preview.filter_stats.removed_lines || 0} 行
          </span>
        )}
        <span className="text-gray-400">
          抽样第 {sampledIndexes || '—'} 块 / 共 {preview.total_chunk_count} 块 · {preview.elapsed_seconds.toFixed(1)} 秒
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="overflow-hidden rounded-lg border border-gray-200">
          <div className="border-b border-gray-200 bg-gray-50 px-3 py-2 text-xs font-semibold text-gray-700">
            实体预览{preview.entities_truncated ? '（仅显示前 200 项）' : ''}
          </div>
          <div className="max-h-80 divide-y divide-gray-100 overflow-y-auto">
            {preview.entities.map((entity) => (
              <div key={entity.name} className="px-3 py-2.5">
                <div className="flex items-start justify-between gap-3">
                  <span className="min-w-0 break-words text-sm font-medium text-gray-800">{entity.name}</span>
                  <span className="shrink-0 rounded bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600">
                    {entity.entity_type || '未分类'}
                  </span>
                </div>
                {entity.description && (
                  <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-gray-500">{entity.description}</p>
                )}
              </div>
            ))}
            {preview.entities.length === 0 && (
              <p className="px-3 py-6 text-center text-xs text-gray-400">采样块未抽取到实体</p>
            )}
          </div>
        </div>

        <div className="overflow-hidden rounded-lg border border-gray-200">
          <div className="border-b border-gray-200 bg-gray-50 px-3 py-2 text-xs font-semibold text-gray-700">
            关系预览{preview.relations_truncated ? '（仅显示前 200 项）' : ''}
          </div>
          <div className="max-h-80 divide-y divide-gray-100 overflow-y-auto">
            {preview.relations.map((relation, index) => (
              <div key={`${relation.source}-${relation.target}-${relation.keywords}-${index}`} className="px-3 py-2.5">
                <div className="text-sm text-gray-800">
                  <span className="font-medium">{relation.source}</span>
                  <span className="mx-2 text-gray-300">→</span>
                  <span className="font-medium">{relation.target}</span>
                </div>
                <div className="mt-1 text-xs text-sky-700">{relation.keywords || '未分类关系'}</div>
                {relation.description && (
                  <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-gray-500">{relation.description}</p>
                )}
              </div>
            ))}
            {preview.relations.length === 0 && (
              <p className="px-3 py-6 text-center text-xs text-gray-400">采样块未抽取到关系</p>
            )}
          </div>
        </div>
      </div>

      <p className="text-xs leading-relaxed text-gray-400">
        结果来自均匀采样，不代表全文最终数量；正式索引仍会处理全部有效文本块。当前预览只读取结果，不执行图谱合并和落盘。
      </p>
    </div>
  )
}
