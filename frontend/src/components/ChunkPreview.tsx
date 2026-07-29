interface ChunkItem {
  index: number
  text: string
  char_count: number
}

interface Props {
  chunks: ChunkItem[]
  loading?: boolean
}

export default function ChunkPreview({ chunks, loading }: Props) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        <div className="animate-spin w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full mr-2" />
        <span className="text-sm">正在切分文本...</span>
      </div>
    )
  }

  if (chunks.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400 text-sm">
        暂无切分结果，请先上传文档并配置切分参数。
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm text-gray-500">
          共 <span className="font-semibold text-gray-700">{chunks.length}</span> 个文本块
        </p>
      </div>
      <div className="max-h-[500px] overflow-y-auto space-y-2 pr-1">
        {chunks.map((chunk) => (
          <div
            key={chunk.index}
            className="border border-gray-200 rounded-lg p-3 bg-white hover:border-primary-300 transition-colors"
          >
            <div className="flex items-center gap-2 mb-1.5">
              <span className="inline-flex items-center justify-center w-5 h-5 rounded bg-primary-100 text-primary-700 text-[10px] font-bold">
                {chunk.index + 1}
              </span>
              <span className="text-[11px] text-gray-400">
                {chunk.char_count} 字符
              </span>
            </div>
            <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
              {chunk.text}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
