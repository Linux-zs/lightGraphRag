import type { DocumentChunkItem } from '../api'

/** Never interpret the historical retrieval rank as a source chunk number. */
export function locateCitationChunk(chunks: DocumentChunkItem[], chunkId?: string, excerpt = '') {
  if (chunkId) return chunks.find(chunk => chunk.chunk_id === chunkId)
  const text = excerpt.trim()
  if (!text) return undefined
  const matches = chunks.filter(chunk => chunk.text.trim().startsWith(text))
  return matches.length === 1 ? matches[0] : undefined
}
