import { describe, expect, it } from 'vitest'
import { locateCitationChunk } from './citationLocator'

const chunks = [
  { chunk_id: 'first', chunk_index: 0, text: 'unrelated', char_count: 9 },
  { chunk_id: 'evidence', chunk_index: 8, text: 'actual evidence text', char_count: 20 },
]
describe('citation source locator', () => {
  it('locates by ID regardless of retrieval rank', () => {
    expect(locateCitationChunk(chunks, 'evidence')?.chunk_index).toBe(8)
  })
  it('does not substitute another block for an obsolete ID', () => {
    expect(locateCitationChunk(chunks, 'removed', 'actual evidence')).toBeUndefined()
  })
  it('supports unique legacy excerpt matches', () => {
    expect(locateCitationChunk(chunks, '', 'actual evidence')?.chunk_id).toBe('evidence')
  })
  it('rejects ambiguous or empty legacy references', () => {
    expect(locateCitationChunk([...chunks, { ...chunks[1], chunk_id: 'duplicate' }], '', 'actual')).toBeUndefined()
    expect(locateCitationChunk(chunks)).toBeUndefined()
  })
})
