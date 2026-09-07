import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import KBManagement from './KBManagement'
import * as api from '../api'
import type { IndexTask, UploadedDocument, ChunkPreviewItem } from '../api'

vi.mock('../api', async (importOriginal) => ({
  ...await importOriginal<typeof import('../api')>(),
  listDocuments: vi.fn(), listIndexTasks: vi.fn(), getGraphGovernanceConfig: vi.fn(),
  uploadDocument: vi.fn(), previewChunks: vi.fn(), indexDocument: vi.fn(), getIndexTask: vi.fn(),
  batchIndexDocuments: vi.fn(), getDocumentChunks: vi.fn(),
}))

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

const uploaded = { file_name: 'old.txt', file_type: 'txt', char_count: 12 } as UploadedDocument
const task: IndexTask = {
  task_id: 'old-task', workspace: 'a', kind: 'single', status: 'running', doc_names: ['old.txt'],
  total: 1, current: 0, progress: 0, message: 'processing', results: [], errors: [],
  created_at: '', updated_at: '',
}
const page = (workspace: string) => <KBManagement workspace={workspace} isDefaultWorkspace={false} onDeleteWorkspace={vi.fn()} />
const settle = () => act(async () => { await Promise.resolve() })
async function upload(container: HTMLElement) {
  fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [new File(['text'], 'old.txt')] } })
  await settle()
}

beforeEach(() => {
  vi.resetAllMocks()
  vi.mocked(api.listDocuments).mockResolvedValue([])
  vi.mocked(api.listIndexTasks).mockResolvedValue([])
  vi.mocked(api.getGraphGovernanceConfig).mockRejectedValue(new Error('no rule'))
  vi.mocked(api.uploadDocument).mockResolvedValue(uploaded)
})
afterEach(() => { cleanup(); vi.useRealTimers() })

it('discards a late upload and resets loading when switching workspace', async () => {
  const late = deferred<UploadedDocument>()
  vi.mocked(api.uploadDocument).mockReturnValueOnce(late.promise)
  const view = render(page('a'))
  await upload(view.container)
  const signal = vi.mocked(api.uploadDocument).mock.calls[0][2]!
  view.rerender(page('b'))
  expect(signal.aborted).toBe(true)
  expect(screen.queryByText('正在上传文档...')).toBeNull()
  await act(async () => late.resolve(uploaded))
  expect(screen.queryByText('old.txt')).toBeNull()
})

it('does not display old preview results after a workspace switch', async () => {
  const late = deferred<ChunkPreviewItem[]>()
  vi.mocked(api.previewChunks).mockReturnValueOnce(late.promise)
  const view = render(page('a'))
  await upload(view.container)
  fireEvent.click(screen.getByRole('button', { name: '预览切分' }))
  view.rerender(page('b'))
  await act(async () => late.resolve([{ index: 0, text: 'stale preview', char_count: 13 }]))
  expect(screen.queryByText('stale preview')).toBeNull()
  expect(vi.mocked(api.previewChunks).mock.calls[0][1]?.aborted).toBe(true)
})

it('does not attach an old task whose creation response arrived in a new workspace', async () => {
  const late = deferred<IndexTask>()
  vi.mocked(api.indexDocument).mockReturnValueOnce(late.promise)
  const view = render(page('a'))
  await upload(view.container)
  fireEvent.click(screen.getByRole('button', { name: '确认索引' }))
  view.rerender(page('b'))
  await act(async () => late.resolve(task))
  expect(api.getIndexTask).not.toHaveBeenCalled()
  expect(screen.queryByText(/old-task/)).toBeNull()
  expect(vi.mocked(api.indexDocument).mock.calls[0][1]?.aborted).toBe(true)
})

it('recovers restored polling after exhausting network retries without an unhandled rejection', async () => {
  vi.useFakeTimers()
  vi.mocked(api.listIndexTasks).mockResolvedValue([task])
  vi.mocked(api.getIndexTask)
    .mockRejectedValueOnce(new TypeError('offline'))
    .mockRejectedValueOnce(new TypeError('offline'))
    .mockRejectedValueOnce(new TypeError('offline'))
    .mockResolvedValue({ ...task, status: 'succeeded', message: 'recovered completion', current: 1 })
  render(page('a'))
  await settle()
  await act(async () => vi.advanceTimersByTimeAsync(1500))
  expect(screen.getByText(/任务状态同步中断/)).toBeInTheDocument()
  await act(async () => vi.advanceTimersByTimeAsync(2000))
  expect(api.getIndexTask).toHaveBeenCalledTimes(4)
  expect(screen.getAllByText(/recovered completion/).length).toBeGreaterThan(0)
  expect(screen.queryByRole('button', { name: '取消任务' })).toBeNull()
})

it('discards delayed batch creation and clears the previous selection', async () => {
  vi.mocked(api.listDocuments).mockResolvedValue([{ doc_id: 'probe', doc_name: 'same.txt', file_type: 'txt', chunk_count: 1 }])
  const late = deferred<IndexTask>()
  vi.mocked(api.batchIndexDocuments).mockReturnValueOnce(late.promise)
  const view = render(page('a'))
  await settle()
  fireEvent.click(screen.getAllByRole('checkbox')[0])
  fireEvent.click(screen.getByRole('button', { name: '批量索引' }))
  view.rerender(page('b'))
  await act(async () => late.resolve({ ...task, kind: 'batch' }))
  expect(api.getIndexTask).not.toHaveBeenCalled()
  expect(screen.queryByText(/已选 1 个文档/)).toBeNull()
})

it('does not reopen a chunk viewer for a same-named document in another workspace', async () => {
  vi.mocked(api.listDocuments).mockResolvedValue([{ doc_id: 'probe', doc_name: 'same.txt', file_type: 'txt', chunk_count: 1 }])
  const late = deferred<api.DocumentChunksResponse>()
  vi.mocked(api.getDocumentChunks).mockReturnValueOnce(late.promise)
  const view = render(page('a'))
  await settle()
  fireEvent.click(screen.getByRole('button', { name: 'same.txt' }))
  view.rerender(page('b'))
  await act(async () => late.resolve({doc_name:'same.txt', total:1, chunks:[{chunk_id:'old', chunk_index:0, text:'old workspace chunk', char_count:19}]}))
  expect(screen.queryByText('old workspace chunk')).toBeNull()
  expect(vi.mocked(api.getDocumentChunks).mock.calls[0][2]?.aborted).toBe(true)
})

it('stops retries and aborts the captured request on unmount', async () => {
  vi.useFakeTimers()
  vi.mocked(api.listIndexTasks).mockResolvedValue([task])
  vi.mocked(api.getIndexTask).mockRejectedValue(new TypeError('offline'))
  const view = render(page('a'))
  await settle()
  const signal = vi.mocked(api.getIndexTask).mock.calls[0][1]!
  view.unmount()
  await act(async () => vi.advanceTimersByTimeAsync(20000))
  expect(signal.aborted).toBe(true)
  expect(api.getIndexTask).toHaveBeenCalledTimes(1)
})

it('retries an initial task list failure and restores a persisted task', async () => {
  vi.useFakeTimers()
  vi.mocked(api.listIndexTasks).mockRejectedValueOnce(new TypeError('offline')).mockResolvedValue([task])
  vi.mocked(api.getIndexTask).mockResolvedValue({ ...task, status: 'succeeded', message: 'restored completion' })
  render(page('a'))
  await settle()
  await act(async () => vi.advanceTimersByTimeAsync(5000))
  expect(screen.getAllByText(/restored completion/).length).toBeGreaterThan(0)
})
