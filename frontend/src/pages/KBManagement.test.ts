import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getIndexTaskWithRetry } from './KBManagement'
import * as api from '../api'
import type { IndexTask } from '../api'

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>()
  return {
    ...actual,
    getIndexTask: vi.fn(),
  }
})

describe('index task polling recovery', () => {
  beforeEach(() => {
    vi.mocked(api.getIndexTask).mockReset()
  })

  it('recovers after a temporary first request failure', async () => {
    const task: IndexTask = {
      task_id: 'task-1',
      kind: 'single',
      workspace: 'kb',
      status: 'running',
      doc_names: ['doc.txt'],
      total: 1,
      current: 0,
      progress: 0,
      current_doc: 'doc.txt',
      message: 'running',
      results: [],
      errors: [],
      created_at: '',
      updated_at: '',
    }
    vi.mocked(api.getIndexTask)
      .mockRejectedValueOnce(new TypeError('temporary network error'))
      .mockResolvedValueOnce(task)
    const wait = vi.fn().mockResolvedValue(undefined)

    await expect(
      getIndexTaskWithRetry('task-1', undefined, 3, wait),
    ).resolves.toEqual(task)

    expect(api.getIndexTask).toHaveBeenCalledTimes(2)
    expect(wait).toHaveBeenCalledWith(500)
  })

  it('does not retry an aborted request', async () => {
    const controller = new AbortController()
    controller.abort()
    vi.mocked(api.getIndexTask).mockRejectedValueOnce(
      new DOMException('aborted', 'AbortError'),
    )
    const wait = vi.fn()

    await expect(
      getIndexTaskWithRetry('task-2', controller.signal, 3, wait),
    ).rejects.toMatchObject({ name: 'AbortError' })

    expect(api.getIndexTask).toHaveBeenCalledOnce()
    expect(wait).not.toHaveBeenCalled()
  })
})
