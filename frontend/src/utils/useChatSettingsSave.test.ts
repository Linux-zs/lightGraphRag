import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { updateChatSessionSettings, type ChatSettings } from '../api'
import { useChatSettingsSave } from './useChatSettingsSave'

vi.mock('../api', () => ({ updateChatSessionSettings: vi.fn() }))
const settings: ChatSettings = {
  answer_profile_id: 'fixture', answer_model: 'fixture', temperature: 0.7, top_p: 0.9,
  max_tokens: 4096, frequency_penalty: 0, presence_penalty: 0,
  mode: 'mix', top_k: 40, chunk_top_k: 20, enable_rerank: false,
}
function deferred() {
  let resolve!: (value: ChatSettings) => void
  let reject!: (error: Error) => void
  const promise = new Promise<ChatSettings>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
beforeEach(() => {
  vi.useFakeTimers()
  vi.mocked(updateChatSessionSettings).mockReset().mockResolvedValue(settings)
})
afterEach(() => { cleanup(); vi.useRealTimers() })

it('serializes in-flight requests and coalesces pending changes to the newest draft', async () => {
  const first = deferred()
  vi.mocked(updateChatSessionSettings).mockReturnValueOnce(first.promise)
  const { result } = renderHook(() => useChatSettingsSave('kb', 'one'))
  act(() => result.current.schedule(settings))
  await act(() => vi.advanceTimersByTimeAsync(350))
  expect(result.current.status).toBe('saving')
  act(() => result.current.schedule({ ...settings, context_window: 16384 }))
  act(() => result.current.schedule({ ...settings, context_window: 65536 }))
  await act(() => vi.advanceTimersByTimeAsync(350))
  expect(updateChatSessionSettings).toHaveBeenCalledTimes(1)
  await act(async () => { first.resolve(settings); await first.promise })
  expect(updateChatSessionSettings).toHaveBeenCalledTimes(2)
  expect(updateChatSessionSettings).toHaveBeenLastCalledWith('one', expect.objectContaining({ context_window: 65536 }), 'kb')
  expect(result.current.status).toBe('saved')
})

it('retains a failed draft and retries without throwing an unhandled rejection', async () => {
  vi.mocked(updateChatSessionSettings).mockRejectedValueOnce(new Error('offline'))
  const { result } = renderHook(() => useChatSettingsSave('kb', 'one'))
  act(() => result.current.schedule(settings))
  await act(() => vi.advanceTimersByTimeAsync(350))
  expect(result.current.status).toBe('error')
  expect(result.current.error).toBe('offline')
  expect(result.current.getUnsavedDraft('kb', 'one')).toEqual(settings)
  await act(async () => result.current.retry())
  expect(result.current.status).toBe('saved')
  expect(result.current.getUnsavedDraft('kb', 'one')).toBeUndefined()
})

it('waits for the active save and its newest pending payload before releasing a sender', async () => {
  const first = deferred()
  const second = deferred()
  vi.mocked(updateChatSessionSettings).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise)
  const { result } = renderHook(() => useChatSettingsSave('kb', 'one'))
  act(() => result.current.schedule(settings))
  await act(() => vi.advanceTimersByTimeAsync(350))
  act(() => result.current.schedule({ ...settings, context_window: 16384 }))
  let released = false
  const wait = result.current.flushCurrent().then(() => { released = true })
  await act(async () => { first.resolve(settings); await first.promise })
  expect(released).toBe(false)
  expect(updateChatSessionSettings).toHaveBeenCalledTimes(2)
  await act(async () => { second.resolve(settings); await wait })
  expect(released).toBe(true)
})

it('flushes on navigation to the captured owner and isolates late failure from the new scope', async () => {
  const old = deferred()
  vi.mocked(updateChatSessionSettings).mockReturnValueOnce(old.promise)
  const { result, rerender } = renderHook(({ workspace, id }) => useChatSettingsSave(workspace, id), {
    initialProps: { workspace: 'old', id: 'one' },
  })
  act(() => result.current.schedule(settings))
  rerender({ workspace: 'new', id: 'two' })
  expect(updateChatSessionSettings).toHaveBeenCalledWith('one', settings, 'old')
  await act(async () => { old.reject(new Error('old failed')); await old.promise.catch(() => {}) })
  expect(result.current.status).toBe('idle')
  expect(result.current.error).toBe('')
  rerender({ workspace: 'old', id: 'one' })
  expect(result.current.status).toBe('error')
  expect(result.current.getUnsavedDraft('old', 'one')).toEqual(settings)
})

it('flushes pending changes on unmount and does not save a nonexistent session', async () => {
  const { result, rerender, unmount } = renderHook(({ id }: { id: string | null }) => useChatSettingsSave('kb', id), {
    initialProps: { id: null as string | null },
  })
  act(() => result.current.schedule(settings))
  await act(() => vi.advanceTimersByTimeAsync(350))
  expect(updateChatSessionSettings).not.toHaveBeenCalled()
  rerender({ id: 'one' })
  act(() => result.current.schedule(settings))
  unmount()
  expect(updateChatSessionSettings).toHaveBeenCalledWith('one', settings, 'kb')
})
