import { useCallback, useEffect, useReducer, useRef } from 'react'
import { updateChatSessionSettings, type ChatSettings } from '../api'

type SaveStatus = 'idle' | 'pending' | 'saving' | 'saved' | 'error'
type Entry = {
  workspace: string
  sessionId: string
  status: SaveStatus
  error: string
  draft?: ChatSettings
  pending?: ChatSettings
  timer?: ReturnType<typeof setTimeout>
  running: boolean
  finished?: Promise<void>
}

/** Serialize saves per conversation, retaining the newest draft during failures. */
export function useChatSettingsSave(workspace: string, sessionId: string | null) {
  const entries = useRef(new Map<string, Entry>())
  const mounted = useRef(false)
  const [, redraw] = useReducer((value: number) => value + 1, 0)
  const key = JSON.stringify([workspace, sessionId])
  const notify = useCallback(() => { if (mounted.current) redraw() }, [])

  const flush = useCallback(async (entry: Entry) => {
    if (entry.timer !== undefined) clearTimeout(entry.timer)
    entry.timer = undefined
    if (entry.running) return entry.finished
    if (!entry.pending) return
    entry.running = true
    let finish!: () => void
    entry.finished = new Promise<void>((resolve) => { finish = resolve })
    try {
      while (entry.pending) {
        const payload = entry.pending
        entry.pending = undefined
        entry.status = 'saving'
        entry.error = ''
        notify()
        try {
          await updateChatSessionSettings(entry.sessionId, payload, entry.workspace)
          entry.status = entry.pending ? 'pending' : 'saved'
        } catch (error) {
          entry.status = entry.pending ? 'pending' : 'error'
          entry.error = error instanceof Error ? error.message : '请求失败'
        }
        notify()
      }
    } finally {
      entry.running = false
      finish()
    }
  }, [notify])

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  useEffect(() => () => {
    // Capture the old key: switching/unmounting must not send its draft to the
    // new conversation or silently discard a debounce that has not fired.
    const entry = entries.current.get(key)
    if (entry) void flush(entry)
  }, [key, flush])

  const schedule = useCallback((settings: ChatSettings) => {
    if (!sessionId) return
    let entry = entries.current.get(key)
    if (!entry) {
      entry = { workspace, sessionId, status: 'idle', error: '', running: false }
      entries.current.set(key, entry)
    }
    entry.draft = { ...settings }
    entry.pending = entry.draft
    entry.status = 'pending'
    entry.error = ''
    if (entry.timer !== undefined) clearTimeout(entry.timer)
    const target = entry
    entry.timer = setTimeout(() => { void flush(target) }, 350)
    notify()
  }, [key, workspace, sessionId, flush, notify])

  const retry = useCallback(() => {
    const entry = entries.current.get(key)
    if (!entry?.draft || entry.running) return
    entry.pending = entry.draft
    void flush(entry)
  }, [key, flush])

  const getUnsavedDraft = useCallback((targetWorkspace: string, targetId: string) => {
    const entry = entries.current.get(JSON.stringify([targetWorkspace, targetId]))
    return entry && entry.status !== 'saved' ? entry.draft : undefined
  }, [])

  const flushCurrent = useCallback(async () => {
    const entry = entries.current.get(key)
    if (entry) await flush(entry)
  }, [key, flush])

  const entry = entries.current.get(key)
  return { status: entry?.status ?? 'idle', error: entry?.error ?? '', schedule, retry, getUnsavedDraft, flushCurrent }
}
