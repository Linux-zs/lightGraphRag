import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  APP_TOKEN_STORAGE_KEY,
  ApiError,
  apiFetch,
  clearAppToken,
  listWorkspaces,
  setAppToken,
  uploadDocument,
} from './index'

describe('api client hardening', () => {
  beforeEach(() => {
    clearAppToken()
    vi.unstubAllGlobals()
  })

  it('injects the configured token into JSON and FormData requests', async () => {
    setAppToken('secret-token', true)
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await apiFetch('/api/example', {
      method: 'POST',
      body: JSON.stringify({ value: 1 }),
    })
    await apiFetch('/api/upload', {
      method: 'POST',
      body: new FormData(),
    })

    const jsonHeaders = new Headers(fetchMock.mock.calls[0][1].headers)
    const formHeaders = new Headers(fetchMock.mock.calls[1][1].headers)
    expect(jsonHeaders.get('X-App-Token')).toBe('secret-token')
    expect(jsonHeaders.get('Content-Type')).toBe('application/json')
    expect(formHeaders.get('X-App-Token')).toBe('secret-token')
    expect(formHeaders.has('Content-Type')).toBe(false)
  })

  it('supports session-only tokens and emits an auth event on 401', async () => {
    setAppToken('session-token', false)
    const listener = vi.fn()
    window.addEventListener('lightgraphrag-auth-required', listener)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 401 })))

    const response = await apiFetch('/api/protected')

    expect(response.status).toBe(401)
    expect(sessionStorage.getItem(APP_TOKEN_STORAGE_KEY)).toBe('session-token')
    expect(localStorage.getItem(APP_TOKEN_STORAGE_KEY)).toBeNull()
    expect(listener).toHaveBeenCalledOnce()
    window.removeEventListener('lightgraphrag-auth-required', listener)
  })

  it('preserves structured backend detail and request id', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            code: 'MODEL_CONFIG_BUSY',
            detail: '索引任务仍在运行',
            request_id: 'req-123',
          }),
          {
            status: 409,
            headers: { 'Content-Type': 'application/json' },
          },
        ),
      ),
    )

    await expect(listWorkspaces()).rejects.toMatchObject({
      message: '索引任务仍在运行',
      status: 409,
      code: 'MODEL_CONFIG_BUSY',
      requestId: 'req-123',
    })
  })

  it('rejects oversized documents before starting a request', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const file = new File([new Uint8Array(50 * 1024 * 1024 + 1)], 'large.txt')

    await expect(uploadDocument(file, 'kb')).rejects.toThrow('50 MiB')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
