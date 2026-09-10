import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import QAChat from './QAChat'
import { ConfirmProvider } from '../components/ConfirmDialog'
import * as api from '../api'

vi.mock('../components/GraphView', () => ({
  default: () => <div data-testid="graph-view" />,
}))

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>()
  return {
    ...actual,
    getChatSession: vi.fn(),
    getDocumentChunks: vi.fn(),
    listModelProfiles: vi.fn(),
    getModelBindings: vi.fn(),
    getModelConfig: vi.fn(),
    chatSendStream: vi.fn(),
    updateChatSessionSettings: vi.fn(),
  }
})

const workspaces = [
  {
    workspace: 'old',
    is_default: true,
    doc_count: 0,
    uploaded_doc_count: 0,
    graph_nodes: 0,
    graph_edges: 0,
    manifest_path: '',
    workspace_path: '',
    exists: true,
  },
  {
    workspace: 'new',
    is_default: false,
    doc_count: 0,
    uploaded_doc_count: 0,
    graph_nodes: 0,
    graph_edges: 0,
    manifest_path: '',
    workspace_path: '',
    exists: true,
  },
]

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((next) => {
    resolve = next
  })
  return { promise, resolve }
}

function session(id: string, workspace: string, content: string) {
  return {
    id,
    workspace,
    title: content,
    settings: {
      answer_profile_id: 'default',
      answer_model: 'model',
      temperature: 0.7,
      top_p: 0.9,
      max_tokens: 4096,
      frequency_penalty: 0,
      presence_penalty: 0,
      mode: 'mix',
      top_k: 40,
      chunk_top_k: 20,
      enable_rerank: true,
    },
    messages: [{ role: 'assistant' as const, content, timestamp: '' }],
    created_at: '',
    updated_at: '',
  }
}

function renderChat(props: Partial<React.ComponentProps<typeof QAChat>> = {}) {
  return render(
    <ConfirmProvider>
      <QAChat
        workspace="old"
        workspaces={workspaces}
        onWorkspaceChange={vi.fn()}
        sessions={[]}
        activeId={null}
        setActiveId={vi.fn()}
        reloadSessions={vi.fn().mockResolvedValue(undefined)}
        {...props}
      />
    </ConfirmProvider>,
  )
}

describe('QAChat async state', () => {
  afterEach(cleanup)

  it('shows verified source pages and does not reopen a closed citation on late response', async () => {
    const saved = session('cited', 'old', 'answer')
    vi.mocked(api.getChatSession).mockResolvedValue({ ...saved, messages: [{
      role: 'assistant', content: 'answer', timestamp: '',
      citations: [{ index: 1, doc_name: 'source.pdf', chunk_index: -1, chunk_id: 'evidence', excerpt: 'evidence' }],
    }] })
    const response = { doc_name: 'source.pdf', total: 1, chunks: [{
      chunk_id: 'evidence', chunk_index: 8, text: 'evidence text', char_count: 13,
      source_location: { start: 12, end: 25, pages: [3, 4], text_sha256: 'hash' },
    }] }
    vi.mocked(api.getDocumentChunks).mockResolvedValueOnce(response)
    renderChat({ activeId: 'cited' })
    await userEvent.click(await screen.findByRole('button', { name: /引用文档/ }))
    const trigger = screen.getByRole('button', { name: 'source.pdf' })
    await userEvent.click(trigger)
    expect(await screen.findByText(/原文第 3、4 页/)).toBeInTheDocument()
    await userEvent.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()

    const pending = deferred<typeof response>()
    vi.mocked(api.getDocumentChunks).mockReturnValueOnce(pending.promise)
    await userEvent.click(trigger)
    await userEvent.click(screen.getByRole('button', { name: '关闭' }))
    const calls = vi.mocked(api.getDocumentChunks).mock.calls
    expect(calls[calls.length - 1][2]?.aborted).toBe(true)
    pending.resolve(response)
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })

  beforeEach(() => {
    vi.mocked(api.listModelProfiles).mockResolvedValue([])
    vi.mocked(api.getModelBindings).mockResolvedValue({
      chat: { profile_id: 'default', model: 'model' },
      kg: { profile_id: 'default', model: 'model' },
      embedding: {
        profile_id: 'default',
        model: 'embed',
        embed_dim: 1024,
        embed_max_chars: 480,
      },
      rerank: { profile_id: 'default', model: 'rerank', enabled: true },
    })
    vi.mocked(api.getModelConfig).mockResolvedValue({
      workspace: 'old',
      embed_model: 'embed',
      embed_base_url: '',
      rerank_model: 'rerank',
      chat_model: 'model',
      chat_temperature: 0.7,
      chat_top_p: 0.9,
      chat_max_tokens: 4096,
      frequency_penalty: 0,
      presence_penalty: 0,
      answer_prompt_template_id: '',
      answer_system_prompt: '',
    })
  })

  it('exposes and saves the total context budget independently of output length', async () => {
    vi.mocked(api.getChatSession).mockResolvedValue(session('budget', 'old', 'answer'))
    vi.mocked(api.updateChatSessionSettings).mockImplementation(async (_id, settings) => settings)
    renderChat({ activeId: 'budget' })
    await screen.findByText('answer')
    await userEvent.click(screen.getByRole('button', { name: /mix.*20 块/ }))
    const select = screen.getByRole('combobox', { name: '上下文总预算（输入＋输出）' })
    expect(select).toHaveValue('32768')
    await userEvent.selectOptions(select, '16384')
    await waitFor(() => expect(api.updateChatSessionSettings).toHaveBeenCalledWith(
      'budget', expect.objectContaining({ context_window: 16384, max_tokens: 4096 }), 'old',
    ))
    expect(screen.getByText(/并非自动识别模型窗口/)).toBeInTheDocument()
  })

  it('shows settings save failure and retries the same draft', async () => {
    vi.mocked(api.getChatSession).mockResolvedValue(session('budget', 'old', 'answer'))
    vi.mocked(api.updateChatSessionSettings).mockRejectedValueOnce(new Error('offline'))
      .mockImplementationOnce(async (_id, settings) => settings)
    renderChat({ activeId: 'budget' })
    await screen.findByText('answer')
    await userEvent.click(screen.getByRole('button', { name: /mix.*20 块/ }))
    await userEvent.selectOptions(screen.getByRole('combobox', { name: '上下文总预算（输入＋输出）' }), '16384')
    expect(await screen.findByRole('alert')).toHaveTextContent('设置保存失败')
    await userEvent.click(screen.getByRole('button', { name: '重试保存' }))
    expect(await screen.findByText('设置已保存')).toBeInTheDocument()
    expect(api.updateChatSessionSettings).toHaveBeenLastCalledWith(
      'budget', expect.objectContaining({ context_window: 16384 }), 'old',
    )
  })

  it('drops a late session response after switching workspace and session', async () => {
    const oldRequest = deferred<ReturnType<typeof session>>()
    vi.mocked(api.getChatSession)
      .mockReturnValueOnce(oldRequest.promise)
      .mockResolvedValueOnce(session('new-session', 'new', 'new answer'))

    const view = renderChat({ workspace: 'old', activeId: 'old-session' })
    view.rerender(
      <ConfirmProvider>
        <QAChat
          workspace="new"
          workspaces={workspaces}
          onWorkspaceChange={vi.fn()}
          sessions={[]}
          activeId="new-session"
          setActiveId={vi.fn()}
          reloadSessions={vi.fn().mockResolvedValue(undefined)}
        />
      </ConfirmProvider>,
    )

    expect(await screen.findByText('new answer')).toBeInTheDocument()
    expect(
      (vi.mocked(api.getChatSession).mock.calls[0][2] as AbortSignal).aborted,
    ).toBe(true)
    oldRequest.resolve(session('old-session', 'old', 'old answer'))
    await waitFor(() => expect(screen.queryByText('old answer')).not.toBeInTheDocument())
  })

  it('aborts an active answer stream when the workspace changes', async () => {
    const streamRequest = deferred<Response>()
    vi.mocked(api.chatSendStream).mockReturnValue(streamRequest.promise)

    const view = renderChat()
    const input = screen.getByPlaceholderText('输入问题，按 Enter 发送')
    await userEvent.type(input, 'question{enter}')
    await waitFor(() => expect(api.chatSendStream).toHaveBeenCalledOnce())

    const signal = vi.mocked(api.chatSendStream).mock.calls[0][1] as AbortSignal
    expect(signal.aborted).toBe(false)
    view.rerender(
      <ConfirmProvider>
        <QAChat
          workspace="new"
          workspaces={workspaces}
          onWorkspaceChange={vi.fn()}
          sessions={[]}
          activeId={null}
          setActiveId={vi.fn()}
          reloadSessions={vi.fn().mockResolvedValue(undefined)}
        />
      </ConfirmProvider>,
    )

    expect(signal.aborted).toBe(true)
    streamRequest.resolve(new Response('', { status: 200 }))
  })

  it('keeps partial answer content when the SSE stream reports an error', async () => {
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(
          encoder.encode(
            'data: {"token":"partial answer"}\n\n'
            + 'event: error\n'
            + 'data: {"code":"ANSWER_GENERATION_FAILED","detail":"boom","content":"partial answer"}\n\n',
          ),
        )
        controller.close()
      },
    })
    vi.mocked(api.chatSendStream).mockResolvedValue(
      new Response(stream, { status: 200 }),
    )

    renderChat()
    const input = screen.getByPlaceholderText('输入问题，按 Enter 发送')
    await userEvent.type(input, 'question{enter}')

    expect(await screen.findByText('partial answer')).toBeInTheDocument()
    expect(screen.queryByText(/回答生成失败/)).not.toBeInTheDocument()
  })
})
