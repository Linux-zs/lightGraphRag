import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { ConfirmProvider } from './components/ConfirmDialog'
import * as api from './api'
import userEvent from '@testing-library/user-event'

vi.mock('./components/Layout', () => ({ default: ({ children, onNewChat, onWorkspaceChange, onCreateWorkspace, activeChatId, workspace }: {
  children: React.ReactNode; onNewChat: () => void; onWorkspaceChange: (workspace: string) => void; onCreateWorkspace: (name: string, rule: string) => void; activeChatId: string | null; workspace: string
}) => <div><button onClick={onNewChat}>new chat</button><button onClick={() => onCreateWorkspace('created', '')}>new workspace</button><button onClick={() => onWorkspaceChange('other')}>switch</button><span data-testid="workspace">{workspace}</span><span data-testid="active">{activeChatId}</span>{children}</div> }))
vi.mock('./pages/QAChat', () => ({ default: () => <p>chat page</p> }))
vi.mock('./api', async (original) => ({
  ...await original<typeof import('./api')>(),
  listWorkspaces: vi.fn(), listChatSessions: vi.fn(), createChatSession: vi.fn(), createWorkspace: vi.fn(),
}))
afterEach(() => { cleanup(); localStorage.clear() })

it('reports failed session loading without deleting saved active selection', async () => {
  localStorage.setItem('lightgraphrag_workspace', 'kb')
  localStorage.setItem('lightgraphrag_active_chat_kb', 'saved')
  vi.mocked(api.listWorkspaces).mockResolvedValue([])
  vi.mocked(api.listChatSessions).mockRejectedValue(new Error('offline'))
  render(<ConfirmProvider><App /></ConfirmProvider>)
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('加载会话失败：offline'))
  expect(localStorage.getItem('lightgraphrag_active_chat_kb')).toBe('saved')
})

it('ignores a chat creation response after switching knowledge bases', async () => {
  vi.mocked(api.listWorkspaces).mockResolvedValue([])
  vi.mocked(api.listChatSessions).mockResolvedValue([])
  let resolve!: (value: Awaited<ReturnType<typeof api.createChatSession>>) => void
  vi.mocked(api.createChatSession).mockReturnValue(new Promise(next => { resolve = next }))
  render(<ConfirmProvider><App /></ConfirmProvider>)
  await userEvent.click(screen.getByRole('button', { name: 'new chat' }))
  await userEvent.click(screen.getByRole('button', { name: 'switch' }))
  resolve({ id: 'old-created', workspace: 'default', title: 'old', created_at: '', updated_at: '', message_count: 0 })
  await waitFor(() => expect(screen.getByTestId('active')).not.toHaveTextContent('old-created'))
  expect(localStorage.getItem('lightgraphrag_active_chat_default')).toBeNull()
})

it('does not navigate to a newly created workspace after the user switched away', async () => {
  const info = (workspace: string): api.WorkspaceInfo => ({
    workspace, is_default: workspace === 'default', doc_count: 0, uploaded_doc_count: 0,
    graph_nodes: 0, graph_edges: 0, manifest_path: '', workspace_path: '', exists: true,
  })
  vi.mocked(api.listWorkspaces).mockResolvedValue([info('default'), info('other'), info('created')])
  vi.mocked(api.listChatSessions).mockResolvedValue([])
  let resolve!: (value: api.WorkspaceInfo) => void
  vi.mocked(api.createWorkspace).mockReturnValue(new Promise(next => { resolve = next }))
  render(<ConfirmProvider><App /></ConfirmProvider>)
  await userEvent.click(screen.getByRole('button', { name: 'new workspace' }))
  await userEvent.click(screen.getByRole('button', { name: 'switch' }))
  resolve(info('created'))
  await waitFor(() => expect(screen.getByTestId('workspace')).toHaveTextContent('other'))
  expect(localStorage.getItem('lightgraphrag_workspace')).toBe('other')
})
