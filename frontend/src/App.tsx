import { FormEvent, useEffect, useRef, useState } from 'react'
import Layout from './components/Layout'
import QAChat from './pages/QAChat'
import KBManagement from './pages/KBManagement'
import RecallTest from './pages/RecallTest'
import GraphPage from './pages/GraphPage'
import ModelSettings from './pages/ModelSettings'
import Dashboard from './pages/Dashboard'
import {
  ChatSessionListItem,
  createChatSession,
  createWorkspace,
  deleteChatSession,
  deleteWorkspace,
  listChatSessions,
  listWorkspaces,
  clearAppToken,
  getAppToken,
  setAppToken,
  WorkspaceInfo,
} from './api'
import { useConfirm } from './components/ConfirmDialog'

type Page = 'chat' | 'kb' | 'recall' | 'graph' | 'models' | 'dashboard'
const PAGES: Page[] = ['chat', 'kb', 'recall', 'graph', 'models', 'dashboard']
const DEFAULT_PAGE: Page = 'chat'
const DEFAULT_WORKSPACE = 'default'
const PAGE_STORAGE_KEY = 'lightgraphrag_page'
const WORKSPACE_STORAGE_KEY = 'lightgraphrag_workspace'

function isPage(value: string | null): value is Page {
  return !!value && PAGES.includes(value as Page)
}

function pageFromHash(): Page | null {
  const raw = window.location.hash.replace(/^#\/?/, '').split('/')[0]
  return isPage(raw) ? raw : null
}

function initialPage(): Page {
  const storedPage = localStorage.getItem(PAGE_STORAGE_KEY)
  return pageFromHash() || (isPage(storedPage) ? storedPage : DEFAULT_PAGE)
}

function pageHash(page: Page) {
  return `#/${page}`
}

function chatStorageKey(workspace: string) {
  return `lightgraphrag_active_chat_${workspace}`
}

export default function App() {
  const confirm = useConfirm()
  const [page, setPage] = useState<Page>(() => initialPage())
  const [workspace, setWorkspace] = useState(
    () => localStorage.getItem(WORKSPACE_STORAGE_KEY) || DEFAULT_WORKSPACE,
  )
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([])
  const [chatSessions, setChatSessions] = useState<ChatSessionListItem[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(
    () => localStorage.getItem(
      chatStorageKey(localStorage.getItem(WORKSPACE_STORAGE_KEY) || DEFAULT_WORKSPACE),
    ),
  )
  const [authOpen, setAuthOpen] = useState(false)
  const [authToken, setAuthToken] = useState(() => getAppToken())
  const [rememberToken, setRememberToken] = useState(true)
  const workspaceRequestRef = useRef(0)
  const sessionRequestRef = useRef(0)

  const loadWorkspaces = async (current = workspace, signal?: AbortSignal) => {
    const requestId = ++workspaceRequestRef.current
    try {
      const data = await listWorkspaces(signal)
      if (requestId !== workspaceRequestRef.current) return []
      setWorkspaces(data)
      if (!data.some((item) => item.workspace === current) && data[0]) {
        setWorkspace(data[0].workspace)
        localStorage.setItem(WORKSPACE_STORAGE_KEY, data[0].workspace)
      }
      return data
    } catch {/* ignore */}
    return []
  }

  const loadChatSessions = async (targetWorkspace = workspace, signal?: AbortSignal) => {
    const requestId = ++sessionRequestRef.current
    try {
      const data = await listChatSessions(targetWorkspace, signal)
      if (requestId !== sessionRequestRef.current) return []
      setChatSessions(data)
      return data
    } catch {/* ignore */}
    return []
  }

  const reloadCurrentChatSessions = async () => {
    await loadChatSessions(workspace)
  }

  useEffect(() => {
    const controller = new AbortController()
    void loadWorkspaces(workspace, controller.signal)
    if (!pageFromHash()) {
      window.history.replaceState(null, '', pageHash(page))
    }
    const handleHashChange = () => {
      const next = pageFromHash()
      if (!next) return
      setPage(next)
      localStorage.setItem(PAGE_STORAGE_KEY, next)
    }
    window.addEventListener('hashchange', handleHashChange)
    const showAuth = () => setAuthOpen(true)
    window.addEventListener('lightgraphrag-auth-required', showAuth)
    return () => {
      controller.abort()
      window.removeEventListener('hashchange', handleHashChange)
      window.removeEventListener('lightgraphrag-auth-required', showAuth)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    const storedActive = localStorage.getItem(chatStorageKey(workspace))
    setActiveChatId(storedActive)
    setChatSessions([])
    void loadChatSessions(workspace, controller.signal).then((sessions) => {
      if (storedActive && !sessions.some((session) => session.id === storedActive)) {
        setActiveChatId(null)
        localStorage.removeItem(chatStorageKey(workspace))
      }
    })
    return () => controller.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace])

  const handleNavigate = (next: Page) => {
    setPage(next)
    localStorage.setItem(PAGE_STORAGE_KEY, next)
    const nextHash = pageHash(next)
    if (window.location.hash !== nextHash) {
      window.location.hash = nextHash
    }
  }

  const handleSetActiveChatId = (id: string | null) => {
    setActiveChatId(id)
    if (id) {
      localStorage.setItem(chatStorageKey(workspace), id)
    } else {
      localStorage.removeItem(chatStorageKey(workspace))
    }
  }

  const handleWorkspaceChange = (next: string) => {
    if (next === workspace) return
    setActiveChatId(localStorage.getItem(chatStorageKey(next)))
    setWorkspace(next)
    localStorage.setItem(WORKSPACE_STORAGE_KEY, next)
  }

  const handleCreateWorkspace = async (name: string, ruleTemplateId: string) => {
    const created = await createWorkspace(name, ruleTemplateId)
    await loadWorkspaces()
    handleWorkspaceChange(created.workspace)
  }

  const handleDeleteWorkspace = async (target: string) => {
    const info = workspaces.find((item) => item.workspace === target)
    if (info?.is_default) return
    const ok = await confirm({
      title: '删除知识库',
      message: `将删除知识库“${target}”的上传文档、LightRAG 索引、manifest、图谱和会话入口。该操作不可撤销。`,
      confirmLabel: '删除知识库',
      tone: 'danger',
    })
    if (!ok) return
    try {
      await deleteWorkspace(target)
      const nextWorkspaces = await loadWorkspaces(target)
      if (target === workspace) {
        const next = nextWorkspaces[0]?.workspace || DEFAULT_WORKSPACE
        setWorkspace(next)
        localStorage.setItem(WORKSPACE_STORAGE_KEY, next)
        handleSetActiveChatId(null)
        setChatSessions([])
        void loadChatSessions(next)
        handleNavigate('chat')
      }
    } catch (error) {
      throw new Error((error as Error).message || '删除知识库失败')
    }
  }

  const handleNewChat = async () => {
    try {
      const created = await createChatSession(workspace)
      setChatSessions((prev) => [created, ...prev])
      handleSetActiveChatId(created.id)
      handleNavigate('chat')
    } catch {/* ignore */}
  }

  const handleSelectChat = (id: string) => {
    handleSetActiveChatId(id)
    handleNavigate('chat')
  }

  const handleDeleteChat = async (id: string) => {
    try {
      await deleteChatSession(id, workspace)
      setChatSessions((prev) => prev.filter((item) => item.id !== id))
      if (activeChatId === id) {
        handleSetActiveChatId(null)
      }
    } catch {/* ignore */}
  }

  const submitToken = (event: FormEvent) => {
    event.preventDefault()
    setAppToken(authToken, rememberToken)
    setAuthOpen(false)
    void loadWorkspaces(workspace)
    void loadChatSessions(workspace)
  }

  return (
    <>
    <Layout
      currentPage={page}
      onNavigate={handleNavigate}
      workspace={workspace}
      workspaces={workspaces}
      onWorkspaceChange={handleWorkspaceChange}
      onCreateWorkspace={handleCreateWorkspace}
      chatSessions={chatSessions}
      activeChatId={activeChatId}
      onNewChat={handleNewChat}
      onSelectChat={handleSelectChat}
      onDeleteChat={handleDeleteChat}
    >
      {page === 'chat' && (
        <QAChat
          workspace={workspace}
          workspaces={workspaces}
          onWorkspaceChange={handleWorkspaceChange}
          sessions={chatSessions}
          activeId={activeChatId}
          setActiveId={handleSetActiveChatId}
          reloadSessions={reloadCurrentChatSessions}
        />
      )}
      {page === 'kb' && (
        <KBManagement
          workspace={workspace}
          isDefaultWorkspace={
            workspaces.find((item) => item.workspace === workspace)?.is_default !== false
          }
          onDeleteWorkspace={() => handleDeleteWorkspace(workspace)}
        />
      )}
      {page === 'recall' && <RecallTest workspace={workspace} />}
      {page === 'graph' && <GraphPage workspace={workspace} />}
      {page === 'models' && <ModelSettings workspace={workspace} />}
      {page === 'dashboard' && <Dashboard workspace={workspace} onWorkspaceChanged={loadWorkspaces} />}
    </Layout>
    {authOpen && (
      <div className="fixed inset-0 z-[200] grid place-items-center bg-gray-950/35 p-4">
        <form
          onSubmit={submitToken}
          className="w-full max-w-sm rounded-lg border border-gray-200 bg-white p-5 shadow-2xl"
          role="dialog"
          aria-modal="true"
          aria-labelledby="app-token-title"
        >
          <h2 id="app-token-title" className="text-base font-semibold text-gray-900">
            输入访问令牌
          </h2>
          <p className="mt-1 text-xs text-gray-500">
            后端已启用 API 访问保护。
          </p>
          <input
            autoFocus
            type="password"
            value={authToken}
            onChange={(event) => setAuthToken(event.target.value)}
            className="ui-control mt-4 w-full"
            placeholder="X-App-Token"
          />
          <label className="mt-3 flex items-center gap-2 text-sm text-gray-600">
            <input
              type="checkbox"
              checked={rememberToken}
              onChange={(event) => setRememberToken(event.target.checked)}
            />
            在此浏览器中长期记住
          </label>
          <div className="mt-5 flex items-center justify-between">
            <button
              type="button"
              className="text-sm text-gray-500 hover:text-red-600"
              onClick={() => {
                clearAppToken()
                setAuthToken('')
              }}
            >
              清除已保存令牌
            </button>
            <button type="submit" className="ui-button-primary" disabled={!authToken.trim()}>
              连接
            </button>
          </div>
        </form>
      </div>
    )}
    </>
  )
}
