import { useEffect, useState } from 'react'
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
  WorkspaceInfo,
} from './api'
import { useConfirm } from './components/ConfirmDialog'

type Page = 'chat' | 'kb' | 'recall' | 'graph' | 'models' | 'dashboard'
const PAGES: Page[] = ['chat', 'kb', 'recall', 'graph', 'models', 'dashboard']
const DEFAULT_PAGE: Page = 'chat'

function isPage(value: string | null): value is Page {
  return !!value && PAGES.includes(value as Page)
}

function pageFromHash(): Page | null {
  const raw = window.location.hash.replace(/^#\/?/, '').split('/')[0]
  return isPage(raw) ? raw : null
}

function initialPage(): Page {
  return pageFromHash() || (isPage(localStorage.getItem('tdx_page')) ? localStorage.getItem('tdx_page') as Page : DEFAULT_PAGE)
}

function pageHash(page: Page) {
  return `#/${page}`
}

function chatStorageKey(workspace: string) {
  return `tdx_active_chat_${workspace}`
}

export default function App() {
  const confirm = useConfirm()
  const [page, setPage] = useState<Page>(() => initialPage())
  const [workspace, setWorkspace] = useState(() => localStorage.getItem('tdx_workspace') || 'tdx_default')
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([])
  const [chatSessions, setChatSessions] = useState<ChatSessionListItem[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(
    () => localStorage.getItem(chatStorageKey(localStorage.getItem('tdx_workspace') || 'tdx_default')),
  )

  const loadWorkspaces = async (current = workspace) => {
    try {
      const data = await listWorkspaces()
      setWorkspaces(data)
      if (!data.some((item) => item.workspace === current) && data[0]) {
        setWorkspace(data[0].workspace)
        localStorage.setItem('tdx_workspace', data[0].workspace)
      }
      return data
    } catch {/* ignore */}
    return []
  }

  const loadChatSessions = async (targetWorkspace = workspace) => {
    try {
      const data = await listChatSessions(targetWorkspace)
      setChatSessions(data)
      return data
    } catch {/* ignore */}
    return []
  }

  const reloadCurrentChatSessions = async () => {
    await loadChatSessions(workspace)
  }

  useEffect(() => {
    loadWorkspaces()
    if (!pageFromHash()) {
      window.history.replaceState(null, '', pageHash(page))
    }
    const handleHashChange = () => {
      const next = pageFromHash()
      if (!next) return
      setPage(next)
      localStorage.setItem('tdx_page', next)
    }
    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const storedActive = localStorage.getItem(chatStorageKey(workspace))
    setActiveChatId(storedActive)
    setChatSessions([])
    void loadChatSessions(workspace).then((sessions) => {
      if (storedActive && !sessions.some((session) => session.id === storedActive)) {
        setActiveChatId(null)
        localStorage.removeItem(chatStorageKey(workspace))
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace])

  const handleNavigate = (next: Page) => {
    setPage(next)
    localStorage.setItem('tdx_page', next)
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
    localStorage.setItem('tdx_workspace', next)
  }

  const handleCreateWorkspace = async (name: string) => {
    const created = await createWorkspace(name)
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
        const next = nextWorkspaces[0]?.workspace || 'tdx_default'
        setWorkspace(next)
        localStorage.setItem('tdx_workspace', next)
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

  return (
    <Layout
      currentPage={page}
      onNavigate={handleNavigate}
      workspace={workspace}
      workspaces={workspaces}
      onWorkspaceChange={handleWorkspaceChange}
      onCreateWorkspace={handleCreateWorkspace}
      onDeleteWorkspace={handleDeleteWorkspace}
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
      {page === 'kb' && <KBManagement workspace={workspace} />}
      {page === 'recall' && <RecallTest workspace={workspace} />}
      {page === 'graph' && <GraphPage workspace={workspace} />}
      {page === 'models' && <ModelSettings workspace={workspace} />}
      {page === 'dashboard' && <Dashboard workspace={workspace} onWorkspaceChanged={loadWorkspaces} />}
    </Layout>
  )
}
