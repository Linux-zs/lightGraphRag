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
  listChatSessions,
  listWorkspaces,
  WorkspaceInfo,
} from './api'

type Page = 'chat' | 'kb' | 'recall' | 'graph' | 'models' | 'dashboard'

export default function App() {
  const [page, setPage] = useState<Page>('chat')
  const [workspace, setWorkspace] = useState(() => localStorage.getItem('tdx_workspace') || 'tdx_default')
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([])
  const [chatSessions, setChatSessions] = useState<ChatSessionListItem[]>([])
  const [activeChatId, setActiveChatId] = useState<string | null>(null)

  const loadWorkspaces = async () => {
    try {
      const data = await listWorkspaces()
      setWorkspaces(data)
      if (!data.some((item) => item.workspace === workspace) && data[0]) {
        setWorkspace(data[0].workspace)
      }
    } catch {/* ignore */}
  }

  const loadChatSessions = async (targetWorkspace = workspace) => {
    try {
      const data = await listChatSessions(targetWorkspace)
      setChatSessions(data)
    } catch {/* ignore */}
  }

  useEffect(() => {
    loadWorkspaces()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    setActiveChatId(null)
    setChatSessions([])
    void loadChatSessions(workspace)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace])

  const handleWorkspaceChange = (next: string) => {
    if (next === workspace) return
    setActiveChatId(null)
    setWorkspace(next)
    localStorage.setItem('tdx_workspace', next)
  }

  const handleCreateWorkspace = async (name: string) => {
    const created = await createWorkspace(name)
    await loadWorkspaces()
    handleWorkspaceChange(created.workspace)
  }

  const handleNewChat = async () => {
    try {
      const created = await createChatSession(workspace)
      setChatSessions((prev) => [created, ...prev])
      setActiveChatId(created.id)
      setPage('chat')
    } catch {/* ignore */}
  }

  const handleSelectChat = (id: string) => {
    setActiveChatId(id)
    setPage('chat')
  }

  const handleDeleteChat = async (id: string) => {
    try {
      await deleteChatSession(id, workspace)
      setChatSessions((prev) => prev.filter((item) => item.id !== id))
      if (activeChatId === id) {
        setActiveChatId(null)
      }
    } catch {/* ignore */}
  }

  return (
    <Layout
      currentPage={page}
      onNavigate={setPage}
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
          setActiveId={setActiveChatId}
          reloadSessions={loadChatSessions}
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
