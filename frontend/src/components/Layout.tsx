import { FormEvent, ReactNode, useEffect, useMemo, useState } from 'react'
import {
  BarChart3,
  ChevronDown,
  Database,
  FileSearch,
  GitBranch,
  MessageSquarePlus,
  Search,
  SlidersHorizontal,
  Trash2,
  X,
} from 'lucide-react'
import { ChatSessionListItem, WorkspaceInfo } from '../api'
import WorkspaceSwitcher from './WorkspaceSwitcher'

type Page = 'chat' | 'kb' | 'recall' | 'graph' | 'models' | 'dashboard'

interface Props {
  currentPage: Page
  onNavigate: (page: Page) => void
  workspace: string
  workspaces: WorkspaceInfo[]
  onWorkspaceChange: (workspace: string) => void
  onCreateWorkspace: (name: string) => Promise<void>
  onDeleteWorkspace: (workspace: string) => void
  chatSessions: ChatSessionListItem[]
  activeChatId: string | null
  onNewChat: () => void
  onSelectChat: (id: string) => void
  onDeleteChat: (id: string) => void
  children: ReactNode
}

const NAV_ITEMS = [
  { key: 'recall' as const, label: '召回调试', icon: FileSearch },
  { key: 'graph' as const, label: '知识图谱', icon: GitBranch },
  { key: 'dashboard' as const, label: '系统状态', icon: BarChart3 },
  { key: 'models' as const, label: '模型设置', icon: SlidersHorizontal },
]

const PAGE_LABELS: Record<Page, string> = {
  chat: '知识库问答',
  kb: '知识库管理',
  recall: '召回调试',
  graph: '知识图谱',
  models: '模型设置',
  dashboard: '系统状态',
}

function BrandMark() {
  return (
    <img
      src="/favicon.ico"
      alt=""
      className="h-7 w-7 rounded-md object-contain"
      aria-hidden="true"
    />
  )
}

export default function Layout({
  currentPage,
  onNavigate,
  workspace,
  workspaces,
  onWorkspaceChange,
  onCreateWorkspace,
  onDeleteWorkspace,
  chatSessions,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  children,
}: Props) {
  const [chatSearch, setChatSearch] = useState('')
  const [sessionsOpen, setSessionsOpen] = useState(
    () => localStorage.getItem('ui_sessions_open') !== 'collapsed',
  )
  const [createOpen, setCreateOpen] = useState(false)
  const [workspaceName, setWorkspaceName] = useState('')
  const [createError, setCreateError] = useState('')
  const [creating, setCreating] = useState(false)

  const setSection = (
    setter: (value: boolean) => void,
    key: string,
    value: boolean,
  ) => {
    setter(value)
    localStorage.setItem(key, value ? 'expanded' : 'collapsed')
  }

  const filteredSessions = useMemo(() => {
    const keyword = chatSearch.trim().toLowerCase()
    if (!keyword) return chatSessions
    return chatSessions.filter((item) => item.title.toLowerCase().includes(keyword))
  }, [chatSearch, chatSessions])

  useEffect(() => {
    if (!createOpen) return
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !creating) setCreateOpen(false)
    }
    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [createOpen, creating])

  const openCreateDialog = () => {
    setWorkspaceName('')
    setCreateError('')
    setCreateOpen(true)
  }

  const submitWorkspace = async (event: FormEvent) => {
    event.preventDefault()
    const name = workspaceName.trim()
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) {
      setCreateError('名称需为 1-64 位字母、数字、下划线或短横线')
      return
    }
    if (workspaces.some((item) => item.workspace === name)) {
      setCreateError('该知识库已经存在')
      return
    }
    setCreating(true)
    setCreateError('')
    try {
      await onCreateWorkspace(name)
      setCreateOpen(false)
    } catch (error) {
      setCreateError((error as Error).message || '创建知识库失败')
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="flex h-screen bg-white">
      <aside className="flex w-[248px] shrink-0 flex-col border-r border-gray-200 bg-[#f7f7f8]">
        <div className="px-3 pb-2 pt-3">
          <div className="flex items-center gap-2 px-2 py-1.5">
            <BrandMark />
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold text-gray-900">
                LightGraphRAG
              </h1>
              <p className="truncate text-[11px] text-gray-500">LightRAG 知识库工作台</p>
            </div>
          </div>
        </div>

        <div className="px-3 pb-3">
          <WorkspaceSwitcher
            workspace={workspace}
            workspaces={workspaces}
            onChange={onWorkspaceChange}
            onManage={() => onNavigate('kb')}
            onCreate={openCreateDialog}
            onDelete={onDeleteWorkspace}
          />
        </div>

        <div className="px-4 pb-1 text-[10px] font-medium uppercase text-gray-400">
          知识库工具
        </div>
        <nav className="space-y-0.5 px-2 pb-2">
          {NAV_ITEMS.map((item) => {
            const active = currentPage === item.key
            const NavIcon = item.icon
            return (
              <div key={item.key}>
                {item.key === 'models' && (
                  <div className="mx-2 my-2 border-t border-gray-200 pt-2 text-[10px] font-medium uppercase text-gray-400">
                    系统
                  </div>
                )}
                <button
                  onClick={() => onNavigate(item.key)}
                  className={`flex h-9 w-full items-center gap-2.5 rounded-md px-3 text-left text-sm transition-colors ${
                    active
                      ? 'bg-gray-200 text-gray-950'
                      : 'text-gray-700 hover:bg-gray-200/70'
                  }`}
                >
                  <NavIcon size={16} strokeWidth={1.8} aria-hidden="true" />
                  <span className="truncate">{item.label}</span>
                </button>
              </div>
            )
          })}
        </nav>

        <div className="flex min-h-0 flex-1 flex-col border-t border-gray-200 px-2 py-2">
          <button
            onClick={() => setSection(setSessionsOpen, 'ui_sessions_open', !sessionsOpen)}
            className="flex h-8 items-center gap-2 rounded-md px-2 text-xs text-gray-500 hover:bg-gray-200/70 hover:text-gray-800"
            aria-expanded={sessionsOpen}
          >
            <span>所有对话</span>
            <span className="text-[10px] tabular-nums text-gray-400">{chatSessions.length}</span>
            <ChevronDown
              size={14}
              className={`ml-auto transition-transform ${sessionsOpen ? '' : '-rotate-90'}`}
            />
          </button>

          {sessionsOpen && (
            <>
              <button
                onClick={onNewChat}
                className="mt-1 flex h-9 w-full items-center gap-2.5 rounded-md px-2 text-sm text-gray-800 hover:bg-gray-200/70"
              >
                <MessageSquarePlus size={16} strokeWidth={1.8} />
                <span>新建对话</span>
              </button>
              <label className="mt-1 flex h-9 items-center gap-2 rounded-md border border-gray-200 bg-white px-2.5 text-gray-500 focus-within:border-primary-400 focus-within:ring-2 focus-within:ring-primary-100">
                <Search size={15} strokeWidth={1.8} />
                <input
                  value={chatSearch}
                  onChange={(event) => setChatSearch(event.target.value)}
                  placeholder="搜索对话"
                  className="min-w-0 flex-1 bg-transparent text-sm text-gray-800 outline-none placeholder:text-gray-400"
                />
              </label>

              <div className="mt-1 min-h-0 flex-1 space-y-0.5 overflow-y-auto pr-1">
                {filteredSessions.length === 0 ? (
                  <p className="px-2 py-4 text-xs text-gray-400">
                    {chatSearch.trim() ? '没有匹配的对话' : '暂无对话记录'}
                  </p>
                ) : (
                  filteredSessions.map((session) => {
                    const active = activeChatId === session.id && currentPage === 'chat'
                    return (
                      <div
                        key={session.id}
                        role="button"
                        tabIndex={0}
                        onClick={() => onSelectChat(session.id)}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault()
                            onSelectChat(session.id)
                          }
                        }}
                        className={`group flex h-9 w-full items-center gap-2 rounded-md px-2 text-left transition-colors ${
                          active
                            ? 'bg-gray-200 text-gray-950'
                            : 'text-gray-700 hover:bg-gray-200/70'
                        }`}
                      >
                        <span className="min-w-0 flex-1 truncate text-sm">
                          {session.title || '新对话'}
                        </span>
                        <button
                          onClick={(event) => {
                            event.stopPropagation()
                            onDeleteChat(session.id)
                          }}
                          className="grid h-6 w-6 shrink-0 place-items-center rounded text-gray-400 opacity-0 hover:bg-white hover:text-red-500 focus:opacity-100 group-hover:opacity-100"
                          title="删除对话"
                          aria-label={`删除对话 ${session.title}`}
                        >
                          <Trash2 size={13} strokeWidth={1.8} />
                        </button>
                      </div>
                    )
                  })
                )}
              </div>
            </>
          )}
        </div>

        <div className="px-5 py-2 text-[11px] text-gray-400">v2.1.0</div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {currentPage !== 'models' && (
          <div className="flex h-11 shrink-0 items-center gap-2 border-b border-gray-200 bg-white px-6 text-xs">
            <span className="flex items-center gap-1.5 font-medium text-gray-500">
              <Database size={14} className="text-primary-600" />
              知识库工作区
            </span>
            <span className="text-gray-300">/</span>
            <span className="max-w-56 truncate font-semibold text-gray-900">{workspace}</span>
            <span className="ml-auto rounded-md bg-gray-100 px-2 py-1 text-[11px] text-gray-500">
              {PAGE_LABELS[currentPage]}
            </span>
          </div>
        )}
        <div className={`min-h-0 flex-1 ${currentPage === 'chat' ? '' : 'overflow-auto'}`}>
          {currentPage === 'chat' ? (
            children
          ) : (
            <div className="mx-auto max-w-5xl px-8 py-6">
              {children}
            </div>
          )}
        </div>
      </main>

      {createOpen && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-gray-950/35 p-4 backdrop-blur-[1px]"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !creating) setCreateOpen(false)
          }}
        >
          <form
            onSubmit={submitWorkspace}
            className="w-full max-w-md rounded-lg border border-gray-200 bg-white shadow-2xl"
            role="dialog"
            aria-modal="true"
            aria-labelledby="create-workspace-title"
          >
            <div className="flex items-start justify-between border-b border-gray-100 px-5 py-4">
              <div>
                <h2 id="create-workspace-title" className="text-base font-semibold text-gray-900">
                  新建知识库
                </h2>
                <p className="mt-1 text-xs text-gray-500">创建独立的文档、索引和知识图谱工作区。</p>
              </div>
              <button
                type="button"
                onClick={() => setCreateOpen(false)}
                className="ui-icon-button"
                disabled={creating}
                aria-label="关闭"
              >
                <X size={16} />
              </button>
            </div>
            <div className="px-5 py-5">
              <label className="block">
                <span className="ui-label">知识库名称</span>
                <input
                  autoFocus
                  value={workspaceName}
                  onChange={(event) => {
                    setWorkspaceName(event.target.value)
                    setCreateError('')
                  }}
                  className="ui-control w-full"
                  placeholder="例如 product_docs"
                  maxLength={64}
                />
              </label>
              <p className="mt-2 text-xs text-gray-400">支持字母、数字、下划线和短横线。</p>
              {createError && (
                <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-xs text-red-600">
                  {createError}
                </p>
              )}
            </div>
            <div className="flex justify-end gap-2 border-t border-gray-100 px-5 py-3">
              <button
                type="button"
                onClick={() => setCreateOpen(false)}
                className="ui-button-secondary"
                disabled={creating}
              >
                取消
              </button>
              <button type="submit" className="ui-button-primary" disabled={creating}>
                {creating ? '创建中...' : '创建知识库'}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  )
}
