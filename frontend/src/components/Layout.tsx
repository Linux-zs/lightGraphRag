import { ReactNode, useMemo, useState } from 'react'
import { ChatSessionListItem, WorkspaceInfo } from '../api'

type Page = 'chat' | 'kb' | 'recall' | 'graph' | 'models' | 'dashboard'
type IconName =
  | 'chat'
  | 'kb'
  | 'recall'
  | 'graph'
  | 'models'
  | 'dashboard'
  | 'plus'
  | 'search'
  | 'trash'
  | 'folder'
  | 'chevron'

interface Props {
  currentPage: Page
  onNavigate: (page: Page) => void
  workspace: string
  workspaces: WorkspaceInfo[]
  onWorkspaceChange: (workspace: string) => void
  onCreateWorkspace: () => void
  chatSessions: ChatSessionListItem[]
  activeChatId: string | null
  onNewChat: () => void
  onSelectChat: (id: string) => void
  onDeleteChat: (id: string) => void
  children: ReactNode
}

const NAV_ITEMS: { key: Page; label: string; icon: IconName }[] = [
  { key: 'kb', label: '知识库管理', icon: 'kb' },
  { key: 'recall', label: '上下文预览', icon: 'recall' },
  { key: 'graph', label: '知识图谱', icon: 'graph' },
  { key: 'models', label: '模型设置', icon: 'models' },
  { key: 'dashboard', label: '系统状态', icon: 'dashboard' },
]

function Icon({ name, className = 'h-4 w-4' }: { name: IconName; className?: string }) {
  const common = {
    className,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.9,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true,
  }

  if (name === 'chat') {
    return (
      <svg {...common}>
        <path d="M5.5 18.2c-1.6-1.3-2.5-3.1-2.5-5.2 0-4.2 3.8-7.5 9-7.5s9 3.3 9 7.5-3.8 7.5-9 7.5c-1 0-2-.1-2.9-.4L4 21l1.5-2.8Z" />
        <path d="M8.5 12h7" />
        <path d="M8.5 15h4.5" />
      </svg>
    )
  }
  if (name === 'kb') {
    return (
      <svg {...common}>
        <path d="M4 6.5A2.5 2.5 0 0 1 6.5 4H19a1 1 0 0 1 1 1v14.5a.5.5 0 0 1-.7.46c-1.45-.63-3.2-.63-5.3 0-2.1.63-3.85.63-5.3 0A4.5 4.5 0 0 0 4 19.5v-13Z" />
        <path d="M4 7c1.45-.63 3.2-.63 5.3 0 1.45.43 2.7.55 3.7.35" />
        <path d="M8 10.5h8" />
        <path d="M8 14h6" />
      </svg>
    )
  }
  if (name === 'recall') {
    return (
      <svg {...common}>
        <circle cx="10.8" cy="10.8" r="6.2" />
        <path d="m15.5 15.5 4 4" />
        <path d="M8.4 10.7h4.8" />
        <path d="M10.8 8.3v4.8" />
      </svg>
    )
  }
  if (name === 'graph') {
    return (
      <svg {...common}>
        <circle cx="6" cy="7" r="2.4" />
        <circle cx="18" cy="8" r="2.4" />
        <circle cx="8" cy="18" r="2.4" />
        <circle cx="17" cy="17" r="2.4" />
        <path d="m8.3 7.2 7.3.6" />
        <path d="m6.5 9.3 1.1 6.4" />
        <path d="m10.2 17.8 4.6-.5" />
        <path d="m17.7 10.4-.5 4.2" />
      </svg>
    )
  }
  if (name === 'models') {
    return (
      <svg {...common}>
        <path d="M4 7h9" />
        <path d="M17 7h3" />
        <circle cx="15" cy="7" r="2" />
        <path d="M4 17h3" />
        <path d="M11 17h9" />
        <circle cx="9" cy="17" r="2" />
      </svg>
    )
  }
  if (name === 'dashboard') {
    return (
      <svg {...common}>
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <rect x="7" y="11" width="3" height="5" rx="1" />
        <rect x="12" y="7" width="3" height="9" rx="1" />
        <rect x="17" y="9" width="3" height="7" rx="1" />
      </svg>
    )
  }
  if (name === 'plus') {
    return (
      <svg {...common}>
        <path d="M12 5v14" />
        <path d="M5 12h14" />
      </svg>
    )
  }
  if (name === 'search') {
    return (
      <svg {...common}>
        <circle cx="10.8" cy="10.8" r="6.5" />
        <path d="m15.7 15.7 3.6 3.6" />
      </svg>
    )
  }
  if (name === 'trash') {
    return (
      <svg {...common}>
        <path d="M5 7h14" />
        <path d="M9 7V5.8A1.8 1.8 0 0 1 10.8 4h2.4A1.8 1.8 0 0 1 15 5.8V7" />
        <path d="M8 10v8" />
        <path d="M12 10v8" />
        <path d="M16 10v8" />
        <path d="M7 7l.7 13h8.6L17 7" />
      </svg>
    )
  }
  if (name === 'folder') {
    return (
      <svg {...common}>
        <path d="M3.5 7.5A2.5 2.5 0 0 1 6 5h4l2 2h6A2.5 2.5 0 0 1 20.5 9.5v7A2.5 2.5 0 0 1 18 19H6a2.5 2.5 0 0 1-2.5-2.5v-9Z" />
      </svg>
    )
  }
  return (
    <svg {...common}>
      <path d="m7 10 5 5 5-5" />
    </svg>
  )
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
  chatSessions,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  children,
}: Props) {
  const [chatSearch, setChatSearch] = useState('')

  const filteredSessions = useMemo(() => {
    const keyword = chatSearch.trim().toLowerCase()
    if (!keyword) return chatSessions
    return chatSessions.filter((item) => item.title.toLowerCase().includes(keyword))
  }, [chatSearch, chatSessions])

  return (
    <div className="flex h-screen bg-white">
      <aside className="w-64 bg-[#f7f7f8] border-r border-gray-200 flex flex-col shrink-0">
        <div className="px-3 py-3">
          <div className="flex items-center gap-2 px-2 py-2">
            <BrandMark />
            <div className="min-w-0">
              <h1 className="truncate text-sm font-semibold text-gray-900 tracking-tight">
                LightGraphRAG
              </h1>
              <p className="truncate text-[11px] text-gray-500">LightRAG 知识库工作台</p>
            </div>
          </div>

          <div className="mt-3 space-y-1">
            <button
              onClick={onNewChat}
              className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-gray-800 hover:bg-gray-200/70 transition-colors"
            >
              <Icon name="plus" className="h-4 w-4 text-gray-700" />
              <span>新建对话</span>
            </button>
            <div className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-gray-500 bg-white/60 border border-gray-200/80">
              <Icon name="search" className="h-4 w-4 text-gray-500" />
              <input
                value={chatSearch}
                onChange={(e) => setChatSearch(e.target.value)}
                placeholder="搜索对话"
                className="min-w-0 flex-1 bg-transparent outline-none placeholder:text-gray-500"
              />
            </div>
          </div>
        </div>

        <nav className="px-2 pb-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const active = currentPage === item.key
            return (
              <button
                key={item.key}
                onClick={() => onNavigate(item.key)}
                className={`w-full rounded-lg px-3 py-2 text-left text-sm flex items-center gap-3 transition-colors ${
                  active
                    ? 'bg-gray-200 text-gray-950'
                    : 'text-gray-700 hover:bg-gray-200/70'
                }`}
              >
                <Icon name={item.icon} className={active ? 'h-4 w-4 text-gray-950' : 'h-4 w-4 text-gray-600'} />
                <span className="truncate">{item.label}</span>
              </button>
            )
          })}
        </nav>

        <div className="px-3 pb-3">
          <div className="flex items-center justify-between px-2 pb-1 text-xs text-gray-500">
            <span>知识库</span>
            <Icon name="chevron" className="h-3.5 w-3.5" />
          </div>
          <div className="rounded-lg border border-gray-200 bg-white p-2">
            <div className="flex items-center gap-2">
              <Icon name="folder" className="h-4 w-4 shrink-0 text-gray-600" />
              <select
                value={workspace}
                onChange={(e) => onWorkspaceChange(e.target.value)}
                className="min-w-0 flex-1 bg-transparent text-xs text-gray-800 outline-none"
              >
                {workspaces.map((item) => (
                  <option key={item.workspace} value={item.workspace}>
                    {item.workspace}{item.is_default ? ' *' : ''}
                  </option>
                ))}
                {workspaces.length === 0 && <option value={workspace}>{workspace}</option>}
              </select>
              <button
                onClick={onCreateWorkspace}
                className="grid h-7 w-7 place-items-center rounded-md text-gray-500 hover:bg-gray-100 hover:text-gray-900"
                title="新建知识库"
              >
                <Icon name="plus" className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </div>

        <div className="min-h-0 flex-1 border-t border-gray-200 px-2 py-3">
          <div className="flex items-center justify-between px-2 text-xs text-gray-500">
            <span>所有对话</span>
            <Icon name="chevron" className="h-3.5 w-3.5" />
          </div>
          <div className="mt-2 h-[calc(100%-1.5rem)] overflow-y-auto pr-1 space-y-1">
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
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        onSelectChat(session.id)
                      }
                    }}
                    className={`group flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left transition-colors ${
                      active
                        ? 'bg-gray-200 text-gray-950'
                        : 'text-gray-700 hover:bg-gray-200/70'
                    }`}
                  >
                    <span className="min-w-0 flex-1 truncate text-sm leading-5">
                      {session.title || '未命名对话'}
                    </span>
                    <span className="shrink-0 text-[10px] text-gray-400 opacity-0 group-hover:opacity-100">
                      {session.message_count}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onDeleteChat(session.id)
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          e.stopPropagation()
                          onDeleteChat(session.id)
                        }
                      }}
                      className="grid h-6 w-6 shrink-0 place-items-center rounded-md text-gray-400 opacity-0 hover:bg-white hover:text-red-500 group-hover:opacity-100"
                      title="删除对话"
                    >
                      <Icon name="trash" className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )
              })
            )}
          </div>
        </div>

        <div className="px-5 py-3 text-[11px] text-gray-400">
          v2.0.0
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        {currentPage === 'chat' ? (
          children
        ) : (
          <div className="max-w-5xl mx-auto px-8 py-6">
            {children}
          </div>
        )}
      </main>
    </div>
  )
}
