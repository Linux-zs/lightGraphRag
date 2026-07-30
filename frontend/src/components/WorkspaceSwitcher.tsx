import { useEffect, useId, useMemo, useRef, useState } from 'react'
import {
  Check,
  ChevronDown,
  Database,
  Plus,
  Search,
  Settings2,
  Trash2,
} from 'lucide-react'
import type { WorkspaceInfo } from '../api'

interface Props {
  workspace: string
  workspaces: WorkspaceInfo[]
  onChange: (workspace: string) => void
  onManage?: () => void
  onCreate?: () => void
  onDelete?: (workspace: string) => Promise<void> | void
  placement?: 'bottom' | 'top'
  compact?: boolean
  className?: string
}

export default function WorkspaceSwitcher({
  workspace,
  workspaces,
  onChange,
  onManage,
  onCreate,
  onDelete,
  placement = 'bottom',
  compact = false,
  className = '',
}: Props) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [actionError, setActionError] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)
  const menuId = useId()

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false)
    }
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [])

  const filtered = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    if (!keyword) return workspaces
    return workspaces.filter((item) => item.workspace.toLowerCase().includes(keyword))
  }, [query, workspaces])
  const currentWorkspace = workspaces.find((item) => item.workspace === workspace)
  const canDeleteCurrent = Boolean(onDelete && currentWorkspace && !currentWorkspace.is_default)

  const selectWorkspace = (next: string) => {
    onChange(next)
    setOpen(false)
    setQuery('')
    setActionError('')
  }

  return (
    <div ref={rootRef} className={`relative min-w-0 ${className}`}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className={`group flex w-full min-w-0 items-center border border-gray-200 bg-white text-left text-gray-800 shadow-sm transition hover:border-gray-300 hover:bg-gray-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-200 ${
          compact ? 'h-8 gap-1.5 rounded-md px-2 text-xs' : 'h-10 gap-2.5 rounded-lg px-3 text-sm'
        }`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={menuId}
        aria-label={`当前知识库 ${workspace}`}
      >
        <span className={`grid shrink-0 place-items-center rounded-md bg-primary-50 text-primary-700 ${
          compact ? 'h-5 w-5' : 'h-7 w-7'
        }`}>
          <Database size={compact ? 13 : 15} strokeWidth={1.9} />
        </span>
        <span className="min-w-0 flex-1">
          {!compact && <span className="block text-[10px] leading-3 text-gray-400">当前知识库</span>}
          <span className="block truncate font-medium">{workspace}</span>
        </span>
        <ChevronDown
          size={14}
          className={`shrink-0 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div
          id={menuId}
          className={`absolute left-0 z-50 w-[280px] overflow-hidden rounded-lg border border-gray-200 bg-white shadow-[var(--ui-shadow)] ${
            placement === 'top' ? 'bottom-full mb-2' : 'top-full mt-2'
          }`}
        >
          <div className="border-b border-gray-100 p-2">
            <label className="flex h-8 items-center gap-2 rounded-md border border-gray-200 bg-gray-50 px-2 text-gray-400 focus-within:border-primary-400 focus-within:bg-white focus-within:ring-2 focus-within:ring-primary-100">
              <Search size={14} />
              <input
                autoFocus
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="搜索知识库"
                className="min-w-0 flex-1 bg-transparent text-xs text-gray-800 outline-none"
              />
            </label>
          </div>

          <div className="max-h-56 overflow-y-auto p-1.5" role="listbox">
            {filtered.map((item) => {
              const selected = item.workspace === workspace
              return (
                <button
                  type="button"
                  key={item.workspace}
                  onClick={() => selectWorkspace(item.workspace)}
                  className={`flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm ${
                    selected ? 'bg-primary-50 text-primary-800' : 'text-gray-700 hover:bg-gray-50'
                  }`}
                  role="option"
                  aria-selected={selected}
                >
                  <Database size={14} className="shrink-0" />
                  <span className="min-w-0 flex-1 truncate">{item.workspace}</span>
                  <span className="shrink-0 text-[10px] text-gray-400">{item.doc_count} 文档</span>
                  {selected && <Check size={14} className="shrink-0 text-primary-600" />}
                </button>
              )
            })}
            {filtered.length === 0 && (
              <div className="px-3 py-5 text-center text-xs text-gray-400">没有匹配的知识库</div>
            )}
          </div>

          {(onManage || onCreate || onDelete) && (
            <div className="flex flex-wrap gap-1 border-t border-gray-100 p-1.5">
              {onManage && (
                <button
                  type="button"
                  onClick={() => {
                    setOpen(false)
                    onManage()
                  }}
                  className="flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md px-2 text-xs text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                >
                  <Settings2 size={14} />
                  管理知识库
                </button>
              )}
              {onCreate && (
                <button
                  type="button"
                  onClick={() => {
                    setOpen(false)
                    onCreate()
                  }}
                  className="flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md px-2 text-xs text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                >
                  <Plus size={14} />
                  新建知识库
                </button>
              )}
              {onDelete && (
                <button
                  type="button"
                  onClick={async () => {
                    if (!canDeleteCurrent) return
                    setActionError('')
                    try {
                      await onDelete(workspace)
                      setOpen(false)
                    } catch (error) {
                      setActionError((error as Error).message || '删除知识库失败')
                    }
                  }}
                  disabled={!canDeleteCurrent}
                  title={currentWorkspace?.is_default ? '默认知识库不能删除' : '删除当前知识库'}
                  className="flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md px-2 text-xs text-red-600 hover:bg-red-50 disabled:cursor-not-allowed disabled:text-gray-300 disabled:hover:bg-transparent"
                >
                  <Trash2 size={14} />
                  删除
                </button>
              )}
              {actionError && (
                <div className="basis-full rounded-md bg-red-50 px-2 py-1.5 text-xs text-red-600">
                  {actionError}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
