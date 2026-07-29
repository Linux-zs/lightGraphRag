import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, ChevronDown, Search, X } from 'lucide-react'
import type { GraphNode } from '../api'

interface Props {
  nodes: GraphNode[]
  value: string[]
  onChange: (value: string[]) => void
  placeholder: string
  multiple?: boolean
  exclude?: string[]
}

export default function EntityPicker({
  nodes,
  value,
  onChange,
  placeholder,
  multiple = false,
  exclude = [],
}: Props) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)

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

  const nodeMap = useMemo(
    () => new Map(nodes.map((node) => [node.id, node])),
    [nodes],
  )
  const excluded = useMemo(() => new Set(exclude), [exclude])
  const filtered = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    return nodes
      .filter((node) => !excluded.has(node.id))
      .filter((node) => !keyword || [node.id, node.label, node.entity_type]
        .join(' ')
        .toLowerCase()
        .includes(keyword))
      .slice(0, 120)
  }, [excluded, nodes, query])

  const toggle = (id: string) => {
    if (!multiple) {
      onChange([id])
      setOpen(false)
      setQuery('')
      return
    }
    onChange(value.includes(id) ? value.filter((item) => item !== id) : [...value, id])
  }

  return (
    <div ref={rootRef} className="relative min-w-0">
      <button
        type="button"
        onClick={() => setOpen((visible) => !visible)}
        className="flex min-h-10 w-full items-center gap-2 rounded-md border border-gray-300 bg-white px-2.5 py-1.5 text-left text-sm transition hover:border-gray-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-200"
        aria-expanded={open}
      >
        <span className="flex min-w-0 flex-1 flex-wrap gap-1">
          {value.length === 0 ? (
            <span className="py-0.5 text-gray-400">{placeholder}</span>
          ) : (
            value.map((id) => (
              <span
                key={id}
                className="inline-flex max-w-full items-center gap-1 rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-700"
              >
                <span className="truncate">{nodeMap.get(id)?.label || id}</span>
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(event) => {
                    event.stopPropagation()
                    onChange(value.filter((item) => item !== id))
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      event.stopPropagation()
                      onChange(value.filter((item) => item !== id))
                    }
                  }}
                  aria-label={`移除实体 ${id}`}
                  className="grid h-4 w-4 place-items-center rounded text-gray-400 hover:bg-gray-200 hover:text-gray-700"
                >
                  <X size={11} />
                </span>
              </span>
            ))
          )}
        </span>
        <ChevronDown size={14} className={`shrink-0 text-gray-400 transition ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute left-0 right-0 top-full z-40 mt-1 overflow-hidden rounded-lg border border-gray-200 bg-white shadow-[var(--ui-shadow)]">
          <div className="border-b border-gray-100 p-2">
            <label className="flex h-8 items-center gap-2 rounded-md border border-gray-200 bg-gray-50 px-2 focus-within:border-primary-400 focus-within:bg-white">
              <Search size={14} className="text-gray-400" />
              <input
                autoFocus
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="搜索实体名称或类型"
                className="min-w-0 flex-1 bg-transparent text-xs outline-none"
              />
            </label>
          </div>
          <div className="max-h-60 overflow-y-auto p-1.5">
            {filtered.map((node) => {
              const selected = value.includes(node.id)
              return (
                <button
                  type="button"
                  key={node.id}
                  onClick={() => toggle(node.id)}
                  className={`flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left ${
                    selected ? 'bg-primary-50' : 'hover:bg-gray-50'
                  }`}
                >
                  <span className={`grid h-4 w-4 shrink-0 place-items-center rounded border ${
                    selected
                      ? 'border-primary-600 bg-primary-600 text-white'
                      : 'border-gray-300 bg-white text-transparent'
                  }`}>
                    <Check size={11} strokeWidth={2.5} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm text-gray-800">{node.label || node.id}</span>
                    <span className="block truncate text-[10px] text-gray-400">
                      {node.entity_type || 'entity'} · {node.id}
                    </span>
                  </span>
                </button>
              )
            })}
            {filtered.length === 0 && (
              <div className="px-3 py-5 text-center text-xs text-gray-400">没有匹配实体</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
