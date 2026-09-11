import { useEffect, useRef, useState, useMemo, useId } from 'react'
import type { GraphNode, GraphEdge } from '../api'
import { placeGraphLabels } from '../utils/graphLabels'
import { orientGraph } from '../utils/graphOrientation'
import { searchGraphNodes } from '../utils/graphSearch'

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  directed?: boolean
  /** Node IDs to highlight (hit by retrieval) */
  hitNodes?: Set<string>
  /** Actual retrieved relations; node co-occurrence alone is not evidence. */
  pathEdges?: Pick<GraphEdge, 'source' | 'target'>[]
  /** Search the full persisted graph when the overview is truncated. */
  searchAllNodes?: (query: string, signal: AbortSignal) => Promise<GraphNode[]>
  /** Load a focused subgraph when a search hit is outside the current view. */
  onOpenSearchResult?: (node: GraphNode) => Promise<void>
  focusNodeId?: string | null
  className?: string
}

interface PositionedNode extends GraphNode {
  x: number
  y: number
}

/** Category → color mapping */
const CATEGORY_COLORS: Record<string, string> = {
  '物品': '#0284c7', '组织': '#7c3aed', '概念': '#0d9488', '地点': '#d97706',
  '问题': '#e11d48', '事件': '#4f46e5', 'unknown': '#64748b', '其他': '#64748b',
  '数据源': '#3b82f6',
  '核心系统': '#ef4444',
  '传输层': '#f59e0b',
  '接入层': '#8b5cf6',
  '服务层': '#06b6d4',
  '终端': '#10b981',
  '安全': '#ec4899',
  '同步': '#84cc16',
  '内容': '#f97316',
  '传输': '#f59e0b',
  '监控': '#6366f1',
}
const DEFAULT_COLOR = '#6b7280'

export default function GraphView({
  nodes,
  edges,
  directed = false,
  hitNodes,
  pathEdges,
  searchAllNodes,
  onOpenSearchResult,
  focusNodeId,
  className,
}: Props) {
  const pathEdgeKeys = useMemo(() => new Set((pathEdges || []).map(edge =>
    JSON.stringify(directed ? [edge.source, edge.target] : [edge.source, edge.target].sort())
  )), [pathEdges, directed])
  const containerRef = useRef<HTMLDivElement>(null)
  const marker = useId().replace(/:/g, '')
  const [dims, setDims] = useState({ width: 600, height: 500 })
  const [hovered, setHovered] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [localOnly, setLocalOnly] = useState(false)
  const [allLabels, setAllLabels] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const [remoteSearch, setRemoteSearch] = useState<{
    query: string
    nodes: GraphNode[]
    loading: boolean
    error: string
  }>({ query: '', nodes: [], loading: false, error: '' })
  const [focusRequest, setFocusRequest] = useState<{ id: string } | null>(null)
  const [camera, setCamera] = useState({ x: 0, y: 0, zoom: 1 })
  const drag = useRef<{ x: number; y: number; cx: number; cy: number } | null>(null)
  useEffect(() => {
    if (!containerRef.current) return
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width && entry.contentRect.height)
        setDims({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])
  useEffect(() => {
    const requested = focusNodeId && nodes.some(node => node.id === focusNodeId)
      ? focusNodeId
      : null
    const requestedNode = requested ? nodes.find(node => node.id === requested) : null
    setSelected(requested); setHovered(null); setLocalOnly(false); setCamera({ x: 0, y: 0, zoom: 1 })
    setSearchQuery(requestedNode?.label || ''); setSearchOpen(false)
    setFocusRequest(requested ? { id: requested } : null)
  }, [nodes, focusNodeId])
  const initialPositions = useMemo(() => nodes.map((node, i) => ({
    ...node, x: 500 + 260 * Math.cos(2 * Math.PI * i / nodes.length),
    y: 375 + 260 * Math.sin(2 * Math.PI * i / nodes.length),
  })), [nodes])
  const [layout, setLayout] = useState<{ source: GraphNode[]; edges: GraphEdge[]; positions: PositionedNode[] } | null>(null)
  const [layoutStatus, setLayoutStatus] = useState('')
  useEffect(() => {
    if (!nodes.length) { setLayoutStatus(''); return }
    const fallback = '自动布局不可用，当前显示基础布局'
    if (typeof Worker === 'undefined') { setLayoutStatus(fallback); return }
    let worker: Worker
    try {
      worker = new Worker(new URL('../utils/graphLayout.worker.ts', import.meta.url), { type: 'module' })
    } catch { setLayoutStatus(fallback); return }
    setLayoutStatus('正在计算布局，可继续浏览')
    let active = true
    worker.onmessage = (event: MessageEvent<PositionedNode[]>) => {
      if (active) {
        setLayout({ source: nodes, edges, positions: event.data })
        setLayoutStatus('')
      }
      worker.terminate()
    }
    worker.onerror = () => { if (active) setLayoutStatus(fallback); worker.terminate() }
    try { worker.postMessage({ nodes, edges }) }
    catch { setLayoutStatus(fallback); worker.terminate() }
    return () => { active = false; worker.terminate() }
  }, [nodes, edges])
  const rawPositions = layout?.source === nodes && layout.edges === edges ? layout.positions : initialPositions
  // Selection/hover must not change orientation or invalidate the worker layout.
  const positioned = useMemo(() => orientGraph(rawPositions, Math.max(100, dims.width - 40),
    Math.max(100, dims.height - (dims.width < 520 ? 180 : 140))), [rawPositions, dims])
  const nodeMap = useMemo(() => new Map(positioned.map(n => [n.id, n])), [positioned])
  const degrees = useMemo(() => {
    const result = new Map<string, number>()
    edges.forEach(e => { result.set(e.source, (result.get(e.source) || 0) + 1); result.set(e.target, (result.get(e.target) || 0) + 1) })
    return result
  }, [edges])
  const localSearchResults = useMemo(
    () => searchGraphNodes(nodes, searchQuery, degrees),
    [nodes, searchQuery, degrees],
  )
  const normalizedSearchQuery = searchQuery.trim()
  const searchResults = useMemo(() => {
    if (!searchAllNodes || remoteSearch.query !== normalizedSearchQuery) {
      return localSearchResults
    }
    const combined = [...remoteSearch.nodes, ...localSearchResults]
    return combined.filter(
      (node, index) => combined.findIndex(candidate => candidate.id === node.id) === index,
    ).slice(0, 8)
  }, [localSearchResults, normalizedSearchQuery, remoteSearch, searchAllNodes])
  useEffect(() => {
    if (!searchAllNodes || !normalizedSearchQuery) {
      setRemoteSearch({ query: '', nodes: [], loading: false, error: '' })
      return
    }
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setRemoteSearch({ query: normalizedSearchQuery, nodes: [], loading: true, error: '' })
      searchAllNodes(normalizedSearchQuery, controller.signal)
        .then(result => {
          if (!controller.signal.aborted) {
            setRemoteSearch({ query: normalizedSearchQuery, nodes: result, loading: false, error: '' })
          }
        })
        .catch(error => {
          if (!controller.signal.aborted) {
            setRemoteSearch({
              query: normalizedSearchQuery,
              nodes: [],
              loading: false,
              error: (error as Error).message || '全库搜索失败',
            })
          }
        })
    }, 180)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [normalizedSearchQuery, searchAllNodes])
  const active = selected || hovered
  const neighbors = useMemo(() => {
    const result = new Set(active ? [active] : [])
    edges.forEach(e => { if (e.source === active) result.add(e.target); if (e.target === active) result.add(e.source) })
    return result
  }, [active, edges])
  const visible = localOnly && active ? positioned.filter(n => neighbors.has(n.id)) : positioned
  const visibleIds = new Set(visible.map(node => node.id))
  const visibleEdges = edges.filter(edge => visibleIds.has(edge.source) && visibleIds.has(edge.target))
  const bounds = useMemo(() => {
    if (!visible.length) return { x: 0, y: 0, width: 1000, height: 750 }
    const xs = visible.map(n => n.x), ys = visible.map(n => n.y)
    return { x: Math.min(...xs) - 85, y: Math.min(...ys) - 65,
      width: Math.max(...xs) - Math.min(...xs) + 170, height: Math.max(...ys) - Math.min(...ys) + 130 }
  }, [visible])
  // Hover must not change the camera fit: moving the target under the pointer
  // would repeatedly enter/leave it. Only a deliberate selection opens details.
  const detail = selected ? nodeMap.get(selected) : null
  const topInset = dims.width < 520 ? 160 : 70
  const bottomInset = detail ? 170 : 70
  const canvasHeight = Math.max(100, dims.height - topInset - bottomInset)
  const scale = Math.min(Math.max(100, dims.width - 40) / bounds.width, canvasHeight / bounds.height) * camera.zoom
  const tx = dims.width / 2 - (bounds.x + bounds.width / 2) * scale + camera.x
  const ty = topInset + canvasHeight / 2 - (bounds.y + bounds.height / 2) * scale + camera.y
  useEffect(() => {
    if (!focusRequest || localOnly) return
    const node = nodeMap.get(focusRequest.id)
    if (!node) { setFocusRequest(null); return }
    const baseScale = scale / camera.zoom
    setCamera({ zoom: 1,
      x: (bounds.x + bounds.width / 2 - node.x) * baseScale,
      y: (bounds.y + bounds.height / 2 - node.y) * baseScale })
    setFocusRequest(null)
  }, [focusRequest, localOnly, nodeMap, bounds, scale, camera.zoom])
  const focusSearchResult = (node: PositionedNode | GraphNode) => {
    if (!nodeMap.has(node.id) && onOpenSearchResult) {
      setSearchQuery(node.label); setSearchOpen(false)
      void onOpenSearchResult(node).catch(error => {
        setRemoteSearch(current => ({
          ...current,
          loading: false,
          error: (error as Error).message || '实体邻域加载失败',
        }))
      })
      return
    }
    setFocusRequest({ id: node.id })
    setSelected(node.id); setHovered(null); setLocalOnly(false)
    setSearchQuery(node.label); setSearchOpen(false)
  }
  const nodeRadius = (id: string) => 4.5 + Math.min(4, Math.sqrt(degrees.get(id) || 0) * 0.9)
  const nodeLabelKey = (id: string) => JSON.stringify(['node', id])
  const edgeLabelKey = (index: number) => JSON.stringify(['edge', index])
  const nodeLabelCandidates = visible.map(node => {
    const focused = node.id === active
    const label = node.label.length > 22 && !focused ? node.label.slice(0, 22) + '…' : node.label
    const radius = nodeRadius(node.id)
    const width = Array.from(label).reduce((sum, char) => sum + (/[^\x00-\x7f]/.test(char) ? 11 : 6.5), 0)
    return { id: nodeLabelKey(node.id), x: node.x * scale + tx, y: node.y * scale + ty + radius + 15,
      width, alternatives: [
        { dx: 0, dy: -(2 * radius + 22) },
        { dx: width / 2 + radius + 8, dy: -(radius + 11) },
        { dx: -(width / 2 + radius + 8), dy: -(radius + 11) },
      ],
      height: 13, force: focused, ownerId: node.id,
      priority: (hitNodes?.has(node.id) ? 2000000 : neighbors.has(node.id) ? 1000000 : 0) + Math.min(999999, degrees.get(node.id) || 0) }
  })
  const edgeLabelCandidates = visibleEdges.flatMap((edge, index) => {
    const source = nodeMap.get(edge.source), target = nodeMap.get(edge.target)
    if (!selected || !source || !target || (active !== source.id && active !== target.id)) return []
    return [{ id: edgeLabelKey(index), x: (source.x + target.x) / 2 * scale + tx,
      y: (source.y + target.y) / 2 * scale + ty - 5,
      width: Array.from(edge.relation).reduce((sum, char) => sum + (/[^\x00-\x7f]/.test(char) ? 10 : 6), 0),
      height: 12, priority: 1500000, group: 'relations', groupLimit: 8 }]
  })
  const automaticLabels = placeGraphLabels([...nodeLabelCandidates, ...edgeLabelCandidates].map(candidate => ({
    ...candidate, x: candidate.x - 16, y: candidate.y - topInset,
  })), Math.max(0, dims.width - 32), canvasHeight, 32, visible.map(node => ({
    id: node.id, x: node.x * scale + tx - 16, y: node.y * scale + ty - topInset,
    radius: nodeRadius(node.id) + 2,
  })))
  const buttonClass = 'rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-600 hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-teal-500 disabled:opacity-40'
  return (
    <div ref={containerRef} className={`relative w-full h-full overflow-hidden bg-slate-50 ${className || ''}`}
      style={{ backgroundImage: 'radial-gradient(#cbd5e1 0.7px, transparent 0.7px)', backgroundSize: '22px 22px' }}>
      <div className="absolute left-4 right-4 top-3 z-10 flex flex-wrap items-start justify-between gap-2">
        <div><div className="text-xs font-semibold tracking-widest text-slate-700">实体关系网络</div><div className="mt-1 text-[11px] text-slate-500">{visible.length} 个实体 · {visibleEdges.length} 条关系{localOnly ? ' · 邻居视图' : ''}</div>{layoutStatus && <p role="status" className="text-xs text-slate-600">{layoutStatus}</p>}</div>
        <div className="flex w-full max-w-full flex-wrap justify-end gap-1 rounded-lg bg-white/90 p-1 shadow-sm sm:w-auto">
          <div className="relative w-full sm:w-44" onBlur={event => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setSearchOpen(false)
          }}>
            <input role="combobox" aria-label="查找实体" aria-expanded={searchOpen}
              aria-controls={`${marker}-search-results`} aria-autocomplete="list" value={searchQuery}
              onFocus={() => setSearchOpen(Boolean(searchQuery.trim()))}
              onChange={event => { setSearchQuery(event.target.value); setSearchOpen(Boolean(event.target.value.trim())) }}
              onKeyDown={event => {
                if (event.key === 'Enter' && searchResults[0]) { event.preventDefault(); focusSearchResult(searchResults[0]) }
                if (event.key === 'Escape') { setSearchOpen(false); setSearchQuery('') }
              }}
              className="h-[30px] w-full rounded-md border border-slate-200 bg-white px-2.5 text-xs text-slate-700 outline-none placeholder:text-slate-400 focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
              placeholder={searchAllNodes ? '查找全库实体…' : '查找实体名称…'} />
            {searchOpen && <div id={`${marker}-search-results`} role="listbox"
              className="absolute right-0 top-[34px] z-20 max-h-56 w-full min-w-56 overflow-auto rounded-lg border border-slate-200 bg-white p-1 shadow-lg">
              {searchResults.map(node => <button key={node.id} role="option" aria-selected={selected === node.id}
                onPointerDown={event => event.preventDefault()} onClick={() => focusSearchResult(node)}
                className="block w-full rounded-md px-2.5 py-2 text-left hover:bg-teal-50 focus-visible:bg-teal-50 focus-visible:outline-none">
                <span className="block truncate text-xs font-medium text-slate-700">{node.label}</span>
                <span className="mt-0.5 block truncate text-[10px] text-slate-400">{node.category} · {degrees.get(node.id) ?? node.degree ?? 0} 条关联</span>
              </button>)}
              {remoteSearch.loading && remoteSearch.query === normalizedSearchQuery && (
                <div role="status" className="px-2.5 py-2 text-xs text-slate-400">正在搜索全库…</div>
              )}
              {remoteSearch.error && remoteSearch.query === normalizedSearchQuery && (
                <div role="alert" className="px-2.5 py-2 text-xs text-amber-700">全库搜索失败，当前仅显示已加载结果</div>
              )}
              {searchResults.length === 0 && !remoteSearch.loading && (
                <div className="px-2.5 py-3 text-xs text-slate-400">没有匹配实体</div>
              )}
            </div>}
          </div>
          <button className={buttonClass} aria-label="缩小" onClick={() => setCamera(c => ({ ...c, zoom: Math.max(.3, c.zoom / 1.3) }))}>−</button>
          <button className={buttonClass} aria-label="放大" onClick={() => setCamera(c => ({ ...c, zoom: Math.min(5, c.zoom * 1.3) }))}>＋</button>
          <button className={buttonClass} onClick={() => setCamera({ x: 0, y: 0, zoom: 1 })}>适配视野</button>
          <button className={buttonClass} aria-pressed={allLabels} onClick={() => setAllLabels(v => !v)}>{allLabels ? '精简标签' : '全部标签'}</button>
          <button className={buttonClass} disabled={!selected} aria-pressed={localOnly} onClick={() => { setLocalOnly(v => !v); setCamera({ x: 0, y: 0, zoom: 1 }) }}>{localOnly ? '返回全图' : '只看邻居'}</button>
        </div>
      </div>
      <svg width={dims.width} height={dims.height} className="block touch-none cursor-grab active:cursor-grabbing" aria-label="知识图谱，拖动平移，点击实体查看关系"
        onPointerDown={e => { if (e.button !== 0) return; drag.current = { x: e.clientX, y: e.clientY, cx: camera.x, cy: camera.y }; e.currentTarget.setPointerCapture(e.pointerId) }}
        onPointerMove={e => { const d = drag.current; if (d) setCamera(c => ({ ...c, x: d.cx + e.clientX - d.x, y: d.cy + e.clientY - d.y })) }}
        onPointerUp={e => { drag.current = null; if (e.currentTarget.hasPointerCapture?.(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId) }}
        onLostPointerCapture={() => { drag.current = null }}
        onPointerCancel={() => { drag.current = null }}>
        <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10" fill="#94a3b8" /></marker></defs>
        <g transform={`translate(${tx},${ty}) scale(${scale})`}>
          {visibleEdges.map((edge, i) => {
            const s = nodeMap.get(edge.source), t = nodeMap.get(edge.target)
            if (!s || !t || (localOnly && active && (!neighbors.has(s.id) || !neighbors.has(t.id)))) return null
            const connected = active === s.id || active === t.id
            const path = pathEdgeKeys.has(JSON.stringify(directed ? [s.id, t.id] : [s.id, t.id].sort()))
            const distance = Math.hypot(t.x-s.x,t.y-s.y) || 1
            return <g key={i} opacity={active && !connected ? .08 : 1}>
              <line x1={s.x} y1={s.y} x2={t.x-(t.x-s.x)/distance*nodeRadius(t.id)/scale} y2={t.y-(t.y-s.y)/distance*nodeRadius(t.id)/scale}
                stroke={path ? '#d97706' : connected ? '#0d9488' : '#cbd5e1'} strokeWidth={(connected || path ? 1.5 : 0.75)/scale} markerEnd={directed ? `url(#${marker})` : undefined} />
              {connected && selected && (allLabels || automaticLabels.has(edgeLabelKey(i))) && <text x={(s.x+t.x)/2} y={(s.y+t.y)/2-5/scale} textAnchor="middle" fontSize={10/scale} fill="#0f766e" stroke="#f8fafc" strokeWidth={3/scale} paintOrder="stroke">{edge.relation}</text>}
            </g>
          })}
          {/* Paint the active entity last so dense neighbors cannot cover its name. */}
          {[...visible.filter(node => node.id !== active), ...visible.filter(node => node.id === active)].map(node => {
            const color = CATEGORY_COLORS[node.category] || DEFAULT_COLOR
            const focused = active === node.id
            const radius = nodeRadius(node.id) / scale
            const showLabel = allLabels || automaticLabels.has(nodeLabelKey(node.id))
            const labelOffset = !allLabels ? automaticLabels.get(nodeLabelKey(node.id)) : undefined
            return <g key={node.id} transform={`translate(${node.x},${node.y})`} opacity={active && !neighbors.has(node.id) ? .14 : 1}
              className="cursor-pointer" role="button" tabIndex={0} aria-label={`查看实体 ${node.label}`}
              onPointerDown={e => e.stopPropagation()} onClick={() => {
                if (selected === node.id) { setSelected(null); setLocalOnly(false) }
                else setSelected(node.id)
              }}
              onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                if (selected === node.id) { setSelected(null); setLocalOnly(false) }
                else setSelected(node.id)
              } }}
              onMouseEnter={() => setHovered(node.id)} onMouseLeave={() => setHovered(null)}>
              <title>{node.label} · {node.category}</title>
              {(focused || hitNodes?.has(node.id)) && <circle r={radius+4/scale} fill="none" stroke={hitNodes?.has(node.id) ? '#f59e0b' : color} strokeWidth={2/scale} opacity={.6} />}
              <circle r={radius} fill={color} stroke="white" strokeWidth={1.5/scale} />
              {showLabel && <text x={(labelOffset?.dx || 0)/scale} y={radius+(15+(labelOffset?.dy || 0))/scale} textAnchor="middle" fontSize={11/scale} fontWeight={focused ? 700 : 500} fill="#334155" stroke="#f8fafc" strokeWidth={4/scale} paintOrder="stroke" className="select-none pointer-events-none">{node.label.length > 22 && !focused ? node.label.slice(0,22)+'…' : node.label}</text>}
            </g>
          })}
        </g>
      </svg>
      {detail ? <div className="absolute bottom-3 left-3 right-3 rounded-xl border border-slate-200 bg-white/95 p-4 shadow-sm">
        <div className="flex items-start justify-between gap-3"><div className="min-w-0"><div className="break-all text-sm font-semibold text-slate-800">{detail.label}</div><div className="mt-1 text-xs text-teal-700">{detail.category} · {degrees.get(detail.id) || 0} 条关联{hitNodes?.has(detail.id) ? ' · 检索命中' : ''}</div></div>
        {selected && <button className={buttonClass} onClick={() => { setSelected(null); setHovered(null); setLocalOnly(false) }}>取消选择</button>}</div>
        <p className="mt-2 max-h-20 overflow-auto text-xs leading-relaxed text-slate-500">{detail.description || '暂无实体描述'}</p>
      </div> : <div className="absolute bottom-3 left-4 right-4 flex flex-wrap justify-between gap-3 text-[10px] text-slate-500">
        <div className="flex flex-wrap gap-x-3 gap-y-1">{Array.from(new Set(nodes.map(n => n.category))).map(cat => <span key={cat} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: CATEGORY_COLORS[cat] || DEFAULT_COLOR }} />{cat}</span>)}</div>
        <span>拖动平移 · 点击查看关系 · 标签自动避让，可切换全部标签</span>
      </div>}
    </div>
  )
}
