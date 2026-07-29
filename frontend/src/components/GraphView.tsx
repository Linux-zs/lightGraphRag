import { useEffect, useRef, useState, useMemo } from 'react'
import type { GraphNode, GraphEdge } from '../api'

interface Props {
  nodes: GraphNode[]
  edges: GraphEdge[]
  /** Node IDs to highlight (hit by retrieval) */
  hitNodes?: Set<string>
  /** Node IDs on the active path (highlighted edges) */
  pathNodes?: Set<string>
  className?: string
}

interface PositionedNode extends GraphNode {
  x: number
  y: number
}

/** Category → color mapping */
const CATEGORY_COLORS: Record<string, string> = {
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

/** Compute force-directed layout for the given nodes and edges */
function computeLayout(
  nodes: GraphNode[],
  edges: GraphEdge[],
  width: number,
  height: number,
): PositionedNode[] {
  if (nodes.length === 0) return []

  const cx = width / 2
  const cy = height / 2
  const radius = Math.min(width, height) * 0.35

  // Initialize positions on a circle
  const positioned: PositionedNode[] = nodes.map((node, i) => {
    const angle = (i / nodes.length) * 2 * Math.PI
    return {
      ...node,
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    }
  })

  const nodeMap = new Map(positioned.map((n) => [n.id, n]))
  const vel = new Map<string, { vx: number; vy: number }>(
    positioned.map((n) => [n.id, { vx: 0, vy: 0 }]),
  )

  // Run force simulation
  const iterations = 300
  const repulsion = 4000
  const attraction = 0.02
  const centering = 0.005
  const damping = 0.85

  for (let iter = 0; iter < iterations; iter++) {
    // Repulsive force between all pairs
    for (let i = 0; i < positioned.length; i++) {
      for (let j = i + 1; j < positioned.length; j++) {
        const dx = positioned[i].x - positioned[j].x
        const dy = positioned[i].y - positioned[j].y
        const distSq = dx * dx + dy * dy || 1
        const dist = Math.sqrt(distSq)
        const force = repulsion / distSq
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        const vi = vel.get(positioned[i].id)!
        const vj = vel.get(positioned[j].id)!
        vi.vx += fx
        vi.vy += fy
        vj.vx -= fx
        vj.vy -= fy
      }
    }

    // Attractive force along edges
    for (const edge of edges) {
      const s = nodeMap.get(edge.source)
      const t = nodeMap.get(edge.target)
      if (!s || !t) continue
      const dx = t.x - s.x
      const dy = t.y - s.y
      const dist = Math.sqrt(dx * dx + dy * dy) || 1
      const force = dist * attraction
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force
      const vs = vel.get(s.id)!
      const vt = vel.get(t.id)!
      vs.vx += fx
      vs.vy += fy
      vt.vx -= fx
      vt.vy -= fy
    }

    // Centering + apply velocity
    for (const node of positioned) {
      const v = vel.get(node.id)!
      v.vx += (cx - node.x) * centering
      v.vy += (cy - node.y) * centering
      node.x += v.vx * 0.1
      node.y += v.vy * 0.1
      v.vx *= damping
      v.vy *= damping
    }
  }

  return positioned
}

export default function GraphView({
  nodes,
  edges,
  hitNodes,
  pathNodes,
  className,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [dims, setDims] = useState({ width: 600, height: 500 })
  const [hovered, setHovered] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)

  // Measure container
  useEffect(() => {
    if (!containerRef.current) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          setDims({ width, height })
        }
      }
    })
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  // Compute layout
  const positionedNodes = useMemo(
    () => computeLayout(nodes, edges, dims.width, dims.height),
    [nodes, edges, dims.width, dims.height],
  )

  const nodeMap = useMemo(
    () => new Map(positionedNodes.map((n) => [n.id, n])),
    [positionedNodes],
  )

  const activeNode = hovered || selected
  const activeNodeData = activeNode ? nodeMap.get(activeNode) : null

  return (
    <div ref={containerRef} className={`relative w-full h-full ${className || ''}`}>
      <svg width={dims.width} height={dims.height} className="block">
        {/* Edges */}
        {edges.map((edge, i) => {
          const s = nodeMap.get(edge.source)
          const t = nodeMap.get(edge.target)
          if (!s || !t) return null

          const isOnPath =
            pathNodes?.has(edge.source) && pathNodes?.has(edge.target)
          const isDimmed = !!activeNode && activeNode !== edge.source && activeNode !== edge.target

          // Calculate arrow endpoint (shorten so arrow doesn't overlap node)
          const dx = t.x - s.x
          const dy = t.y - s.y
          const dist = Math.sqrt(dx * dx + dy * dy) || 1
          const nodeRadius = 28
          const endX = t.x - (dx / dist) * nodeRadius
          const endY = t.y - (dy / dist) * nodeRadius

          return (
            <g key={`edge-${i}`}>
              <line
                x1={s.x}
                y1={s.y}
                x2={endX}
                y2={endY}
                stroke={isOnPath ? '#ef4444' : '#cbd5e1'}
                strokeWidth={isOnPath ? 2.5 : 1.2}
                strokeOpacity={isDimmed ? 0.15 : isOnPath ? 0.8 : 0.4}
                markerEnd={`url(#arrow-${isOnPath ? 'active' : 'default'})`}
              />
            </g>
          )
        })}

        {/* Arrow markers */}
        <defs>
          <marker
            id="arrow-default"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#cbd5e1" />
          </marker>
          <marker
            id="arrow-active"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444" />
          </marker>
        </defs>

        {/* Nodes */}
        {positionedNodes.map((node) => {
          const isHit = hitNodes?.has(node.id)
          const isHovered = hovered === node.id
          const isSelected = selected === node.id
          const isDimmed = !!activeNode && activeNode !== node.id
          const color = CATEGORY_COLORS[node.category] || DEFAULT_COLOR

          return (
            <g
              key={node.id}
              transform={`translate(${node.x}, ${node.y})`}
              className="cursor-pointer"
              onClick={() => setSelected(isSelected ? null : node.id)}
              onMouseEnter={() => setHovered(node.id)}
              onMouseLeave={() => setHovered(null)}
            >
              {/* Glow ring for hit nodes */}
              {isHit && (
                <circle
                  r={34}
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  strokeOpacity={0.6}
                  className="animate-pulse"
                />
              )}
              {/* Node circle */}
              <circle
                r={26}
                fill={isHit ? color : '#fff'}
                stroke={color}
                strokeWidth={isHit ? 2.5 : isHovered || isSelected ? 2 : 1.2}
                fillOpacity={isHit ? 0.9 : isDimmed ? 0.3 : 1}
                strokeOpacity={isDimmed ? 0.3 : 1}
                className="transition-all duration-200"
              />
              {/* Node label */}
              <text
                textAnchor="middle"
                y={4}
                fontSize={10}
                fontWeight={isHit ? 700 : 500}
                fill={isHit ? '#fff' : '#374151'}
                fillOpacity={isDimmed ? 0.3 : 1}
                className="pointer-events-none select-none"
              >
                {node.label.length > 5 ? node.label.slice(0, 4) + '…' : node.label}
              </text>
            </g>
          )
        })}
      </svg>

      {/* Tooltip / detail card */}
      {activeNodeData && (
        <div className="absolute bottom-3 left-3 right-3 bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-xs z-10 pointer-events-none">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="inline-block w-3 h-3 rounded-full"
              style={{ background: CATEGORY_COLORS[activeNodeData.category] || DEFAULT_COLOR }}
            />
            <span className="font-bold text-gray-800">{activeNodeData.label}</span>
            <span className="text-gray-400">|</span>
            <span className="text-gray-500">{activeNodeData.category}</span>
            {activeNodeData.critical && (
              <span className="bg-red-100 text-red-600 px-1.5 py-0.5 rounded text-[10px] font-medium">
                关键
              </span>
            )}
            {hitNodes?.has(activeNodeData.id) && (
              <span className="bg-amber-100 text-amber-600 px-1.5 py-0.5 rounded text-[10px] font-medium">
                命中
              </span>
            )}
          </div>
          <p className="text-gray-600 leading-relaxed">{activeNodeData.description}</p>
        </div>
      )}

      {/* Legend */}
      <div className="absolute top-2 right-2 bg-white/80 backdrop-blur rounded-lg p-2 text-[10px] space-y-1 z-10 pointer-events-none">
        {Array.from(new Set(nodes.map((n) => n.category))).map((cat) => (
          <div key={cat} className="flex items-center gap-1.5">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ background: CATEGORY_COLORS[cat] || DEFAULT_COLOR }}
            />
            <span className="text-gray-600">{cat}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
