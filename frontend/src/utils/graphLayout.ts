import type { GraphNode, GraphEdge } from '../api'
import { graphRepulsion } from './graphRepulsion'
export interface PositionedNode extends GraphNode { x: number; y: number }

/** Compute force-directed layout for the given nodes and edges */
export function computeLayout(
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
  const repulsion = 24000
  const attraction = 0.02
  const centering = 0.005
  const damping = 0.85

  for (let iter = 0; iter < iterations; iter++) {
    // Preserve exact small-graph behavior; aggregate distant nodes for large graphs.
    if (positioned.length > 80) {
      const { forces } = graphRepulsion(positioned, repulsion)
      positioned.forEach((node, index) => {
        const velocity = vel.get(node.id)!
        velocity.vx += forces[index * 2]
        velocity.vy += forces[index * 2 + 1]
      })
    } else for (let i = 0; i < positioned.length; i++) {
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
      const force = (dist - 115) * attraction
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
