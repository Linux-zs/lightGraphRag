import { computeLayout } from './graphLayout'
import type { GraphNode, GraphEdge } from '../api'

self.onmessage = (event: MessageEvent<{ nodes: GraphNode[]; edges: GraphEdge[] }>) => {
  self.postMessage(computeLayout(event.data.nodes, event.data.edges, 1000, 750))
}
