import type { GraphNode } from '../api'

/** Rank local graph matches without another backend request. */
export function searchGraphNodes(
  nodes: GraphNode[],
  query: string,
  degrees: ReadonlyMap<string, number>,
  limit = 6,
) {
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)
  if (!terms.length || limit <= 0) return []
  return nodes.flatMap(node => {
    const label = node.label.toLocaleLowerCase()
    const id = node.id.toLocaleLowerCase()
    const haystack = [label, id, node.category, node.entity_type, node.description]
      .filter(Boolean).join(' ').toLocaleLowerCase()
    if (!terms.every(term => haystack.includes(term))) return []
    const phrase = terms.join(' ')
    const rank = label === phrase || id === phrase ? 0
      : label.startsWith(phrase) || id.startsWith(phrase) ? 1
        : label.includes(phrase) || id.includes(phrase) ? 2 : 3
    return [{ node, rank, degree: degrees.get(node.id) || node.degree || 0 }]
  }).sort((a, b) => a.rank - b.rank || b.degree - a.degree
    || a.node.label.localeCompare(b.node.label, 'zh-CN') || a.node.id.localeCompare(b.node.id))
    .slice(0, limit).map(item => item.node)
}
