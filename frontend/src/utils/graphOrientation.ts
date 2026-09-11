/** Rotate coordinates, not SVG text, when doing so materially improves fit. */
export function orientGraph<T extends { x: number; y: number }>(nodes: T[], width: number, height: number): T[] {
  if (nodes.length < 2 || width <= 0 || height <= 0) return nodes
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const node of nodes) {
    minX = Math.min(minX, node.x); maxX = Math.max(maxX, node.x)
    minY = Math.min(minY, node.y); maxY = Math.max(maxY, node.y)
  }
  const spanX = maxX - minX, spanY = maxY - minY
  const normalFit = Math.min(width / (spanX + 170), height / (spanY + 130))
  const rotatedFit = Math.min(width / (spanY + 170), height / (spanX + 130))
  if (rotatedFit <= normalFit * 1.15) return nodes
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2
  return nodes.map(node => ({ ...node, x: cx - (node.y - cy), y: cy + (node.x - cx) }))
}
