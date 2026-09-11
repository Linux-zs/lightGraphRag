export type LabelCandidate = {
  id: string; x: number; y: number; width: number; height: number; priority: number
  force?: boolean
  group?: string
  groupLimit?: number
  ownerId?: string
  alternatives?: { dx: number; dy: number }[]
}

export type LabelObstacle = { id: string; x: number; y: number; radius: number }

/** Greedy screen-space label packing; never removes the underlying graph nodes. */
export function chooseGraphLabels(candidates: LabelCandidate[], width: number, height: number, limit = 32, obstacles: LabelObstacle[] = []) {
  return new Set(placeGraphLabels(candidates, width, height, limit, obstacles).keys())
}

export function placeGraphLabels(candidates: LabelCandidate[], width: number, height: number, limit = 32, obstacles: LabelObstacle[] = []) {
  const chosen = new Map<string, { dx: number; dy: number }>()
  const groupCounts = new Map<string, number>()
  const occupied: { left: number; right: number; top: number; bottom: number }[] = []
  // Index screen-space node bounds so dense graphs don't require scanning every
  // node for every candidate on each pointer/zoom update.
  const cells = new Map<string, LabelObstacle[]>()
  const cellSize = 64
  for (const obstacle of obstacles) {
    for (let x = Math.floor((obstacle.x - obstacle.radius) / cellSize); x <= Math.floor((obstacle.x + obstacle.radius) / cellSize); x++) {
      for (let y = Math.floor((obstacle.y - obstacle.radius) / cellSize); y <= Math.floor((obstacle.y + obstacle.radius) / cellSize); y++) {
        const key = `${x},${y}`
        const cell = cells.get(key) || []
        cell.push(obstacle); cells.set(key, cell)
      }
    }
  }
  const coversNode = (box: typeof occupied[number], ownerId?: string) => {
    for (let x = Math.floor(box.left / cellSize); x <= Math.floor(box.right / cellSize); x++) {
      for (let y = Math.floor(box.top / cellSize); y <= Math.floor(box.bottom / cellSize); y++) {
        if (cells.get(`${x},${y}`)?.some(node => node.id !== ownerId &&
          Math.hypot(node.x - Math.max(box.left, Math.min(node.x, box.right)),
            node.y - Math.max(box.top, Math.min(node.y, box.bottom))) < node.radius)) return true
      }
    }
    return false
  }
  const ranked = [...candidates].sort((a, b) => Number(!!b.force) - Number(!!a.force)
    || b.priority - a.priority || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
  for (const candidate of ranked) {
    if (chosen.size >= limit && !candidate.force) continue
    if (candidate.group && (groupCounts.get(candidate.group) || 0) >= (candidate.groupLimit ?? limit)) continue
    if (chosen.has(candidate.id)) continue
    const boxAt = ({ dx, dy }: { dx: number; dy: number }) => ({
      left: candidate.x + dx - candidate.width / 2 - 4, right: candidate.x + dx + candidate.width / 2 + 4,
      top: candidate.y + dy - candidate.height - 3, bottom: candidate.y + dy + 3,
    })
    const offset = [{ dx: 0, dy: 0 }, ...(candidate.alternatives || [])].find(position => {
      const box = boxAt(position)
      return box.left >= 0 && box.right <= width && box.top >= 0 && box.bottom <= height
        && !coversNode(box, candidate.ownerId)
        && !occupied.some(other => box.left < other.right && box.right > other.left
          && box.top < other.bottom && box.bottom > other.top)
    }) || (candidate.force ? { dx: 0, dy: 0 } : undefined)
    if (!offset) continue
    const box = boxAt(offset)
    chosen.set(candidate.id, offset)
    if (candidate.group) groupCounts.set(candidate.group, (groupCounts.get(candidate.group) || 0) + 1)
    occupied.push(box)
  }
  return chosen
}
