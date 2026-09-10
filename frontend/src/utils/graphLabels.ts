export type LabelCandidate = {
  id: string; x: number; y: number; width: number; height: number; priority: number
  force?: boolean
  group?: string
  groupLimit?: number
}

/** Greedy screen-space label packing; never removes the underlying graph nodes. */
export function chooseGraphLabels(candidates: LabelCandidate[], width: number, height: number, limit = 32) {
  const chosen = new Set<string>()
  const groupCounts = new Map<string, number>()
  const occupied: { left: number; right: number; top: number; bottom: number }[] = []
  const ranked = [...candidates].sort((a, b) => Number(!!b.force) - Number(!!a.force)
    || b.priority - a.priority || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
  for (const candidate of ranked) {
    if (chosen.size >= limit && !candidate.force) continue
    if (candidate.group && (groupCounts.get(candidate.group) || 0) >= (candidate.groupLimit ?? limit)) continue
    const box = { left: candidate.x - candidate.width / 2 - 4, right: candidate.x + candidate.width / 2 + 4,
      top: candidate.y - candidate.height - 3, bottom: candidate.y + 3 }
    if (!candidate.force && (box.left < 0 || box.right > width || box.top < 0 || box.bottom > height)) continue
    if (!candidate.force && occupied.some(other => box.left < other.right && box.right > other.left
      && box.top < other.bottom && box.bottom > other.top)) continue
    chosen.add(candidate.id)
    if (candidate.group) groupCounts.set(candidate.group, (groupCounts.get(candidate.group) || 0) + 1)
    occupied.push(box)
  }
  return chosen
}
