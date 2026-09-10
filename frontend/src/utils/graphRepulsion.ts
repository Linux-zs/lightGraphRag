type Point = { x: number; y: number }
type Cell = {
  left: number; top: number; size: number
  x: number; y: number; count: number
  indices?: number[]
  children?: Cell[]
  coincident?: boolean
}

/** Barnes-Hut approximation; traversal counts support non-timing-based scale tests. */
export function graphRepulsion(points: Point[], strength: number, theta = 0.7) {
  const forces = new Float64Array(points.length * 2)
  if (!points.length) return { forces, visits: 0 }
  let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity
  for (const point of points) {
    left = Math.min(left, point.x); right = Math.max(right, point.x)
    top = Math.min(top, point.y); bottom = Math.max(bottom, point.y)
  }

  function build(indices: number[], x: number, y: number, size: number, depth: number): Cell {
    const cell: Cell = { left: x, top: y, size, x: 0, y: 0, count: indices.length }
    for (const index of indices) { cell.x += points[index].x; cell.y += points[index].y }
    cell.x /= indices.length; cell.y /= indices.length
    if (indices.length > 1 && indices.every(index => points[index].x === points[indices[0]].x && points[index].y === points[indices[0]].y)) {
      cell.x = points[indices[0]].x
      cell.y = points[indices[0]].y
      cell.coincident = true
      return cell
    }
    if (indices.length <= 1 || depth >= 24 || size < 1e-6) {
      cell.indices = indices
      return cell
    }
    const half = size / 2
    const groups: number[][] = [[], [], [], []]
    for (const index of indices) {
      const point = points[index]
      groups[(point.x >= x + half ? 1 : 0) + (point.y >= y + half ? 2 : 0)].push(index)
    }
    cell.children = groups.flatMap((group, quadrant) => group.length ? [build(
      group, x + (quadrant % 2) * half, y + (quadrant >= 2 ? half : 0), half, depth + 1,
    )] : [])
    return cell
  }

  const root = build(points.map((_, index) => index), left, top, Math.max(right - left, bottom - top, 1), 0)
  let visits = 0
  for (let index = 0; index < points.length; index++) {
    const point = points[index]
    function add(x: number, y: number, count: number) {
      const dx = point.x - x, dy = point.y - y
      const distanceSquared = dx * dx + dy * dy || 1
      const factor = strength * count / (distanceSquared * Math.sqrt(distanceSquared))
      forces[index * 2] += dx * factor
      forces[index * 2 + 1] += dy * factor
    }
    function visit(cell: Cell) {
      visits++
      if (cell.coincident) {
        add(cell.x, cell.y, cell.count)
        return
      }
      if (cell.indices) {
        for (const other of cell.indices) if (other !== index) add(points[other].x, points[other].y, 1)
        return
      }
      const dx = point.x - cell.x, dy = point.y - cell.y
      const contains = point.x >= cell.left && point.x <= cell.left + cell.size
        && point.y >= cell.top && point.y <= cell.top + cell.size
      if (!contains && cell.size * cell.size < theta * theta * (dx * dx + dy * dy)) {
        add(cell.x, cell.y, cell.count)
      } else {
        for (const child of cell.children!) visit(child)
      }
    }
    visit(root)
  }
  return { forces, visits }
}
