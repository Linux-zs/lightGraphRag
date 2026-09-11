import { expect, it } from 'vitest'
import { orientGraph } from './graphOrientation'

it('uses horizontal space for a tall graph without changing identities or distances', () => {
  const nodes = [{ id: 'A', x: 10, y: 20 }, { id: 'B', x: 30, y: 820 }]
  const result = orientGraph(nodes, 1200, 450)
  expect(result.map(node => node.id)).toEqual(['A', 'B'])
  expect(Math.abs(result[1].x - result[0].x)).toBe(800)
  expect(Math.hypot(result[1].x - result[0].x, result[1].y - result[0].y))
    .toBeCloseTo(Math.hypot(20, 800))
  expect(nodes[0]).toEqual({ id: 'A', x: 10, y: 20 })
  expect(orientGraph(nodes, 1200, 450)).toEqual(result)
})

it('retains vertical layouts on narrow screens and avoids immaterial rotations', () => {
  const tall = [{ x: 10, y: 20 }, { x: 30, y: 820 }]
  expect(orientGraph(tall, 350, 450)).toBe(tall)
  const square = [{ x: 0, y: 0 }, { x: 400, y: 400 }]
  expect(orientGraph(square, 1200, 450)).toBe(square)
  expect(orientGraph([], 1200, 450)).toEqual([])
  expect(orientGraph(tall, 0, 450)).toBe(tall)
})
