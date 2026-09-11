import { expect, it } from 'vitest'
import { computeLayout } from './graphLayout'

it('keeps dense multi-hub layouts bounded instead of ejecting outliers', () => {
  const nodes = Array.from({ length: 600 }, (_, i) => ({ id: String(i), label: String(i), category: '概念', description: '', critical: false }))
  const edges = nodes.slice(1).map((node, i) => ({ source: i < 5 ? String(i) : String(Number(node.id) % 6), target: node.id, relation: 'related' }))
  const result = computeLayout(nodes, edges, 1000, 750)
  const xs = result.map(node => node.x), ys = result.map(node => node.y)
  expect(Math.max(...xs) - Math.min(...xs)).toBeLessThan(5000)
  expect(Math.max(...ys) - Math.min(...ys)).toBeLessThan(5000)
  expect(result).toHaveLength(600)
})

it('produces finite distinct positions for a large connected graph without dropping nodes', () => {
  const nodes = Array.from({ length: 400 }, (_, i) => ({ id: String(i), label: String(i), category: '概念', description: '', critical: false }))
  const edges = nodes.slice(1).map((node, i) => ({ source: nodes[i].id, target: node.id, relation: 'related' }))
  const result = computeLayout(nodes, edges, 1000, 750)
  expect(result.map(node => node.id)).toEqual(nodes.map(node => node.id))
  expect(result.every(node => Number.isFinite(node.x) && Number.isFinite(node.y))).toBe(true)
  expect(new Set(result.map(node => `${node.x},${node.y}`)).size).toBe(400)
})

it('returns deterministic finite coordinates without mutating graph inputs', () => {
  const nodes = ['A', 'B', 'C'].map(id => ({ id, label: id, category: '概念', description: '', critical: false }))
  const edges = [{ source: 'A', target: 'B', relation: 'related' }]
  const result = computeLayout(nodes, edges, 1000, 750)
  expect(result).toEqual(computeLayout(nodes, edges, 1000, 750))
  expect(result.every(node => Number.isFinite(node.x) && Number.isFinite(node.y))).toBe(true)
  expect(nodes.every(node => !('x' in node))).toBe(true)
  expect(computeLayout([], [], 1000, 750)).toEqual([])
})
