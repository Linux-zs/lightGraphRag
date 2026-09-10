import { expect, it } from 'vitest'
import { graphRepulsion } from './graphRepulsion'

function points(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    x: (i % 40) * 20 + Math.sin(i) * 2, y: Math.floor(i / 40) * 20 + Math.cos(i) * 2,
  }))
}

it('matches exact pairwise forces when approximation is disabled', () => {
  const input = points(100)
  const expected = new Float64Array(input.length * 2)
  input.forEach((point, i) => input.forEach((other, j) => {
    if (i === j) return
    const dx = point.x - other.x, dy = point.y - other.y
    const squared = dx * dx + dy * dy || 1
    expected[i * 2] += dx / (squared * Math.sqrt(squared))
    expected[i * 2 + 1] += dy / (squared * Math.sqrt(squared))
  }))
  const actual = graphRepulsion(input, 1, 0).forces
  actual.forEach((force, index) => expect(force).toBeCloseTo(expected[index], 12))
})

it('bounds approximation error and reduces work on a 1600-node fixture', () => {
  const input = points(1600)
  const exact = graphRepulsion(input, 1, 0)
  const approximate = graphRepulsion(input, 1)
  let error = 0, magnitude = 0
  exact.forces.forEach((force, index) => {
    error += Math.abs(force - approximate.forces[index])
    magnitude += Math.abs(force)
  })
  expect(error / magnitude).toBeLessThan(0.08)
  expect(approximate.visits).toBeLessThan(exact.visits / 8)
  expect(approximate.forces).toEqual(graphRepulsion(input, 1).forces)
})

it('handles empty, single and coincident points without self-force or deep recursion', () => {
  expect(graphRepulsion([], 1).forces).toHaveLength(0)
  expect([...graphRepulsion([{ x: 1, y: 1 }], 1).forces]).toEqual([0, 0])
  const result = graphRepulsion(Array.from({ length: 1000 }, () => ({ x: 0.1, y: 0.7 })), 1)
  expect(result.forces.every(value => value === 0)).toBe(true)
  expect(result.visits).toBe(1000)
})
