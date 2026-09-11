import { expect, it } from 'vitest'
import { chooseGraphLabels, placeGraphLabels } from './graphLabels'

it('tries alternate positions before hiding a label and reserves its actual box', () => {
  const candidates = [
    { id: 'A', x: 100, y: 100, width: 40, height: 12, priority: 10, alternatives: [{ dx: 0, dy: -30 }] },
    { id: 'B', x: 100, y: 70, width: 40, height: 12, priority: 1 },
  ]
  const obstacles = [{ id: 'node', x: 100, y: 95, radius: 8 }]
  const result = placeGraphLabels(candidates, 500, 500, 32, obstacles)
  expect(result.get('A')).toEqual({ dx: 0, dy: -30 })
  expect(result.has('B')).toBe(false)
  expect(candidates[0].y).toBe(100)
})

it('tries the next alternative when one is clipped and counts each label once', () => {
  const candidate = { id: 'A', x: 10, y: 20, width: 40, height: 12, priority: 1,
    alternatives: [{ dx: 0, dy: -30 }, { dx: 40, dy: 0 }] }
  expect([...placeGraphLabels([candidate, candidate], 500, 500, 1)]).toEqual([['A', { dx: 40, dy: 0 }]])
})

it('avoids node bodies for both node and relation labels without consuming the budget', () => {
  const candidates = [
    { id: 'name', ownerId: 'A', x: 128, y: 128, width: 40, height: 12, priority: 10 },
    { id: 'relation', x: 256, y: 128, width: 40, height: 12, priority: 9 },
    { id: 'clear', x: 400, y: 128, width: 40, height: 12, priority: 1 },
  ]
  const obstacles = [{ id: 'B', x: 128, y: 120, radius: 7 }, { id: 'C', x: 256, y: 120, radius: 7 }]
  expect([...chooseGraphLabels(candidates, 500, 500, 1, obstacles)]).toEqual(['clear'])
  expect([...chooseGraphLabels([{ ...candidates[0], force: true }], 500, 500, 1, obstacles)]).toEqual(['name'])
})

it('does not reject its owner and detects obstacles across cell boundaries', () => {
  const label = { id: 'name', ownerId: 'A', x: 64, y: 64, width: 20, height: 12, priority: 1 }
  expect(chooseGraphLabels([label], 500, 500, 32, [{ id: 'A', x: 64, y: 60, radius: 7 }]).has('name')).toBe(true)
  expect(chooseGraphLabels([label], 500, 500, 32, [{ id: 'B', x: 80, y: 60, radius: 7 }]).size).toBe(0)
  expect(chooseGraphLabels([label], 500, 500, 32, [{ id: 'B', x: 90, y: 60, radius: 7 }]).has('name')).toBe(true)
})

it('rejects partially clipped labels at every viewport boundary', () => {
  const base = { width: 40, height: 12, priority: 1 }
  const candidates = [
    { ...base, id: 'left', x: 20, y: 100 },
    { ...base, id: 'right', x: 480, y: 100 },
    { ...base, id: 'top', x: 100, y: 10 },
    { ...base, id: 'bottom', x: 100, y: 499 },
    { ...base, id: 'inside', x: 100, y: 100 },
  ]
  expect([...chooseGraphLabels(candidates, 500, 500)]).toEqual(['inside'])
})

it('retains a focused label and suppresses overlapping lower-priority labels', () => {
  const candidates = [
    { id: 'neighbor', x: 100, y: 100, width: 80, height: 16, priority: 100 },
    { id: 'selected', x: 100, y: 100, width: 80, height: 16, priority: 0, force: true },
    { id: 'distant', x: 300, y: 100, width: 80, height: 16, priority: 1 },
  ]
  expect([...chooseGraphLabels(candidates, 500, 500)]).toEqual(['selected', 'distant'])
  expect(candidates[0].id).toBe('neighbor')
})

it('bounds labels even when every node is a neighbor or retrieval hit', () => {
  const candidates = Array.from({ length: 200 }, (_, i) => ({
    id: String(i), x: (i % 20) * 100 + 20, y: Math.floor(i / 20) * 100 + 20,
    width: 40, height: 12, priority: 2000000,
  }))
  expect(chooseGraphLabels(candidates, 2000, 2000).size).toBe(32)
})

it('skips labels outside the viewport but always retains the focused label', () => {
  const outside = { id: 'outside', x: -100, y: -100, width: 20, height: 12, priority: 1 }
  expect(chooseGraphLabels([outside], 500, 500).size).toBe(0)
  expect(chooseGraphLabels([{ ...outside, force: true }], 500, 500).has('outside')).toBe(true)
})
