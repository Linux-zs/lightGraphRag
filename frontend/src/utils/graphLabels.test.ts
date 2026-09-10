import { expect, it } from 'vitest'
import { chooseGraphLabels } from './graphLabels'

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
