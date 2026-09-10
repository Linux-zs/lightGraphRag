import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import GraphView from './GraphView'
import userEvent from '@testing-library/user-event'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

it('keeps labels bounded for a selected hub while preserving every node and the all-label override', async () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = Array.from({ length: 80 }, (_, i) => ({ id: String(i), label: `实体${i}`, category: '概念', description: '', critical: false }))
  const edges = nodes.slice(1).map(node => ({ source: '0', target: node.id, relation: '关联' }))
  const view = render(<GraphView nodes={nodes} edges={edges} />)
  await userEvent.click(screen.getByRole('button', { name: '查看实体 实体0' }))
  expect(view.container.querySelectorAll('svg text').length).toBeLessThanOrEqual(32)
  expect(view.container.querySelectorAll('svg [role="button"]')).toHaveLength(80)
  expect(view.container.querySelector('svg [aria-label="查看实体 实体0"] text')).toHaveTextContent('实体0')
  await userEvent.click(screen.getByRole('button', { name: '全部标签' }))
  expect(view.container.querySelectorAll('svg [role="button"] text')).toHaveLength(80)
  expect(view.container.querySelectorAll('svg text')).toHaveLength(159)
})

it('releases pointer capture only when the canvas owns it', () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const view = render(<GraphView nodes={[]} edges={[]} />)
  const canvas = view.container.querySelector('svg')!
  const release = vi.fn()
  const hasCapture = vi.fn().mockReturnValue(false)
  Object.defineProperties(canvas, {
    hasPointerCapture: { value: hasCapture },
    releasePointerCapture: { value: release },
  })
  fireEvent.pointerUp(canvas)
  expect(release).not.toHaveBeenCalled()
  hasCapture.mockReturnValue(true)
  fireEvent.pointerUp(canvas)
  expect(release).toHaveBeenCalledTimes(1)
})

it('reports the displayed subgraph count and exits local view on deselection', async () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = ['A', 'B', 'C'].map(id => ({ id, label: id, category: '概念', description: '', critical: false }))
  const edges = [{ source: 'A', target: 'B', relation: 'one' }, { source: 'B', target: 'C', relation: 'two' }]
  render(<GraphView nodes={nodes} edges={edges} />)
  await userEvent.click(screen.getByRole('button', { name: '查看实体 A' }))
  await userEvent.click(screen.getByRole('button', { name: '只看邻居' }))
  expect(screen.getByText('2 个实体 · 1 条关系 · 邻居视图')).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: '查看实体 A' }))
  expect(screen.getByText('3 个实体 · 2 条关系')).toBeInTheDocument()
})

it('draws arrows only for explicitly directed graph data', () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = ['A', 'B'].map(id => ({ id, label: id, category: '概念', description: '', critical: false }))
  const edges = [{ source: 'A', target: 'B', relation: 'related' }]
  const view = render(<GraphView nodes={nodes} edges={edges} />)
  expect(view.container.querySelectorAll('[marker-end]')).toHaveLength(0)
  view.rerender(<GraphView nodes={nodes} edges={edges} directed />)
  expect(view.container.querySelectorAll('[marker-end]')).toHaveLength(1)
})

it('highlights only returned evidence edges, not all edges among hit nodes', () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = ['A', 'B', 'C'].map(id => ({ id, label: id, category: '概念', description: '', critical: false }))
  const edges = [{ source: 'A', target: 'B', relation: 'related' }, { source: 'B', target: 'C', relation: 'other' }]
  const view = render(<GraphView nodes={nodes} edges={edges} hitNodes={new Set(['A', 'B', 'C'])} pathEdges={[{ source: 'B', target: 'A' }]} />)
  expect(view.container.querySelectorAll('[stroke="#d97706"]')).toHaveLength(1)
  view.rerender(<GraphView nodes={nodes} edges={edges} directed pathEdges={[{ source: 'B', target: 'A' }]} />)
  expect(view.container.querySelectorAll('[stroke="#d97706"]')).toHaveLength(0)
})

it('terminates obsolete layout workers and ignores their late results', () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const workers: FakeWorker[] = []
  class FakeWorker {
    onmessage?: (event: { data: unknown }) => void
    onerror?: () => void
    terminate = vi.fn()
    postMessage = vi.fn()
    constructor() { workers.push(this) }
  }
  vi.stubGlobal('Worker', FakeWorker)
  const nodes = [{ id: 'A', label: 'A', category: '概念', description: '', critical: false }]
  const view = render(<GraphView nodes={nodes} edges={[]} />)
  view.rerender(<GraphView nodes={[{ ...nodes[0], id: 'B', label: 'B' }]} edges={[]} />)
  expect(workers[0].terminate).toHaveBeenCalled()
  act(() => workers[0].onmessage?.({ data: [{ ...nodes[0], label: 'obsolete', x: 1, y: 1 }] }))
  expect(screen.queryByText('obsolete')).not.toBeInTheDocument()
  act(() => workers[1].onerror?.())
  expect(screen.getByRole('status')).toHaveTextContent('当前显示基础布局')
  view.unmount()
  expect(workers[1].terminate).toHaveBeenCalled()
})
