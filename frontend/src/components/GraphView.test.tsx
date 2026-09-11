import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import GraphView from './GraphView'
import userEvent from '@testing-library/user-event'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

it('finds, selects and centers a graph entity without dropping graph data', async () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = [
    { id: 'A', label: 'Alpha', category: '服务', description: '', critical: false },
    { id: 'B', label: 'Beta 数据库', category: '数据库', description: '目标详情', critical: true },
    { id: 'C', label: 'Gamma', category: '服务', description: '', critical: false },
  ]
  const edges = [{ source: 'A', target: 'B', relation: 'uses' }]
  const view = render(<GraphView nodes={nodes} edges={edges} />)
  const before = view.container.querySelector('svg > g')!.getAttribute('transform')
  const search = screen.getByRole('combobox', { name: '查找实体' })
  await userEvent.type(search, 'beta')
  await userEvent.click(screen.getByRole('option', { name: /Beta 数据库/ }))
  expect(search).toHaveValue('Beta 数据库')
  expect(search).toHaveAttribute('aria-expanded', 'false')
  expect(screen.getByText('目标详情')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '只看邻居' })).toBeEnabled()
  expect(view.container.querySelectorAll('svg [role="button"]')).toHaveLength(3)
  expect(view.container.querySelector('svg > g')!.getAttribute('transform')).not.toBe(before)
  fireEvent.click(screen.getByRole('button', { name: '放大' }))
  const zoomed = view.container.querySelector('svg > g')!.getAttribute('transform')
  fireEvent.focus(search)
  await userEvent.click(screen.getByRole('option', { name: /Beta 数据库/ }))
  expect(view.container.querySelector('svg > g')!.getAttribute('transform')).not.toBe(zoomed)
})

it('supports Enter selection and reports an empty entity search', async () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = [{ id: 'A', label: 'Alpha', category: '服务', description: 'A detail', critical: false }]
  render(<GraphView nodes={nodes} edges={[]} />)
  const search = screen.getByRole('combobox', { name: '查找实体' })
  await userEvent.type(search, 'missing')
  expect(screen.getByText('没有匹配实体')).toBeInTheDocument()
  await userEvent.clear(search)
  await userEvent.type(search, 'alpha{Enter}')
  expect(screen.getByText('A detail')).toBeInTheDocument()
  await userEvent.type(search, '{Escape}')
  expect(search).toHaveValue('')
})

it('searches the full graph and requests a neighborhood for an unloaded result', async () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const loaded = [{ id: 'A', label: 'Alpha', category: '服务', description: '', critical: false }]
  const hidden = {
    id: 'hidden', label: 'Hidden Atlas', category: '数据库', description: 'full graph result', critical: false, degree: 7,
  }
  const searchAllNodes = vi.fn().mockResolvedValue([hidden])
  const onOpenSearchResult = vi.fn().mockResolvedValue(undefined)
  render(<GraphView
    nodes={loaded}
    edges={[]}
    searchAllNodes={searchAllNodes}
    onOpenSearchResult={onOpenSearchResult}
  />)

  const search = screen.getByRole('combobox', { name: '查找实体' })
  expect(search).toHaveAttribute('placeholder', '查找全库实体…')
  await userEvent.type(search, 'hidden')
  const result = await screen.findByRole('option', { name: /Hidden Atlas/ })
  expect(screen.getByText(/数据库 · 7 条关联/)).toBeInTheDocument()
  await userEvent.click(result)

  expect(searchAllNodes).toHaveBeenCalledWith('hidden', expect.any(AbortSignal))
  expect(onOpenSearchResult).toHaveBeenCalledWith(hidden)
  expect(search).toHaveValue('Hidden Atlas')
})

it('selects and centers a focused node after a neighborhood is loaded', () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = [
    { id: 'A', label: 'Alpha', category: '服务', description: '', critical: false },
    { id: 'B', label: 'Beta', category: '数据库', description: 'focused detail', critical: false },
  ]
  render(<GraphView nodes={nodes} edges={[]} focusNodeId="B" />)

  expect(screen.getByText('focused detail')).toBeInTheDocument()
  expect(screen.getByRole('combobox', { name: '查找实体' })).toHaveValue('Beta')
})

it('paints the active node and its label above other nodes without changing layout', () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = ['A', 'B', 'C'].map(id => ({ id, label: id, category: '概念', description: '', critical: false }))
  const view = render(<GraphView nodes={nodes} edges={[]} />)
  const node = screen.getByRole('button', { name: '查看实体 A' })
  const before = node.getAttribute('transform')
  fireEvent.mouseEnter(node)
  const drawn = view.container.querySelectorAll('svg [role="button"]')
  expect(drawn[drawn.length - 1]).toBe(node)
  expect(node.getAttribute('transform')).toBe(before)
  expect(node.querySelector('text')).toHaveTextContent('A')
  fireEvent.click(node)
  fireEvent.mouseLeave(node)
  expect(view.container.querySelector('svg > g > g:last-child')).toBe(node)
})

it('keeps the camera stable on hover and opens details only on selection', () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = [{ id: 'A', label: 'A', category: '概念', description: '实体详情', critical: false }]
  const view = render(<GraphView nodes={nodes} edges={[]} />)
  const camera = () => view.container.querySelector('svg > g')!.getAttribute('transform')
  const before = camera()
  const node = screen.getByRole('button', { name: '查看实体 A' })
  fireEvent.mouseEnter(node)
  expect(camera()).toBe(before)
  expect(screen.queryByText('实体详情')).not.toBeInTheDocument()
  fireEvent.mouseLeave(node)
  expect(camera()).toBe(before)
  fireEvent.click(node)
  expect(screen.getByText('实体详情')).toBeInTheDocument()
})

it('keeps node radii and label fonts constant in screen pixels when zooming', () => {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  const nodes = [{ id: 'A', label: 'A', category: '概念', description: '', critical: false }]
  const view = render(<GraphView nodes={nodes} edges={[]} />)
  const measure = () => {
    const transform = view.container.querySelector('svg > g')!.getAttribute('transform')!
    const scale = Number(transform.match(/scale\(([^)]+)\)/)![1])
    const node = screen.getByRole('button', { name: '查看实体 A' })
    expect(Number(node.querySelector('text')!.getAttribute('font-size')) * scale).toBeCloseTo(11)
    expect(Number(node.querySelector('circle')!.getAttribute('r')) * scale).toBeCloseTo(4.5)
    return scale
  }
  const initial = measure()
  fireEvent.click(screen.getByRole('button', { name: '放大' }))
  expect(measure()).toBeGreaterThan(initial)
  fireEvent.click(screen.getByRole('button', { name: '缩小' }))
  fireEvent.click(screen.getByRole('button', { name: '缩小' }))
  expect(measure()).toBeLessThan(initial)
})

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
