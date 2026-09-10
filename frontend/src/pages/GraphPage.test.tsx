import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'
import GraphPage from './GraphPage'
import { getGraph, getGraphGovernanceConfig, listGraphImports, listGraphRuleTemplates } from '../api'

vi.mock('../components/ConfirmDialog', () => ({ useConfirm: () => vi.fn() }))
vi.mock('../components/GraphView', () => ({ default: () => <div>graph canvas</div> }))
vi.mock('../api', async importOriginal => ({ ...await importOriginal<typeof import('../api')>(),
  getGraph: vi.fn(), getGraphGovernanceConfig: vi.fn(), listGraphImports: vi.fn(), listGraphRuleTemplates: vi.fn() }))
afterEach(() => { cleanup(); vi.resetAllMocks() })

it('distinguishes empty, stale and unavailable graph data', async () => {
  vi.mocked(getGraphGovernanceConfig).mockResolvedValue({} as Awaited<ReturnType<typeof getGraphGovernanceConfig>>)
  vi.mocked(listGraphImports).mockResolvedValue([])
  vi.mocked(listGraphRuleTemplates).mockResolvedValue([])
  vi.mocked(getGraph).mockResolvedValueOnce({ nodes: [], edges: [], metadata: { total_nodes: 0 } } as Awaited<ReturnType<typeof getGraph>>)
    .mockRejectedValue(new Error('offline'))
  const view = render(<GraphPage workspace="kb" />)
  expect(await screen.findByText(/暂无图谱数据/)).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: /^刷新$/ }))
  expect(await screen.findByText(/以下为上次成功加载的数据/)).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: '修正建议' }))
  await userEvent.type(screen.getByPlaceholderText(/例如：根据参考文件检查/), 'private draft from kb')
  view.rerender(<GraphPage workspace="other" />)
  expect(await screen.findByRole('button', { name: '重试加载图谱' })).toBeInTheDocument()
  expect(screen.queryByText(/暂无图谱数据/)).not.toBeInTheDocument()
  expect(screen.queryByText(/以下为上次成功加载的数据/)).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: '修正建议' }))
  expect(screen.getByPlaceholderText(/例如：根据参考文件检查/)).toHaveValue('')
  await userEvent.click(screen.getByRole('button', { name: /^抽取规则$/ }))
  await userEvent.click(screen.getByRole('button', { name: /^严格/ }))
  expect(screen.getByLabelText('允许 Other 类型')).toBeDisabled()
  expect(screen.getByRole('button', { name: /^严格/ })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByText(/Other 只有明确列入实体白名单/)).toBeInTheDocument()
})
