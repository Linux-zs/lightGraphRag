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

it('adds a selected noisy entity to the exclusion draft without deleting it', async () => {
  vi.mocked(getGraphGovernanceConfig).mockResolvedValue({
    workspace: 'kb',
    rule_template_id: 'general_knowledge',
    rule_template_name: '通用知识库',
    extraction_mode: 'assist',
    allow_other_entity_type: true,
    entity_types: ['概念'],
    relation_types: ['关联'],
    entity_exclusion_rules: [],
    relation_exclusion_rules: [],
    aliases_text: '',
    extraction_prompt: '',
    effective_extraction_prompt: '',
    reference_files: [],
    updated_at: '',
    audit_log: [],
  })
  vi.mocked(listGraphImports).mockResolvedValue([])
  vi.mocked(listGraphRuleTemplates).mockResolvedValue([])
  vi.mocked(getGraph).mockResolvedValue({
    nodes: [{
      id: 'unknown', label: 'unknown', category: '概念', entity_type: '概念',
      description: '噪声实体', critical: false,
    }],
    edges: [],
    metadata: { total_nodes: 1, total_edges: 0 },
  } as Awaited<ReturnType<typeof getGraph>>)

  render(<GraphPage workspace="kb" />)
  await screen.findByText('graph canvas')
  await userEvent.click(screen.getByRole('button', { name: '实体治理' }))
  await userEvent.click(screen.getByRole('button', { name: /unknown/ }))
  await userEvent.click(screen.getByRole('button', { name: '加入排除草稿' }))

  expect(screen.getByLabelText('实体名称排除')).toHaveValue('unknown')
  expect(screen.getByText(/保存规则并重新索引后生效/)).toBeInTheDocument()
  expect(getGraph).toHaveBeenCalledTimes(1)
})
