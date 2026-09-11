import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import GraphExtractionPreviewPanel from './GraphExtractionPreviewPanel'

afterEach(cleanup)

it('explains that an empty preview does not write to the graph', () => {
  render(<GraphExtractionPreviewPanel preview={null} loading={false} error="" />)
  expect(screen.getByText(/预览不会写入图谱/)).toBeInTheDocument()
})

it('renders sampled entities, relations, and policy rejection totals', () => {
  render(<GraphExtractionPreviewPanel
    loading={false}
    error=""
    preview={{
      file_name: 'source.txt',
      total_chunk_count: 8,
      sampled_chunks: [
        { index: 0, text: 'first', char_count: 5 },
        { index: 7, text: 'last', char_count: 4 },
      ],
      entities: [{
        name: '订单服务', entity_type: '服务', description: '处理订单', record_count: 1,
      }],
      relations: [{
        source: '订单服务', target: '订单库', keywords: '写入', description: '保存订单', record_count: 1,
      }],
      entity_count: 1,
      relation_count: 1,
      entities_truncated: false,
      relations_truncated: false,
      filter_stats: {
        policy_rejections: { entity_name_excluded: 2, relation_type_excluded: 1 },
        sanitized: 2,
        removed_lines: 18,
        sanitized_reasons: { bulk_table_rows: 18 },
      },
      elapsed_seconds: 3.25,
      graph_rule: {
        rule_template_id: 'technical_operations',
        rule_template_name: '技术运维知识库',
        extraction_mode: 'enhanced',
        allow_other_entity_type: false,
        entity_type_count: 8,
        relation_type_count: 8,
        entity_exclusion_count: 1,
        relation_exclusion_count: 1,
        extraction_prompt_preview: '',
        updated_at: '',
      },
    }}
  />)

  expect(screen.getByText('实体 1')).toBeInTheDocument()
  expect(screen.getByText('关系 1')).toBeInTheDocument()
  expect(screen.getByText('规则过滤 3')).toBeInTheDocument()
  expect(screen.getByText('输入净化 2 块 / 18 行')).toBeInTheDocument()
  expect(screen.getByText(/抽样第 1、8 块/)).toBeInTheDocument()
  expect(screen.getAllByText('订单服务')).toHaveLength(2)
  expect(screen.getByText('订单库')).toBeInTheDocument()
})
