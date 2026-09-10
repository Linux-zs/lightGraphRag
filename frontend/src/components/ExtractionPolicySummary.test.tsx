import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import ExtractionPolicySummary from './ExtractionPolicySummary'

afterEach(cleanup)
it('shows each rejection reason as record counts', () => {
  render(<ExtractionPolicySummary results={[{ doc_name: 'source.txt', status: 'ok',
    kg_policy_rejections: { entity_type_not_allowed: 3, relation_endpoint_not_allowed: 2, relation_type_not_allowed: 4 } }]} />)
  expect(screen.getByText(/记录数，非去重实体数/)).toBeInTheDocument()
  expect(screen.getByText(/source.txt/)).toHaveTextContent('实体类型不匹配 3')
  expect(screen.getByText(/source.txt/)).toHaveTextContent('关系端点未通过 2')
  expect(screen.getByText(/source.txt/)).toHaveTextContent('关系类型不匹配 4')
})
it('does not report rejection for legacy or zero-count results', () => {
  const view = render(<ExtractionPolicySummary results={[{ doc_name: 'old', status: 'ok' },
    { doc_name: 'new', status: 'ok', kg_policy_rejections: { entity_type_not_allowed: 0 } }]} />)
  expect(view.container).toBeEmptyDOMElement()
})
