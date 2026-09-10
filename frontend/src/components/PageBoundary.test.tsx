import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import PageBoundary from './PageBoundary'

afterEach(cleanup)

it('contains page failure and resets when navigation changes its key', () => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
  function FailedPage(): never { throw new Error('page unavailable') }
  const view = render(<PageBoundary key="failed"><FailedPage /></PageBoundary>)
  expect(screen.getByRole('alert')).toHaveTextContent('页面加载或运行失败')
  expect(screen.getByRole('button', { name: '刷新页面' })).toBeInTheDocument()
  view.rerender(<PageBoundary key="other"><p>其他页面</p></PageBoundary>)
  expect(screen.getByText('其他页面')).toBeInTheDocument()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})
