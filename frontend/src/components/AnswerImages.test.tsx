import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AnswerImageFigure, AnswerImagesContext, inlineAnswerImages } from './AnswerImages'
import type { Citation } from '../api'

vi.mock('./Multimodal', () => ({ StoredImage: ({ path, alt }: { path: string; alt: string }) => <img src={path} alt={alt} /> }))
afterEach(cleanup)

const citations: Citation[] = [1, 2, 3, 4, 5].map(index => ({
  index, image_id: index === 2 ? 'image-1' : `image-${index}`, doc_id: 'doc',
  doc_name: '日志.docx', chunk_index: -1, excerpt: '', location: { paragraph: index },
}))
function Answer({ text }: { text: string }) {
  return <AnswerImagesContext.Provider value={{ workspace: 'test space', citations }}>
    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[inlineAnswerImages(citations)]}
      components={{ figure: AnswerImageFigure }}>{text}</ReactMarkdown>
  </AnswerImagesContext.Provider>
}

it('inserts verified images after their blocks, deduplicates and limits to three', () => {
  const { container } = render(<Answer text={'启动成功[1]。\n\n重复[2]，未知[99]。\n\n- 第二图[3]\n- 第三图[4]\n\n更多[5]。'} />)
  expect(screen.getAllByRole('img')).toHaveLength(3)
  const paragraph = screen.getByText('启动成功[1]。')
  expect(paragraph.nextElementSibling?.tagName).toBe('FIGURE')
  expect(screen.getAllByRole('img')[0].getAttribute('src')).toContain('workspace=test%20space&doc_id=doc')
  expect(container.querySelector('p figure')).toBeNull()
  expect(screen.queryByText('图片 [5]')).toBeNull()
})

it('ignores citation-looking numbers in code and unreferenced images', () => {
  render(<Answer text={'`[1]`\n\n```text\n[3]\n```\n\n普通文字[99]。'} />)
  expect(screen.queryByRole('img')).toBeNull()
})

it('keeps collapse state when streaming text grows and can expand again', async () => {
  const view = render(<Answer text="日志[1]。" />)
  await userEvent.click(screen.getByRole('button', { name: '收起图片' }))
  expect(screen.queryByRole('img')).toBeNull()
  view.rerender(<Answer text={'日志[1]。\n\n继续解释。'} />)
  expect(screen.getByRole('button', { name: '展开图片' })).toHaveAttribute('aria-expanded', 'false')
  await userEvent.click(screen.getByRole('button', { name: '展开图片' }))
  expect(screen.getByRole('img')).toBeInTheDocument()
})
