import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiFetch } from '../api'
import { ImageIndexPanel, ImageRecall, StoredImage } from './Multimodal'

vi.mock('../api', () => ({ apiFetch: vi.fn() }))
afterEach(() => { cleanup(); vi.clearAllMocks() })

describe('multimodal UI', () => {
  it('restores the running image task for this workspace after remount', async () => {
    vi.mocked(apiFetch).mockImplementation(async path => new Response(JSON.stringify(
      String(path).includes('image-documents') ? [] : String(path).endsWith('/index-tasks') ? [
        { task_id: 'other', workspace: 'beta', kind: 'images', status: 'running', message: 'other task' },
        { task_id: 'active', workspace: 'alpha', kind: 'images', status: 'running', message: '图片处理 2/10' },
      ] : { status: 'running', message: '图片处理 2/10' },
    )))
    render(<ImageIndexPanel workspace="alpha" names={['source.docx']} />)
    expect(await screen.findByRole('button', { name: '取消任务' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /建立 \/ 重建图片索引/ })).toBeDisabled()
    expect(await screen.findByText('图片处理 2/10')).toBeInTheDocument()
  })
  it('explains a deleted source instead of leaving a broken image', async () => {
    vi.mocked(apiFetch).mockResolvedValue(new Response('', { status: 404 }))
    render(<StoredImage path="/kb/images/deleted?workspace=a&doc_id=old" alt="old source" />)
    expect(await screen.findByText('图片来源已删除或不可用')).toBeTruthy()
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('submits the workspace and exposes empty results', async () => {
    vi.mocked(apiFetch).mockResolvedValue(new Response(JSON.stringify({ hits: [] })))
    render(<ImageRecall workspace="alpha" />)
    expect((screen.getByRole('button', { name: '检索图片' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.change(screen.getByRole('textbox', { name: '图片检索文字' }), { target: { value: 'architecture' } })
    fireEvent.click(screen.getByRole('button', { name: '检索图片' }))
    await screen.findByText('未找到图片，请先在知识库管理中建立图片索引。')
    expect(JSON.parse(vi.mocked(apiFetch).mock.calls[0][1]?.body as string)).toEqual({ workspace: 'alpha', query: 'architecture', image_id: null })
  })

  it('shows model incompatibility instead of an empty success', async () => {
    vi.mocked(apiFetch).mockResolvedValue(new Response(JSON.stringify({ detail: '请重建图片索引' }), { status: 400 }))
    render(<ImageRecall workspace="alpha" />)
    fireEvent.change(screen.getByRole('textbox', { name: '图片检索文字' }), { target: { value: 'photo' } })
    fireEvent.click(screen.getByRole('button', { name: '检索图片' }))
    await waitFor(() => expect(screen.getByRole('alert').textContent).toBe('请重建图片索引'))
  })
})
