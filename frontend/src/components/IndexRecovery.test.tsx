import { act, cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'
import { listIndexTasks, recoverIndexTask } from '../api'
import IndexRecovery from './IndexRecovery'

vi.mock('../api', () => ({ listIndexTasks: vi.fn(), recoverIndexTask: vi.fn() }))
afterEach(() => { cleanup(); vi.resetAllMocks() })

it('does not describe refresh failure as a failed index recovery', async () => {
  vi.mocked(listIndexTasks).mockResolvedValue([{ task_id: 'task', workspace: 'kb', phase: 'recovery_required' }] as Awaited<ReturnType<typeof listIndexTasks>>)
  vi.mocked(recoverIndexTask).mockResolvedValue({ message: '原索引已恢复' } as Awaited<ReturnType<typeof recoverIndexTask>>)
  const refresh = vi.fn().mockRejectedValue(new Error('offline'))
  render(<IndexRecovery workspace="kb" onRecovered={refresh} />)
  await userEvent.click(await screen.findByRole('button'))
  expect(await screen.findByRole('status')).toHaveTextContent('页面数据刷新失败')
  expect(screen.getByRole('status')).not.toHaveTextContent('索引仍隔离')
  expect(refresh).toHaveBeenCalledOnce()
})

it('ignores late recovery results after switching workspace', async () => {
  vi.mocked(listIndexTasks).mockResolvedValueOnce([{ task_id: 'task', workspace: 'kb', phase: 'recovery_required' }] as Awaited<ReturnType<typeof listIndexTasks>>).mockResolvedValueOnce([])
  let finish!: (task: Awaited<ReturnType<typeof recoverIndexTask>>) => void
  vi.mocked(recoverIndexTask).mockImplementation(() => new Promise(resolve => { finish = resolve }))
  const refresh = vi.fn()
  const view = render(<IndexRecovery workspace="kb" onRecovered={refresh} />)
  await userEvent.click(await screen.findByRole('button'))
  view.rerender(<IndexRecovery workspace="other" onRecovered={refresh} />)
  await act(async () => { finish({ message: 'old result' } as Awaited<ReturnType<typeof recoverIndexTask>>) })
  expect(refresh).not.toHaveBeenCalled()
  expect(screen.queryByText('old result')).not.toBeInTheDocument()
})

it('keeps a failed recovery available and reports verified success on retry', async () => {
  vi.mocked(listIndexTasks).mockResolvedValue([
    { task_id: 'old', workspace: 'kb', phase: 'recovery_required', updated_at: '2020-01-01' },
    { task_id: 'other', workspace: 'elsewhere', phase: 'recovery_required' },
  ] as Awaited<ReturnType<typeof listIndexTasks>>)
  vi.mocked(recoverIndexTask).mockRejectedValueOnce(new Error('ambiguous'))
    .mockResolvedValueOnce({ message: '原索引已恢复' } as Awaited<ReturnType<typeof recoverIndexTask>>)
  render(<IndexRecovery workspace="kb" />)
  const button = await screen.findByRole('button', { name: '校验并尝试恢复' })
  expect(screen.queryByText('任务 other')).not.toBeInTheDocument()
  await userEvent.click(button)
  expect(await screen.findByRole('status')).toHaveTextContent('恢复失败，索引仍隔离')
  expect(button).toBeEnabled()
  await userEvent.click(button)
  expect(await screen.findByRole('status')).toHaveTextContent('原索引已恢复')
  expect(screen.queryByRole('button')).not.toBeInTheDocument()
  expect(recoverIndexTask).toHaveBeenCalledWith('old', 'kb', expect.any(AbortSignal))
})
