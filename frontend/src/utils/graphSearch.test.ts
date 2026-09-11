import { expect, it } from 'vitest'
import { searchGraphNodes } from './graphSearch'

const nodes = [
  { id: 'db-main', label: 'MySQL 主库', category: '数据库', description: '生产订单', critical: true },
  { id: 'db-replica', label: 'MySQL 从库', category: '数据库', description: '只读副本', critical: false },
  { id: 'orders', label: '订单服务', category: '服务', description: 'uses MySQL', critical: false },
]

it('ranks exact and prefix label matches before descriptions, then by degree', () => {
  const degrees = new Map([['db-main', 20], ['db-replica', 2], ['orders', 50]])
  expect(searchGraphNodes(nodes, 'mysql', degrees).map(node => node.id)).toEqual(['db-main', 'db-replica', 'orders'])
  expect(searchGraphNodes(nodes, 'db-main', degrees)[0].id).toBe('db-main')
})

it('supports multi-term category searches, limits results and does not mutate input', () => {
  const before = structuredClone(nodes)
  expect(searchGraphNodes(nodes, '数据库 副本', new Map(), 1).map(node => node.id)).toEqual(['db-replica'])
  expect(searchGraphNodes(nodes, '不存在', new Map())).toEqual([])
  expect(searchGraphNodes(nodes, 'mysql', new Map(), 0)).toEqual([])
  expect(nodes).toEqual(before)
})
