import React, { useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import GraphView from '../../src/components/GraphView'
import '../../src/index.css'

function Fixture() {
  const [size, setSize] = useState(120)
  const [hub, setHub] = useState(false)
  const [narrow, setNarrow] = useState(false)
  const graph = useMemo(() => {
    const names = ['MySQL 主库', '复制链路', '备份系统', '监控平台', '业务服务', '审计日志']
    const categories = ['核心系统', '同步', '内容', '监控', '服务层', '安全']
    const nodes = Array.from({ length: size }, (_, index) => ({
      id: String(index), label: index < 6 ? names[index] : `${names[index % 6]} / 配置项 ${index}`,
      category: categories[index % 6], critical: index < 6,
      description: `合成测试实体 ${index}。用于检查密集布局、完整中文名称、标签避让与邻居视图；不代表真实知识库内容。`,
    }))
    const edges = nodes.slice(1).map((node, index) => ({
      source: hub ? '0' : index < 5 ? String(index) : String(Number(node.id) % 6),
      target: node.id, relation: ['依赖于', '同步到', '写入', '监控'][index % 4],
    }))
    return { nodes, edges }
  }, [size, hub])
  return <main className="p-4">
    <header className="mb-3 flex flex-wrap items-center gap-4 text-sm">
      <strong>图谱视觉测试 · 合成数据，不连接后端</strong>
      <label>节点数 <select aria-label="测试节点数" value={size} onChange={event => setSize(Number(event.target.value))}>
        {[24, 120, 600].map(count => <option key={count}>{count}</option>)}
      </select></label>
      <label><input type="checkbox" checked={hub} onChange={event => setHub(event.target.checked)} /> 高连接度中心</label>
      <label><input type="checkbox" checked={narrow} onChange={event => setNarrow(event.target.checked)} /> 窄屏容器</label>
    </header>
    <div style={{ width: narrow ? 390 : '100%', maxWidth: '100%', height: 'calc(100vh - 110px)', minHeight: 420 }} className="overflow-hidden rounded-xl border border-slate-200">
      <GraphView nodes={graph.nodes} edges={graph.edges} />
    </div>
  </main>
}
createRoot(document.getElementById('root')!).render(<Fixture />)
