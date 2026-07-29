import { useEffect, useMemo, useState } from 'react'
import GraphView from '../components/GraphView'
import {
  applyGraphChanges,
  applyGraphRuleTemplate,
  createGraphEntity,
  createGraphRelation,
  deleteGraphEntity,
  deleteGraphReference,
  deleteGraphRuleTemplate,
  deleteGraphRelation,
  getGraph,
  getGraphGovernanceConfig,
  GraphChange,
  GraphData,
  GraphEdge,
  GraphGovernanceConfig,
  GraphNode,
  GraphRuleTemplate,
  listGraphRuleTemplates,
  mergeGraphEntities,
  saveGraphRuleTemplate,
  suggestGraphChanges,
  updateGraphEntity,
  updateGraphGovernanceConfig,
  updateGraphRelation,
  uploadGraphReference,
  EvidenceChain,
} from '../api'

interface Props {
  workspace: string
}

type Tab = 'overview' | 'rules' | 'entities' | 'relations' | 'suggestions'

const TABS: { key: Tab; label: string }[] = [
  { key: 'overview', label: '图谱总览' },
  { key: 'rules', label: '抽取规则' },
  { key: 'entities', label: '实体治理' },
  { key: 'relations', label: '关系治理' },
  { key: 'suggestions', label: '修正建议' },
]

const emptyConfig: GraphGovernanceConfig = {
  workspace: '',
  rule_template_id: '',
  rule_template_name: '',
  entity_types: [],
  relation_types: [],
  aliases_text: '',
  extraction_prompt: '',
  reference_files: [],
  updated_at: '',
  audit_log: [],
}

function splitLines(text: string) {
  return text.split(/\r?\n/).map((item) => item.trim()).filter(Boolean)
}

function joinLines(items: string[]) {
  return items.join('\n')
}

function formatTime(ts: string) {
  if (!ts) return ''
  try {
    return new Date(ts).toLocaleString('zh-CN', { hour12: false })
  } catch {
    return ts
  }
}

function changeTitle(change: GraphChange) {
  const map: Record<string, string> = {
    create_entity: '新增实体',
    edit_entity: '修改实体',
    delete_entity: '删除实体',
    create_relation: '新增关系',
    edit_relation: '修改关系',
    delete_relation: '删除关系',
    merge_entities: '合并实体',
  }
  return map[change.action] || change.action
}

export default function GraphPage({ workspace }: Props) {
  const [tab, setTab] = useState<Tab>('overview')
  const [data, setData] = useState<GraphData | null>(null)
  const [config, setConfig] = useState<GraphGovernanceConfig>(emptyConfig)
  const [limit, setLimit] = useState(200)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [latestEvidence, setLatestEvidence] = useState<EvidenceChain | null>(null)
  const [ruleTemplates, setRuleTemplates] = useState<GraphRuleTemplate[]>([])
  const [selectedTemplateId, setSelectedTemplateId] = useState('')
  const [templateName, setTemplateName] = useState('')

  const [entityTypesText, setEntityTypesText] = useState('')
  const [relationTypesText, setRelationTypesText] = useState('')
  const [aliasesText, setAliasesText] = useState('')
  const [extractionPrompt, setExtractionPrompt] = useState('')

  const [entityQuery, setEntityQuery] = useState('')
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [entityEdit, setEntityEdit] = useState({ name: '', type: '', description: '' })
  const [entityCreate, setEntityCreate] = useState({ name: '', type: '', description: '' })
  const [mergeDraft, setMergeDraft] = useState({ sources: '', target: '', description: '', type: '' })

  const [relationQuery, setRelationQuery] = useState('')
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null)
  const [relationEdit, setRelationEdit] = useState({ description: '', keywords: '', weight: 1 })
  const [relationCreate, setRelationCreate] = useState({
    source: '',
    target: '',
    description: '',
    keywords: '',
    weight: 1,
  })

  const [suggestInstruction, setSuggestInstruction] = useState('')
  const [suggestions, setSuggestions] = useState<GraphChange[]>([])
  const [selectedSuggestions, setSelectedSuggestions] = useState<Set<number>>(new Set())
  const [suggesting, setSuggesting] = useState(false)

  const loadGraph = async (nextLimit = limit) => {
    const graph = await getGraph(nextLimit, workspace)
    setData(graph)
  }

  const loadConfig = async () => {
    const cfg = await getGraphGovernanceConfig(workspace)
    setConfig(cfg)
    setSelectedTemplateId(cfg.rule_template_id || '')
    setEntityTypesText(joinLines(cfg.entity_types || []))
    setRelationTypesText(joinLines(cfg.relation_types || []))
    setAliasesText(cfg.aliases_text || '')
    setExtractionPrompt(cfg.extraction_prompt || '')
  }

  const loadTemplates = async () => {
    const templates = await listGraphRuleTemplates(workspace)
    setRuleTemplates(templates)
  }

  const loadAll = async (nextLimit = limit) => {
    setLoading(true)
    setError('')
    try {
      await Promise.all([loadGraph(nextLimit), loadConfig(), loadTemplates()])
    } catch (e) {
      setError((e as Error).message || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAll(limit)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace])

  useEffect(() => {
    try {
      const raw = localStorage.getItem(`tdx_latest_evidence_${workspace}`)
      if (!raw) {
        setLatestEvidence(null)
        return
      }
      const payload = JSON.parse(raw)
      setLatestEvidence(payload?.evidence || null)
    } catch {
      setLatestEvidence(null)
    }
  }, [workspace])

  const meta = data?.metadata || {}
  const evidenceNodeIds = useMemo(
    () => new Set((latestEvidence?.nodes || []).map((node) => node.id)),
    [latestEvidence],
  )

  const filteredNodes = useMemo(() => {
    const keyword = entityQuery.trim().toLowerCase()
    const nodes = data?.nodes || []
    if (!keyword) return nodes
    return nodes.filter((node) =>
      [node.id, node.label, node.entity_type, node.description]
        .join(' ')
        .toLowerCase()
        .includes(keyword),
    )
  }, [data, entityQuery])

  const filteredEdges = useMemo(() => {
    const keyword = relationQuery.trim().toLowerCase()
    const edges = data?.edges || []
    if (!keyword) return edges
    return edges.filter((edge) =>
      [edge.source, edge.target, edge.relation, edge.description, edge.keywords]
        .join(' ')
        .toLowerCase()
        .includes(keyword),
    )
  }, [data, relationQuery])

  const selectNode = (node: GraphNode) => {
    setSelectedNode(node)
    setEntityEdit({
      name: node.label || node.id,
      type: node.entity_type || '',
      description: node.description || '',
    })
  }

  const selectEdge = (edge: GraphEdge) => {
    setSelectedEdge(edge)
    setRelationEdit({
      description: edge.description || edge.relation || '',
      keywords: edge.keywords || '',
      weight: Number(edge.weight || 1),
    })
  }

  const runAction = async (fn: () => Promise<unknown>, message: string) => {
    setSaving(true)
    setError('')
    setNotice('')
    try {
      await fn()
      setNotice(message)
      await loadAll(limit)
    } catch (e) {
      setError((e as Error).message || '操作失败')
    } finally {
      setSaving(false)
    }
  }

  const saveRules = () => runAction(async () => {
    const cfg = await updateGraphGovernanceConfig({
      workspace,
      rule_template_id: config.rule_template_id || selectedTemplateId,
      rule_template_name: config.rule_template_name || '当前知识库自定义规则',
      entity_types: splitLines(entityTypesText),
      relation_types: splitLines(relationTypesText),
      aliases_text: aliasesText,
      extraction_prompt: extractionPrompt,
    })
    setConfig(cfg)
  }, '抽取规则已保存')

  const applySelectedTemplate = () => {
    if (!selectedTemplateId) return
    const selected = ruleTemplates.find((item) => item.id === selectedTemplateId)
    const ok = window.confirm(
      `确认将“${selected?.name || selectedTemplateId}”套用到当前知识库？新规则只影响后续上传和重新索引的文档。`,
    )
    if (!ok) return
    return runAction(async () => {
      const cfg = await applyGraphRuleTemplate(workspace, selectedTemplateId)
      setConfig(cfg)
      setSelectedTemplateId(cfg.rule_template_id || '')
      setEntityTypesText(joinLines(cfg.entity_types || []))
      setRelationTypesText(joinLines(cfg.relation_types || []))
      setAliasesText(cfg.aliases_text || '')
      setExtractionPrompt(cfg.extraction_prompt || '')
    }, '已套用抽取规则模板；已索引文档需要重新索引才会按新规则重建图谱')
  }

  const saveCurrentAsTemplate = () => {
    if (!templateName.trim()) return
    return runAction(async () => {
      const saved = await saveGraphRuleTemplate({
        id: '',
        name: templateName.trim(),
        description: `从知识库 ${workspace} 的当前抽取规则另存`,
        entity_types: splitLines(entityTypesText),
        relation_types: splitLines(relationTypesText),
        aliases_text: aliasesText,
        extraction_prompt: extractionPrompt,
        built_in: false,
      })
      setTemplateName('')
      await loadTemplates()
      setSelectedTemplateId(saved.id)
    }, '当前规则已另存为模板')
  }

  const removeTemplate = (template: GraphRuleTemplate) => {
    if (template.built_in) return
    const ok = window.confirm(`确认删除抽取规则模板“${template.name}”？不会影响已套用到知识库的规则内容。`)
    if (!ok) return
    return runAction(async () => {
      await deleteGraphRuleTemplate(template.id)
      await loadTemplates()
      if (selectedTemplateId === template.id) setSelectedTemplateId(config.rule_template_id || '')
    }, '抽取规则模板已删除')
  }

  const handleReferenceUpload = async (file: File | null) => {
    if (!file) return
    await runAction(async () => {
      await uploadGraphReference(file, workspace)
    }, '参考文件已上传')
  }

  const saveEntity = () => {
    if (!selectedNode) return
    return runAction(async () => {
      await updateGraphEntity({
        workspace,
        entity_name: selectedNode.id,
        updated_data: {
          entity_name: entityEdit.name,
          entity_type: entityEdit.type,
          description: entityEdit.description,
        },
        allow_rename: true,
        allow_merge: false,
      })
      setSelectedNode(null)
    }, '实体已更新')
  }

  const addEntity = () => runAction(async () => {
    await createGraphEntity({
      workspace,
      entity_name: entityCreate.name,
      entity_type: entityCreate.type || 'entity',
      description: entityCreate.description,
    })
    setEntityCreate({ name: '', type: '', description: '' })
  }, '实体已新增')

  const removeEntity = () => {
    if (!selectedNode) return
    const ok = window.confirm(`确认删除实体“${selectedNode.id}”及其关系？`)
    if (!ok) return
    return runAction(async () => {
      await deleteGraphEntity(selectedNode.id, workspace)
      setSelectedNode(null)
    }, '实体已删除')
  }

  const addRelation = () => runAction(async () => {
    await createGraphRelation({
      workspace,
      source_entity: relationCreate.source,
      target_entity: relationCreate.target,
      description: relationCreate.description,
      keywords: relationCreate.keywords,
      weight: relationCreate.weight,
    })
    setRelationCreate({ source: '', target: '', description: '', keywords: '', weight: 1 })
  }, '关系已新增')

  const saveRelation = () => {
    if (!selectedEdge) return
    return runAction(async () => {
      await updateGraphRelation({
        workspace,
        source_entity: selectedEdge.source,
        target_entity: selectedEdge.target,
        updated_data: {
          description: relationEdit.description,
          keywords: relationEdit.keywords,
          weight: relationEdit.weight,
        },
      })
      setSelectedEdge(null)
    }, '关系已更新')
  }

  const removeRelation = () => {
    if (!selectedEdge) return
    const ok = window.confirm(`确认删除关系“${selectedEdge.source} - ${selectedEdge.target}”？`)
    if (!ok) return
    return runAction(async () => {
      await deleteGraphRelation({
        workspace,
        source_entity: selectedEdge.source,
        target_entity: selectedEdge.target,
      })
      setSelectedEdge(null)
    }, '关系已删除')
  }

  const mergeEntities = () => runAction(async () => {
    await mergeGraphEntities({
      workspace,
      source_entities: mergeDraft.sources.split(',').map((item) => item.trim()).filter(Boolean),
      target_entity: mergeDraft.target,
      target_entity_data: {
        description: mergeDraft.description,
        entity_type: mergeDraft.type,
      },
    })
    setMergeDraft({ sources: '', target: '', description: '', type: '' })
  }, '实体已合并')

  const generateSuggestions = async () => {
    if (!suggestInstruction.trim()) return
    setSuggesting(true)
    setError('')
    setNotice('')
    try {
      const result = await suggestGraphChanges({
        workspace,
        instruction: suggestInstruction,
        limit,
      })
      setSuggestions(result.changes || [])
      setSelectedSuggestions(new Set())
      setNotice(`已生成 ${result.changes.length} 条候选建议`)
    } catch (e) {
      setError((e as Error).message || '生成建议失败')
    } finally {
      setSuggesting(false)
    }
  }

  const applySelectedSuggestions = () => {
    const changes = suggestions.filter((_, idx) => selectedSuggestions.has(idx))
    if (changes.length === 0) return
    const ok = window.confirm(`确认应用 ${changes.length} 条图谱变更？`)
    if (!ok) return
    return runAction(async () => {
      await applyGraphChanges(workspace, changes)
      setSuggestions([])
      setSelectedSuggestions(new Set())
    }, '已应用选中的图谱建议')
  }

  const renderOverview = () => (
    <div className="space-y-4">
      {latestEvidence && (latestEvidence.nodes.length > 0 || latestEvidence.edges.length > 0) && (
        <div className="flex items-center justify-between gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm">
          <div className="text-amber-800">
            已高亮最近一次问答证据链：
            <span className="font-semibold">{latestEvidence.nodes.length}</span> 个实体，
            <span className="font-semibold">{latestEvidence.edges.length}</span> 条关系。
          </div>
          <button
            onClick={() => {
              localStorage.removeItem(`tdx_latest_evidence_${workspace}`)
              setLatestEvidence(null)
            }}
            className="shrink-0 px-2.5 py-1 text-xs rounded-lg border border-amber-300 bg-white text-amber-700 hover:bg-amber-100"
          >
            清除高亮
          </button>
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          ['总节点', meta.total_nodes ?? 0],
          ['总关系', meta.total_edges ?? 0],
          ['当前展示节点', meta.returned_nodes ?? 0],
          ['当前展示关系', meta.returned_edges ?? 0],
        ].map(([label, value]) => (
          <div key={label} className="border border-gray-200 rounded-lg bg-white p-4">
            <div className="text-xs text-gray-400">{label}</div>
            <div className="text-2xl font-bold text-gray-800">{value}</div>
          </div>
        ))}
      </div>

      {loading ? (
        <div className="h-[560px] border border-gray-200 rounded-lg bg-white flex items-center justify-center text-sm text-gray-400">
          加载图谱...
        </div>
      ) : !data || data.nodes.length === 0 ? (
        <div className="h-[360px] border border-gray-200 rounded-lg bg-white flex items-center justify-center text-sm text-gray-400">
          暂无图谱数据，请先索引文档并等待 LightRAG 完成实体关系抽取。
        </div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px] gap-4">
          <div className="h-[620px] border border-gray-200 rounded-lg bg-white overflow-hidden">
            <GraphView
              nodes={data.nodes}
              edges={data.edges}
              hitNodes={evidenceNodeIds}
              pathNodes={evidenceNodeIds}
            />
          </div>
          <div className="border border-gray-200 rounded-lg bg-white overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-100">
              <h3 className="text-sm font-semibold text-gray-700">关系列表</h3>
              <p className="text-xs text-gray-400 mt-0.5">显示当前图中前 80 条关系</p>
            </div>
            <div className="max-h-[560px] overflow-y-auto divide-y divide-gray-100">
              {(data.edges || []).slice(0, 80).map((edge, i) => (
                <div key={`${edge.source}-${edge.target}-${i}`} className="p-3 text-xs">
                  <div className="font-medium text-gray-800 break-all">
                    {edge.source} <span className="text-gray-400">-</span> {edge.target}
                  </div>
                  <div className="mt-1 text-gray-500 leading-relaxed line-clamp-3">
                    {edge.relation}
                  </div>
                  {edge.file_path && (
                    <div className="mt-1 text-gray-400 truncate">{edge.file_path}</div>
                  )}
                </div>
              ))}
              {(data.edges || []).length === 0 && (
                <div className="p-4 text-xs text-gray-400">当前节点没有可展示关系。</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )

  const renderRules = () => (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_360px] gap-4">
      <div className="space-y-4">
        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-800">当前索引抽取规则</h3>
              <p className="mt-1 text-xs text-gray-500">
                当前知识库使用：{config.rule_template_name || '未命名规则'}。
                后续上传和重新索引会按这里的规则引导 LightRAG 抽取实体关系。
              </p>
            </div>
            <div className="shrink-0 rounded-lg bg-gray-50 px-3 py-2 text-right text-xs text-gray-500">
              <div>实体类型 {splitLines(entityTypesText).length}</div>
              <div>关系类型 {splitLines(relationTypesText).length}</div>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_auto] gap-3">
            <select
              value={selectedTemplateId}
              onChange={(e) => setSelectedTemplateId(e.target.value)}
              className="rounded-lg border border-gray-200 px-3 py-2 text-sm bg-white"
            >
              <option value="">选择抽取规则模板</option>
              {ruleTemplates.map((template) => (
                <option key={template.id} value={template.id}>
                  {template.name}{template.built_in ? ' / 内置' : ''}
                </option>
              ))}
            </select>
            <button
              onClick={applySelectedTemplate}
              disabled={!selectedTemplateId || saving}
              className="px-4 py-2 rounded-lg bg-gray-900 text-white text-sm hover:bg-gray-700 disabled:bg-gray-300"
            >
              套用模板
            </button>
          </div>

          <div className="mt-3 grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_auto] gap-3">
            <input
              value={templateName}
              onChange={(e) => setTemplateName(e.target.value)}
              className="rounded-lg border border-gray-200 px-3 py-2 text-sm"
              placeholder="将当前规则另存为模板名称"
            />
            <button
              onClick={saveCurrentAsTemplate}
              disabled={!templateName.trim() || saving}
              className="px-4 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50 disabled:text-gray-300"
            >
              另存为模板
            </button>
          </div>
        </div>

        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <label className="block text-sm font-semibold text-gray-800 mb-2">抽取提示词</label>
          <textarea
            value={extractionPrompt}
            onChange={(e) => setExtractionPrompt(e.target.value)}
            className="w-full min-h-40 rounded-lg border border-gray-200 px-3 py-2 text-sm leading-relaxed outline-none focus:border-gray-400"
          />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="border border-gray-200 rounded-lg bg-white p-4">
            <label className="block text-sm font-semibold text-gray-800 mb-2">实体类型白名单</label>
            <textarea
              value={entityTypesText}
              onChange={(e) => setEntityTypesText(e.target.value)}
              className="w-full min-h-48 rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-gray-400"
              placeholder="每行一个实体类型"
            />
          </div>
          <div className="border border-gray-200 rounded-lg bg-white p-4">
            <label className="block text-sm font-semibold text-gray-800 mb-2">关系类型白名单</label>
            <textarea
              value={relationTypesText}
              onChange={(e) => setRelationTypesText(e.target.value)}
              className="w-full min-h-48 rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-gray-400"
              placeholder="每行一个关系类型"
            />
          </div>
        </div>
        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <label className="block text-sm font-semibold text-gray-800 mb-2">术语和别名表</label>
          <textarea
            value={aliasesText}
            onChange={(e) => setAliasesText(e.target.value)}
            className="w-full min-h-44 rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-gray-400"
            placeholder="例如：infhost = 资讯主机"
          />
        </div>
        <button
          onClick={saveRules}
          disabled={saving}
          className="px-4 py-2 rounded-lg bg-gray-900 text-white text-sm hover:bg-gray-700 disabled:bg-gray-300"
        >
          保存抽取规则
        </button>
      </div>
      <div className="space-y-4">
        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">规则模板库</h3>
          <div className="mt-3 max-h-72 overflow-y-auto divide-y divide-gray-100">
            {ruleTemplates.map((template) => (
              <div key={template.id} className="py-3 text-xs">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="font-medium text-gray-800">
                      {template.name}
                      {template.built_in && <span className="ml-2 text-[10px] text-gray-400">内置</span>}
                    </div>
                    <div className="mt-1 text-gray-500 leading-relaxed">{template.description || '暂无说明'}</div>
                    <div className="mt-1 text-gray-400">
                      实体 {template.entity_types.length} · 关系 {template.relation_types.length}
                    </div>
                  </div>
                  {!template.built_in && (
                    <button
                      onClick={() => removeTemplate(template)}
                      className="shrink-0 text-red-500 hover:text-red-700"
                    >
                      删除
                    </button>
                  )}
                </div>
              </div>
            ))}
            {ruleTemplates.length === 0 && (
              <div className="py-6 text-xs text-gray-400">暂无规则模板</div>
            )}
          </div>
        </div>

        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">抽取参考文件</h3>
          <p className="mt-1 text-xs text-gray-500">
            仅用于辅助实体关系审校和建议生成，不进入普通问答召回。
          </p>
          <input
            type="file"
            accept=".txt,.md,.json,.yaml,.yml,.csv"
            onChange={(e) => handleReferenceUpload(e.target.files?.[0] || null)}
            className="mt-3 block w-full text-xs text-gray-600"
          />
          <div className="mt-4 divide-y divide-gray-100">
            {(config.reference_files || []).map((file) => (
              <div key={file.id} className="py-3 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="truncate font-medium text-gray-800">{file.file_name}</div>
                    <div className="mt-0.5 text-gray-400">
                      {file.char_count} 字符 · {formatTime(file.created_at)}
                    </div>
                  </div>
                  <button
                    onClick={() => runAction(() => deleteGraphReference(file.id, workspace), '参考文件已删除')}
                    className="shrink-0 text-red-500 hover:text-red-700"
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
            {(config.reference_files || []).length === 0 && (
              <div className="py-6 text-xs text-gray-400">暂无参考文件</div>
            )}
          </div>
        </div>
        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">最近修改</h3>
          <div className="mt-3 max-h-72 overflow-y-auto divide-y divide-gray-100">
            {(config.audit_log || []).slice(0, 12).map((item) => (
              <div key={item.id} className="py-2 text-xs">
                <div className="font-medium text-gray-800">{item.action}</div>
                <div className="text-gray-400">{formatTime(item.created_at)}</div>
              </div>
            ))}
            {(config.audit_log || []).length === 0 && (
              <div className="py-6 text-xs text-gray-400">暂无图谱修改记录</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )

  const renderEntities = () => (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-4">
      <div className="border border-gray-200 rounded-lg bg-white overflow-hidden">
        <div className="p-4 border-b border-gray-100">
          <input
            value={entityQuery}
            onChange={(e) => setEntityQuery(e.target.value)}
            placeholder="搜索实体名称、类型或描述"
            className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-gray-400"
          />
        </div>
        <div className="max-h-[680px] overflow-y-auto divide-y divide-gray-100">
          {filteredNodes.map((node) => (
            <button
              key={node.id}
              onClick={() => selectNode(node)}
              className={`w-full p-3 text-left hover:bg-gray-50 ${
                selectedNode?.id === node.id ? 'bg-gray-100' : ''
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="font-medium text-sm text-gray-800 truncate">{node.label || node.id}</div>
                <span className="shrink-0 rounded bg-gray-100 px-2 py-0.5 text-[11px] text-gray-500">
                  {node.entity_type || 'entity'}
                </span>
              </div>
              <div className="mt-1 text-xs text-gray-500 line-clamp-2">{node.description}</div>
            </button>
          ))}
          {filteredNodes.length === 0 && (
            <div className="p-6 text-sm text-gray-400">没有匹配实体</div>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">编辑实体</h3>
          {selectedNode ? (
            <div className="mt-3 space-y-3">
              <input
                value={entityEdit.name}
                onChange={(e) => setEntityEdit({ ...entityEdit, name: e.target.value })}
                className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm"
                placeholder="实体名称"
              />
              <input
                value={entityEdit.type}
                onChange={(e) => setEntityEdit({ ...entityEdit, type: e.target.value })}
                className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm"
                placeholder="实体类型"
              />
              <textarea
                value={entityEdit.description}
                onChange={(e) => setEntityEdit({ ...entityEdit, description: e.target.value })}
                className="w-full min-h-32 rounded-lg border border-gray-200 px-3 py-2 text-sm"
                placeholder="实体描述"
              />
              <div className="flex gap-2">
                <button onClick={saveEntity} className="px-3 py-2 rounded-lg bg-gray-900 text-white text-sm">
                  保存
                </button>
                <button onClick={removeEntity} className="px-3 py-2 rounded-lg border border-red-200 text-red-600 text-sm">
                  删除
                </button>
              </div>
            </div>
          ) : (
            <div className="mt-4 text-sm text-gray-400">从左侧选择一个实体进行编辑。</div>
          )}
        </div>

        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">新增实体</h3>
          <div className="mt-3 space-y-3">
            <input value={entityCreate.name} onChange={(e) => setEntityCreate({ ...entityCreate, name: e.target.value })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="实体名称" />
            <input value={entityCreate.type} onChange={(e) => setEntityCreate({ ...entityCreate, type: e.target.value })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="实体类型" />
            <textarea value={entityCreate.description} onChange={(e) => setEntityCreate({ ...entityCreate, description: e.target.value })} className="w-full min-h-24 rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="实体描述" />
            <button onClick={addEntity} disabled={!entityCreate.name.trim()} className="px-3 py-2 rounded-lg bg-gray-900 text-white text-sm disabled:bg-gray-300">
              新增实体
            </button>
          </div>
        </div>

        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">合并实体</h3>
          <div className="mt-3 space-y-3">
            <input value={mergeDraft.sources} onChange={(e) => setMergeDraft({ ...mergeDraft, sources: e.target.value })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="源实体，英文逗号分隔" />
            <input value={mergeDraft.target} onChange={(e) => setMergeDraft({ ...mergeDraft, target: e.target.value })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="合并目标实体" />
            <input value={mergeDraft.type} onChange={(e) => setMergeDraft({ ...mergeDraft, type: e.target.value })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="目标实体类型，可选" />
            <textarea value={mergeDraft.description} onChange={(e) => setMergeDraft({ ...mergeDraft, description: e.target.value })} className="w-full min-h-20 rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="目标实体描述，可选" />
            <button onClick={mergeEntities} disabled={!mergeDraft.sources.trim() || !mergeDraft.target.trim()} className="px-3 py-2 rounded-lg bg-gray-900 text-white text-sm disabled:bg-gray-300">
              合并实体
            </button>
          </div>
        </div>
      </div>
    </div>
  )

  const renderRelations = () => (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-4">
      <div className="border border-gray-200 rounded-lg bg-white overflow-hidden">
        <div className="p-4 border-b border-gray-100">
          <input
            value={relationQuery}
            onChange={(e) => setRelationQuery(e.target.value)}
            placeholder="搜索关系实体、描述或关键词"
            className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-gray-400"
          />
        </div>
        <div className="max-h-[680px] overflow-y-auto divide-y divide-gray-100">
          {filteredEdges.map((edge, idx) => (
            <button
              key={`${edge.source}-${edge.target}-${idx}`}
              onClick={() => selectEdge(edge)}
              className={`w-full p-3 text-left hover:bg-gray-50 ${
                selectedEdge === edge ? 'bg-gray-100' : ''
              }`}
            >
              <div className="font-medium text-sm text-gray-800 break-all">
                {edge.source} <span className="text-gray-400">-</span> {edge.target}
              </div>
              <div className="mt-1 text-xs text-gray-500 line-clamp-2">{edge.relation}</div>
              {edge.keywords && <div className="mt-1 text-[11px] text-gray-400 truncate">{edge.keywords}</div>}
            </button>
          ))}
          {filteredEdges.length === 0 && (
            <div className="p-6 text-sm text-gray-400">没有匹配关系</div>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">编辑关系</h3>
          {selectedEdge ? (
            <div className="mt-3 space-y-3">
              <div className="rounded-lg bg-gray-50 px-3 py-2 text-xs text-gray-600 break-all">
                {selectedEdge.source} - {selectedEdge.target}
              </div>
              <textarea value={relationEdit.description} onChange={(e) => setRelationEdit({ ...relationEdit, description: e.target.value })} className="w-full min-h-28 rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="关系描述" />
              <input value={relationEdit.keywords} onChange={(e) => setRelationEdit({ ...relationEdit, keywords: e.target.value })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="关键词" />
              <input type="number" step="0.1" value={relationEdit.weight} onChange={(e) => setRelationEdit({ ...relationEdit, weight: Number(e.target.value) })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="权重" />
              <div className="flex gap-2">
                <button onClick={saveRelation} className="px-3 py-2 rounded-lg bg-gray-900 text-white text-sm">保存</button>
                <button onClick={removeRelation} className="px-3 py-2 rounded-lg border border-red-200 text-red-600 text-sm">删除</button>
              </div>
            </div>
          ) : (
            <div className="mt-4 text-sm text-gray-400">从左侧选择一条关系进行编辑。</div>
          )}
        </div>

        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">新增关系</h3>
          <div className="mt-3 space-y-3">
            <input value={relationCreate.source} onChange={(e) => setRelationCreate({ ...relationCreate, source: e.target.value })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="起点实体" />
            <input value={relationCreate.target} onChange={(e) => setRelationCreate({ ...relationCreate, target: e.target.value })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="终点实体" />
            <textarea value={relationCreate.description} onChange={(e) => setRelationCreate({ ...relationCreate, description: e.target.value })} className="w-full min-h-24 rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="关系描述" />
            <input value={relationCreate.keywords} onChange={(e) => setRelationCreate({ ...relationCreate, keywords: e.target.value })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="关键词" />
            <input type="number" step="0.1" value={relationCreate.weight} onChange={(e) => setRelationCreate({ ...relationCreate, weight: Number(e.target.value) })} className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm" placeholder="权重" />
            <button onClick={addRelation} disabled={!relationCreate.source.trim() || !relationCreate.target.trim()} className="px-3 py-2 rounded-lg bg-gray-900 text-white text-sm disabled:bg-gray-300">
              新增关系
            </button>
          </div>
        </div>
      </div>
    </div>
  )

  const renderSuggestions = () => (
    <div className="space-y-4">
      <div className="border border-gray-200 rounded-lg bg-white p-4">
        <label className="block text-sm font-semibold text-gray-800">修正指令</label>
        <textarea
          value={suggestInstruction}
          onChange={(e) => setSuggestInstruction(e.target.value)}
          className="mt-3 w-full min-h-28 rounded-lg border border-gray-200 px-3 py-2 text-sm leading-relaxed outline-none focus:border-gray-400"
          placeholder="例如：根据参考文件检查 infhost、uts、zxdbtools 的实体类型和数据流向关系，删除无意义编号实体。"
        />
        <div className="mt-3 flex items-center gap-2">
          <button
            onClick={generateSuggestions}
            disabled={suggesting || !suggestInstruction.trim()}
            className="px-4 py-2 rounded-lg bg-gray-900 text-white text-sm hover:bg-gray-700 disabled:bg-gray-300"
          >
            {suggesting ? '生成中...' : '生成修正建议'}
          </button>
          <button
            onClick={applySelectedSuggestions}
            disabled={selectedSuggestions.size === 0}
            className="px-4 py-2 rounded-lg border border-gray-300 text-sm text-gray-700 hover:bg-gray-50 disabled:text-gray-300"
          >
            应用选中建议
          </button>
        </div>
      </div>

      <div className="border border-gray-200 rounded-lg bg-white overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800">候选变更</h3>
          <span className="text-xs text-gray-400">{suggestions.length} 条</span>
        </div>
        <div className="divide-y divide-gray-100">
          {suggestions.map((change, idx) => (
            <label key={idx} className="flex gap-3 p-4 hover:bg-gray-50">
              <input
                type="checkbox"
                checked={selectedSuggestions.has(idx)}
                onChange={(e) => {
                  setSelectedSuggestions((prev) => {
                    const next = new Set(prev)
                    if (e.target.checked) next.add(idx)
                    else next.delete(idx)
                    return next
                  })
                }}
                className="mt-1 h-4 w-4"
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="rounded bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
                    {changeTitle(change)}
                  </span>
                  <span className="text-xs text-gray-400">
                    {change.entity_name || change.source_entity || change.target_entity || change.source_entities?.join(', ')}
                  </span>
                </div>
                {change.reason && <p className="mt-2 text-sm text-gray-600">{change.reason}</p>}
                <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-gray-50 p-3 text-xs text-gray-600">
                  {JSON.stringify(change, null, 2)}
                </pre>
              </div>
            </label>
          ))}
          {suggestions.length === 0 && (
            <div className="p-8 text-sm text-gray-400">
              暂无候选变更。输入修正指令后生成建议，确认后再应用。
            </div>
          )}
        </div>
      </div>
    </div>
  )

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-gray-800">知识图谱</h2>
          <p className="text-sm text-gray-500 mt-1">
            图谱浏览、抽取规则、参考资料和实体关系审校。当前知识库: {workspace}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={limit}
            onChange={(e) => {
              const next = Number(e.target.value)
              setLimit(next)
              loadAll(next)
            }}
            className="border border-gray-200 rounded-lg px-2 py-1.5 text-xs bg-white"
          >
            <option value={100}>100 节点</option>
            <option value={200}>200 节点</option>
            <option value={500}>500 节点</option>
            <option value={1000}>1000 节点</option>
          </select>
          <button
            onClick={() => loadAll()}
            className="px-3 py-1.5 text-xs rounded-lg bg-primary-500 text-white hover:bg-primary-600"
          >
            刷新
          </button>
        </div>
      </div>

      <div className="border-b border-gray-200">
        <div className="flex gap-1 overflow-x-auto">
          {TABS.map((item) => (
            <button
              key={item.key}
              onClick={() => setTab(item.key)}
              className={`px-3 py-2 text-sm border-b-2 transition-colors ${
                tab === item.key
                  ? 'border-gray-900 text-gray-950 font-medium'
                  : 'border-transparent text-gray-500 hover:text-gray-800'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg border border-red-200 bg-red-50 text-sm text-red-700">
          {error}
        </div>
      )}
      {notice && (
        <div className="p-4 rounded-lg border border-emerald-200 bg-emerald-50 text-sm text-emerald-700">
          {notice}
        </div>
      )}

      {tab === 'overview' && renderOverview()}
      {tab === 'rules' && renderRules()}
      {tab === 'entities' && renderEntities()}
      {tab === 'relations' && renderRelations()}
      {tab === 'suggestions' && renderSuggestions()}
    </div>
  )
}
