import { useEffect, useMemo, useRef, useState } from 'react'
import { CheckCircle2, FileUp, Loader2 } from 'lucide-react'
import { useConfirm } from '../components/ConfirmDialog'
import EntityPicker from '../components/EntityPicker'
import GraphView from '../components/GraphView'
import {
  applyGraphChanges,
  applyGraphRuleTemplate,
  createGraphEntity,
  createGraphRelation,
  confirmGraphImport,
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
  GraphImportHistoryItem,
  GraphImportPreview,
  GraphNode,
  GraphRuleTemplate,
  listGraphImports,
  listGraphRuleTemplates,
  mergeGraphEntities,
  previewGraphImport,
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

type Tab = 'overview' | 'rules' | 'import' | 'entities' | 'relations' | 'suggestions'

const TABS: { key: Tab; label: string }[] = [
  { key: 'overview', label: '图谱总览' },
  { key: 'rules', label: '抽取规则' },
  { key: 'import', label: '图谱导入' },
  { key: 'entities', label: '实体治理' },
  { key: 'relations', label: '关系治理' },
  { key: 'suggestions', label: '修正建议' },
]

const emptyConfig: GraphGovernanceConfig = {
  workspace: '',
  rule_template_id: '',
  rule_template_name: '',
  extraction_mode: 'assist',
  allow_other_entity_type: true,
  entity_types: [],
  relation_types: [],
  aliases_text: '',
  extraction_prompt: '',
  effective_extraction_prompt: '',
  reference_files: [],
  updated_at: '',
  audit_log: [],
}

const EXTRACTION_MODES: {
  key: GraphGovernanceConfig['extraction_mode']
  label: string
  hint: string
}[] = [
  { key: 'assist', label: '辅助', hint: '通用抽取优先，规则只做归类和纠偏' },
  { key: 'enhanced', label: '增强', hint: '优先使用领域规则，但不会压制重要实体' },
  { key: 'strict', label: '严格', hint: '类型和关系接近白名单，只适合规范业务库' },
]

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
  const loadRequestRef = useRef(0)
  const loadAbortRef = useRef<AbortController | null>(null)
  const confirm = useConfirm()
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
  const [graphImportFile, setGraphImportFile] = useState<File | null>(null)
  const [graphImportPreview, setGraphImportPreview] = useState<GraphImportPreview | null>(null)
  const [selectedImportEntities, setSelectedImportEntities] = useState<Set<number>>(new Set())
  const [selectedImportRelationships, setSelectedImportRelationships] = useState<Set<number>>(new Set())
  const [graphImportHistory, setGraphImportHistory] = useState<GraphImportHistoryItem[]>([])
  const [previewingImport, setPreviewingImport] = useState(false)
  const [applyingImport, setApplyingImport] = useState(false)

  const [entityTypesText, setEntityTypesText] = useState('')
  const [relationTypesText, setRelationTypesText] = useState('')
  const [aliasesText, setAliasesText] = useState('')
  const [extractionPrompt, setExtractionPrompt] = useState('')
  const [extractionMode, setExtractionMode] = useState<GraphGovernanceConfig['extraction_mode']>('assist')
  const [allowOtherEntityType, setAllowOtherEntityType] = useState(true)
  const [showEffectivePrompt, setShowEffectivePrompt] = useState(false)

  const [entityQuery, setEntityQuery] = useState('')
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [entityEdit, setEntityEdit] = useState({ name: '', type: '', description: '' })
  const [entityCreate, setEntityCreate] = useState({ name: '', type: '', description: '' })
  const [mergeDraft, setMergeDraft] = useState({
    sources: [] as string[],
    target: '',
    description: '',
    type: '',
  })

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

  const loadGraph = async (
    nextLimit = limit,
    requestId = loadRequestRef.current,
    signal?: AbortSignal,
  ) => {
    const graph = await getGraph(nextLimit, workspace, signal)
    if (requestId !== loadRequestRef.current) return
    setData(graph)
  }

  const loadConfig = async (requestId = loadRequestRef.current, signal?: AbortSignal) => {
    const cfg = await getGraphGovernanceConfig(workspace, signal)
    if (requestId !== loadRequestRef.current) return
    setConfig(cfg)
    setSelectedTemplateId(cfg.rule_template_id || '')
    setEntityTypesText(joinLines(cfg.entity_types || []))
    setRelationTypesText(joinLines(cfg.relation_types || []))
    setAliasesText(cfg.aliases_text || '')
    setExtractionPrompt(cfg.extraction_prompt || '')
    setExtractionMode(cfg.extraction_mode || 'assist')
    setAllowOtherEntityType(cfg.allow_other_entity_type !== false)
  }

  const loadTemplates = async (requestId = loadRequestRef.current, signal?: AbortSignal) => {
    const templates = await listGraphRuleTemplates(workspace, signal)
    if (requestId !== loadRequestRef.current) return
    setRuleTemplates(templates)
  }

  const loadImportHistory = async (requestId = loadRequestRef.current, signal?: AbortSignal) => {
    const history = await listGraphImports(workspace, signal)
    if (requestId !== loadRequestRef.current) return
    setGraphImportHistory(history)
  }

  const loadAll = async (nextLimit = limit) => {
    loadAbortRef.current?.abort()
    const controller = new AbortController()
    loadAbortRef.current = controller
    const requestId = ++loadRequestRef.current
    setLoading(true)
    setError('')
    try {
      await Promise.all([
        loadGraph(nextLimit, requestId, controller.signal),
        loadConfig(requestId, controller.signal),
        loadTemplates(requestId, controller.signal),
        loadImportHistory(requestId, controller.signal),
      ])
      if (requestId !== loadRequestRef.current) return
    } catch (e) {
      if (requestId !== loadRequestRef.current) return
      if (controller.signal.aborted) return
      setError((e as Error).message || '加载失败')
    } finally {
      if (requestId === loadRequestRef.current) setLoading(false)
    }
  }

  useEffect(() => {
    loadRequestRef.current += 1
    setGraphImportFile(null)
    setGraphImportPreview(null)
    setSelectedImportEntities(new Set())
    setSelectedImportRelationships(new Set())
    loadAll(limit)
    return () => loadAbortRef.current?.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace])

  useEffect(() => {
    try {
      const raw = localStorage.getItem(`lightgraphrag_latest_evidence_${workspace}`)
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
      extraction_mode: extractionMode,
      allow_other_entity_type: allowOtherEntityType,
      entity_types: splitLines(entityTypesText),
      relation_types: splitLines(relationTypesText),
      aliases_text: aliasesText,
      extraction_prompt: extractionPrompt,
    })
    setConfig(cfg)
  }, '抽取规则已保存')

  const applySelectedTemplate = async () => {
    if (!selectedTemplateId) return
    const selected = ruleTemplates.find((item) => item.id === selectedTemplateId)
    const ok = await confirm({
      title: '套用抽取规则',
      message: `将“${selected?.name || selectedTemplateId}”套用到知识库“${workspace}”。新规则只影响后续上传和重新索引的文档。`,
      confirmLabel: '套用规则',
    })
    if (!ok) return
    return runAction(async () => {
      const cfg = await applyGraphRuleTemplate(workspace, selectedTemplateId)
      setConfig(cfg)
      setSelectedTemplateId(cfg.rule_template_id || '')
      setEntityTypesText(joinLines(cfg.entity_types || []))
      setRelationTypesText(joinLines(cfg.relation_types || []))
      setAliasesText(cfg.aliases_text || '')
      setExtractionPrompt(cfg.extraction_prompt || '')
      setExtractionMode(cfg.extraction_mode || 'assist')
      setAllowOtherEntityType(cfg.allow_other_entity_type !== false)
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

  const removeTemplate = async (template: GraphRuleTemplate) => {
    if (template.built_in) return
    const ok = await confirm({
      title: '删除抽取规则模板',
      message: `将删除模板“${template.name}”，不会影响已经套用到知识库的规则内容。`,
      confirmLabel: '删除模板',
      tone: 'danger',
    })
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

  const removeEntity = async () => {
    if (!selectedNode) return
    const ok = await confirm({
      title: '删除图谱实体',
      message: `将删除实体“${selectedNode.id}”以及与它连接的所有关系。`,
      confirmLabel: '删除实体',
      tone: 'danger',
    })
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

  const removeRelation = async () => {
    if (!selectedEdge) return
    const ok = await confirm({
      title: '删除图谱关系',
      message: `将删除“${selectedEdge.source}”到“${selectedEdge.target}”的关系。`,
      confirmLabel: '删除关系',
      tone: 'danger',
    })
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
      source_entities: mergeDraft.sources,
      target_entity: mergeDraft.target,
      target_entity_data: {
        description: mergeDraft.description,
        entity_type: mergeDraft.type,
      },
    })
    setMergeDraft({ sources: [], target: '', description: '', type: '' })
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

  const applySelectedSuggestions = async () => {
    const changes = suggestions.filter((_, idx) => selectedSuggestions.has(idx))
    if (changes.length === 0) return
    const ok = await confirm({
      title: '应用图谱修正',
      message: `将对知识库“${workspace}”应用 ${changes.length} 条图谱变更。`,
      confirmLabel: '应用变更',
    })
    if (!ok) return
    return runAction(async () => {
      await applyGraphChanges(workspace, changes)
      setSuggestions([])
      setSelectedSuggestions(new Set())
    }, '已应用选中的图谱建议')
  }

  const runGraphImportPreview = async () => {
    if (!graphImportFile) return
    setPreviewingImport(true)
    setError('')
    setNotice('')
    try {
      const preview = await previewGraphImport(graphImportFile, workspace)
      setGraphImportPreview(preview)
      setSelectedImportEntities(new Set(preview.entities.map((_, index) => index)))
      setSelectedImportRelationships(new Set(preview.relationships.map((_, index) => index)))
      setNotice(
        `已识别 ${preview.entities.length} 个实体和 ${preview.relationships.length} 条关系，请审阅后确认`,
      )
    } catch (e) {
      setGraphImportPreview(null)
      setError((e as Error).message || '图谱资料解析失败')
    } finally {
      setPreviewingImport(false)
    }
  }

  const applyReviewedGraphImport = async () => {
    if (!graphImportPreview) return
    const entities = graphImportPreview.entities.filter((_, index) =>
      selectedImportEntities.has(index),
    )
    const relationships = graphImportPreview.relationships.filter((_, index) =>
      selectedImportRelationships.has(index),
    )
    if (entities.length === 0 && relationships.length === 0) return
    const ok = await confirm({
      title: '确认导入图谱',
      message: `将导入 ${entities.length} 个实体和 ${relationships.length} 条关系。专用资料不会进入普通文本向量召回。`,
      confirmLabel: '确认导入',
    })
    if (!ok) return
    setApplyingImport(true)
    setError('')
    try {
      const result = await confirmGraphImport({
        workspace,
        file_name: graphImportPreview.file_name,
        source_text: graphImportPreview.source_text,
        entities,
        relationships,
      })
      setNotice(
        `图谱导入完成：${result.entity_count} 个实体，${result.relationship_count} 条关系`,
      )
      setGraphImportFile(null)
      setGraphImportPreview(null)
      setSelectedImportEntities(new Set())
      setSelectedImportRelationships(new Set())
      await Promise.all([loadGraph(limit), loadImportHistory()])
    } catch (e) {
      setError((e as Error).message || '图谱导入失败')
    } finally {
      setApplyingImport(false)
    }
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
              localStorage.removeItem(`lightgraphrag_latest_evidence_${workspace}`)
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
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div className="space-y-4">
        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-800">当前索引抽取规则</h3>
              <p className="mt-1 text-xs text-gray-500">
                当前知识库使用：{config.rule_template_name || '未命名规则'}。
                后续上传和重新索引会按这里的偏好引导 LightRAG 抽取实体关系。
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
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-800">抽取模式</h3>
              <p className="mt-1 text-xs text-gray-500">
                规则默认只是辅助。只有切到严格模式时，实体类型和关系类型才接近白名单。
              </p>
            </div>
            <label className="shrink-0 flex items-center gap-2 text-xs text-gray-600">
              <input
                type="checkbox"
                checked={allowOtherEntityType}
                onChange={(e) => setAllowOtherEntityType(e.target.checked)}
                className="h-4 w-4 rounded accent-gray-900"
              />
              允许 Other 类型
            </label>
          </div>

          <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-2">
            {EXTRACTION_MODES.map((mode) => (
              <button
                key={mode.key}
                onClick={() => setExtractionMode(mode.key)}
                className={`rounded-lg border px-3 py-2 text-left transition-colors ${
                  extractionMode === mode.key
                    ? 'border-gray-900 bg-gray-900 text-white'
                    : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                <div className="text-sm font-medium">{mode.label}</div>
                <div className={`mt-1 text-xs leading-relaxed ${
                  extractionMode === mode.key ? 'text-gray-200' : 'text-gray-500'
                }`}>
                  {mode.hint}
                </div>
              </button>
            ))}
          </div>
          {extractionMode === 'strict' && (
            <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
              严格模式可能导致非模板领域文档抽不出实体关系。普通知识库建议使用辅助或增强模式。
            </div>
          )}
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
            <label className="block text-sm font-semibold text-gray-800 mb-2">实体类型偏好</label>
            <textarea
              value={entityTypesText}
              onChange={(e) => setEntityTypesText(e.target.value)}
              className="w-full min-h-48 rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-gray-400"
              placeholder="每行一个实体类型；辅助模式下不是硬白名单"
            />
          </div>
          <div className="border border-gray-200 rounded-lg bg-white p-4">
            <label className="block text-sm font-semibold text-gray-800 mb-2">关系类型偏好</label>
            <textarea
              value={relationTypesText}
              onChange={(e) => setRelationTypesText(e.target.value)}
              className="w-full min-h-48 rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-gray-400"
              placeholder="每行一个关系类型；辅助模式下不是硬白名单"
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
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-gray-800">最终送入 LightRAG 的抽取提示</h3>
              <p className="mt-1 text-xs text-gray-500">
                这是基础通用规则、抽取模式、类型偏好、提示词和参考资料拼接后的结果。
              </p>
            </div>
            <button
              onClick={() => setShowEffectivePrompt((v) => !v)}
              className="shrink-0 rounded-lg border border-gray-300 px-2.5 py-1.5 text-xs text-gray-700 hover:bg-gray-50"
            >
              {showEffectivePrompt ? '收起' : '查看'}
            </button>
          </div>
          {showEffectivePrompt && (
            <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap rounded-lg bg-gray-50 p-3 text-xs leading-relaxed text-gray-700">
              {config.effective_extraction_prompt || '暂无有效抽取提示'}
            </pre>
          )}
        </div>

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

  const renderGraphImport = () => (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <div className="flex items-start gap-3">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-violet-50 text-violet-700">
              <FileUp size={18} />
            </span>
            <div>
              <h3 className="text-sm font-semibold text-gray-800">导入实体关系资料</h3>
              <p className="mt-1 text-xs leading-5 text-gray-500">
                支持 TXT、Markdown、JSON、YAML 和 CSV。系统先生成候选项，只有人工确认的内容才会写入图谱。
              </p>
            </div>
          </div>
          <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
            <label className="ui-button-secondary cursor-pointer">
              选择资料
              <input
                type="file"
                accept=".txt,.md,.json,.yaml,.yml,.csv"
                className="sr-only"
                onChange={(event) => {
                  setGraphImportFile(event.target.files?.[0] || null)
                  setGraphImportPreview(null)
                }}
              />
            </label>
            <span className="min-w-0 flex-1 truncate text-sm text-gray-600">
              {graphImportFile?.name || '尚未选择文件'}
            </span>
            <button
              type="button"
              onClick={runGraphImportPreview}
              disabled={!graphImportFile || previewingImport}
              className="ui-button-primary inline-flex items-center justify-center gap-2"
            >
              {previewingImport && <Loader2 size={15} className="animate-spin" />}
              {previewingImport ? '正在分析' : '生成候选'}
            </button>
          </div>
          <p className="mt-3 text-xs text-gray-400">
            结构化 JSON 会直接校验；其他格式使用当前知识库的 KG 模型和抽取规则分析。
          </p>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">最近导入</h3>
          <div className="mt-3 max-h-36 divide-y divide-gray-100 overflow-y-auto">
            {graphImportHistory.map((item) => (
              <div key={item.import_id} className="py-2 text-xs">
                <div className="truncate font-medium text-gray-700">{item.file_name}</div>
                <div className="mt-0.5 text-gray-400">
                  {item.entity_count} 实体 · {item.relationship_count} 关系 · {formatTime(item.created_at)}
                </div>
              </div>
            ))}
            {graphImportHistory.length === 0 && (
              <div className="py-5 text-xs text-gray-400">暂无已确认导入记录</div>
            )}
          </div>
        </div>
      </div>

      {graphImportPreview && (
        <>
          {graphImportPreview.warnings.length > 0 && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800">
              {graphImportPreview.warnings.join('；')}
            </div>
          )}

          <div className="rounded-lg border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3">
              <div>
                <h3 className="text-sm font-semibold text-gray-800">候选实体</h3>
                <p className="mt-0.5 text-xs text-gray-400">
                  已选 {selectedImportEntities.size}/{graphImportPreview.entities.length}
                </p>
              </div>
              <label className="flex items-center gap-2 text-xs text-gray-500">
                <input
                  type="checkbox"
                  checked={
                    graphImportPreview.entities.length > 0 &&
                    selectedImportEntities.size === graphImportPreview.entities.length
                  }
                  onChange={(event) =>
                    setSelectedImportEntities(
                      event.target.checked
                        ? new Set(graphImportPreview.entities.map((_, index) => index))
                        : new Set(),
                    )
                  }
                />
                全选
              </label>
            </div>
            <div className="max-h-[420px] divide-y divide-gray-100 overflow-y-auto">
              {graphImportPreview.entities.map((entity, index) => (
                <div key={`${entity.entity_name}-${index}`} className="grid gap-3 p-4 md:grid-cols-[24px_180px_140px_minmax(0,1fr)]">
                  <input
                    type="checkbox"
                    checked={selectedImportEntities.has(index)}
                    onChange={(event) => {
                      setSelectedImportEntities((current) => {
                        const next = new Set(current)
                        if (event.target.checked) next.add(index)
                        else next.delete(index)
                        return next
                      })
                    }}
                    className="mt-2"
                  />
                  <input
                    value={entity.entity_name}
                    onChange={(event) =>
                      setGraphImportPreview((current) => current ? {
                        ...current,
                        entities: current.entities.map((item, itemIndex) =>
                          itemIndex === index ? { ...item, entity_name: event.target.value } : item,
                        ),
                      } : current)
                    }
                    className="ui-control w-full"
                    aria-label="实体名称"
                  />
                  <input
                    value={entity.entity_type}
                    onChange={(event) =>
                      setGraphImportPreview((current) => current ? {
                        ...current,
                        entities: current.entities.map((item, itemIndex) =>
                          itemIndex === index ? { ...item, entity_type: event.target.value } : item,
                        ),
                      } : current)
                    }
                    className="ui-control w-full"
                    aria-label="实体类型"
                  />
                  <input
                    value={entity.description}
                    onChange={(event) =>
                      setGraphImportPreview((current) => current ? {
                        ...current,
                        entities: current.entities.map((item, itemIndex) =>
                          itemIndex === index ? { ...item, description: event.target.value } : item,
                        ),
                      } : current)
                    }
                    className="ui-control w-full"
                    aria-label="实体描述"
                    placeholder="实体描述"
                  />
                </div>
              ))}
              {graphImportPreview.entities.length === 0 && (
                <div className="p-6 text-sm text-gray-400">未识别到实体候选</div>
              )}
            </div>
          </div>

          <div className="rounded-lg border border-gray-200 bg-white">
            <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3">
              <div>
                <h3 className="text-sm font-semibold text-gray-800">候选关系</h3>
                <p className="mt-0.5 text-xs text-gray-400">
                  已选 {selectedImportRelationships.size}/{graphImportPreview.relationships.length}
                </p>
              </div>
              <label className="flex items-center gap-2 text-xs text-gray-500">
                <input
                  type="checkbox"
                  checked={
                    graphImportPreview.relationships.length > 0 &&
                    selectedImportRelationships.size === graphImportPreview.relationships.length
                  }
                  onChange={(event) =>
                    setSelectedImportRelationships(
                      event.target.checked
                        ? new Set(graphImportPreview.relationships.map((_, index) => index))
                        : new Set(),
                    )
                  }
                />
                全选
              </label>
            </div>
            <div className="max-h-[420px] divide-y divide-gray-100 overflow-y-auto">
              {graphImportPreview.relationships.map((relation, index) => (
                <div key={`${relation.src_id}-${relation.tgt_id}-${index}`} className="grid gap-3 p-4 md:grid-cols-[24px_160px_160px_120px_minmax(0,1fr)]">
                  <input
                    type="checkbox"
                    checked={selectedImportRelationships.has(index)}
                    onChange={(event) => {
                      setSelectedImportRelationships((current) => {
                        const next = new Set(current)
                        if (event.target.checked) next.add(index)
                        else next.delete(index)
                        return next
                      })
                    }}
                    className="mt-2"
                  />
                  {(['src_id', 'tgt_id', 'relation_type', 'description'] as const).map((field) => (
                    <input
                      key={field}
                      value={relation[field]}
                      onChange={(event) =>
                        setGraphImportPreview((current) => current ? {
                          ...current,
                          relationships: current.relationships.map((item, itemIndex) =>
                            itemIndex === index ? { ...item, [field]: event.target.value } : item,
                          ),
                        } : current)
                      }
                      className="ui-control w-full"
                      aria-label={field}
                      placeholder={
                        field === 'src_id'
                          ? '起点实体'
                          : field === 'tgt_id'
                            ? '终点实体'
                            : field === 'relation_type'
                              ? '关系类型'
                              : '关系描述'
                      }
                    />
                  ))}
                </div>
              ))}
              {graphImportPreview.relationships.length === 0 && (
                <div className="p-6 text-sm text-gray-400">未识别到关系候选</div>
              )}
            </div>
          </div>

          <div className="sticky bottom-4 flex items-center justify-between rounded-lg border border-gray-200 bg-white/95 px-4 py-3 shadow-sm backdrop-blur">
            <span className="text-sm text-gray-600">
              将导入 {selectedImportEntities.size} 个实体和 {selectedImportRelationships.size} 条关系
            </span>
            <button
              type="button"
              onClick={applyReviewedGraphImport}
              disabled={
                applyingImport ||
                (selectedImportEntities.size === 0 && selectedImportRelationships.size === 0)
              }
              className="ui-button-primary inline-flex items-center gap-2"
            >
              {applyingImport
                ? <Loader2 size={15} className="animate-spin" />
                : <CheckCircle2 size={15} />}
              {applyingImport ? '正在导入' : '确认导入'}
            </button>
          </div>
        </>
      )}
    </div>
  )

  const renderEntities = () => (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
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
                className="ui-control w-full"
                placeholder="实体名称"
              />
              <input
                value={entityEdit.type}
                onChange={(e) => setEntityEdit({ ...entityEdit, type: e.target.value })}
                className="ui-control w-full"
                placeholder="实体类型"
              />
              <textarea
                value={entityEdit.description}
                onChange={(e) => setEntityEdit({ ...entityEdit, description: e.target.value })}
                className="ui-textarea min-h-32 w-full"
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
            <div className="grid grid-cols-[minmax(0,1fr)_130px] gap-3">
              <input value={entityCreate.name} onChange={(e) => setEntityCreate({ ...entityCreate, name: e.target.value })} className="ui-control w-full" placeholder="实体名称" />
              <input value={entityCreate.type} onChange={(e) => setEntityCreate({ ...entityCreate, type: e.target.value })} className="ui-control w-full" placeholder="实体类型" />
            </div>
            <textarea value={entityCreate.description} onChange={(e) => setEntityCreate({ ...entityCreate, description: e.target.value })} className="ui-textarea min-h-24 w-full" placeholder="实体描述" />
            <button onClick={addEntity} disabled={!entityCreate.name.trim()} className="ui-button-primary">
              新增实体
            </button>
          </div>
        </div>

        <div className="border border-gray-200 rounded-lg bg-white p-4">
          <h3 className="text-sm font-semibold text-gray-800">合并实体</h3>
          <div className="mt-3 space-y-3">
            <label className="block">
              <span className="ui-label">被合并实体</span>
              <EntityPicker
                nodes={data?.nodes || []}
                value={mergeDraft.sources}
                onChange={(sources) => setMergeDraft({ ...mergeDraft, sources })}
                placeholder="搜索并复选一个或多个实体"
                multiple
                exclude={mergeDraft.target ? [mergeDraft.target] : []}
              />
            </label>
            <label className="block">
              <span className="ui-label">保留为目标实体</span>
              <EntityPicker
                nodes={data?.nodes || []}
                value={mergeDraft.target ? [mergeDraft.target] : []}
                onChange={(target) => setMergeDraft({ ...mergeDraft, target: target[0] || '' })}
                placeholder="搜索并选择目标实体"
                exclude={mergeDraft.sources}
              />
            </label>
            <input value={mergeDraft.type} onChange={(e) => setMergeDraft({ ...mergeDraft, type: e.target.value })} className="ui-control w-full" placeholder="目标实体类型，可选" />
            <textarea value={mergeDraft.description} onChange={(e) => setMergeDraft({ ...mergeDraft, description: e.target.value })} className="ui-textarea min-h-20 w-full" placeholder="目标实体描述，可选" />
            <button onClick={mergeEntities} disabled={mergeDraft.sources.length < 1 || !mergeDraft.target} className="ui-button-primary">
              合并实体
            </button>
          </div>
        </div>
      </div>
    </div>
  )

  const renderRelations = () => (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
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
              <textarea value={relationEdit.description} onChange={(e) => setRelationEdit({ ...relationEdit, description: e.target.value })} className="ui-textarea min-h-28 w-full" placeholder="关系描述" />
              <div className="grid grid-cols-[minmax(0,1fr)_96px] gap-3">
                <input value={relationEdit.keywords} onChange={(e) => setRelationEdit({ ...relationEdit, keywords: e.target.value })} className="ui-control w-full" placeholder="关键词" />
                <input type="number" step="0.1" value={relationEdit.weight} onChange={(e) => setRelationEdit({ ...relationEdit, weight: Number(e.target.value) })} className="ui-control w-full px-2 text-right" aria-label="关系权重" />
              </div>
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
            <label className="block">
              <span className="ui-label">起点实体</span>
              <EntityPicker
                nodes={data?.nodes || []}
                value={relationCreate.source ? [relationCreate.source] : []}
                onChange={(source) => setRelationCreate({ ...relationCreate, source: source[0] || '' })}
                placeholder="搜索并选择起点实体"
                exclude={relationCreate.target ? [relationCreate.target] : []}
              />
            </label>
            <label className="block">
              <span className="ui-label">终点实体</span>
              <EntityPicker
                nodes={data?.nodes || []}
                value={relationCreate.target ? [relationCreate.target] : []}
                onChange={(target) => setRelationCreate({ ...relationCreate, target: target[0] || '' })}
                placeholder="搜索并选择终点实体"
                exclude={relationCreate.source ? [relationCreate.source] : []}
              />
            </label>
            <textarea value={relationCreate.description} onChange={(e) => setRelationCreate({ ...relationCreate, description: e.target.value })} className="ui-textarea min-h-24 w-full" placeholder="关系描述" />
            <div className="grid grid-cols-[minmax(0,1fr)_96px] gap-3">
              <input value={relationCreate.keywords} onChange={(e) => setRelationCreate({ ...relationCreate, keywords: e.target.value })} className="ui-control w-full" placeholder="关键词" />
              <input type="number" step="0.1" value={relationCreate.weight} onChange={(e) => setRelationCreate({ ...relationCreate, weight: Number(e.target.value) })} className="ui-control w-full px-2 text-right" aria-label="关系权重" />
            </div>
            <button onClick={addRelation} disabled={!relationCreate.source || !relationCreate.target} className="ui-button-primary">
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
            图谱浏览、抽取规则、资料导入和实体关系审校。
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
            className="whitespace-nowrap px-3 py-1.5 text-xs rounded-lg bg-primary-500 text-white hover:bg-primary-600"
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
              className={`whitespace-nowrap px-3 py-2 text-sm border-b-2 transition-colors ${
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
      {tab === 'import' && renderGraphImport()}
      {tab === 'entities' && renderEntities()}
      {tab === 'relations' && renderRelations()}
      {tab === 'suggestions' && renderSuggestions()}
    </div>
  )
}
