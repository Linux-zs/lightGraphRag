import { Fragment, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import {
  Check,
  ChevronDown,
  Copy,
  FileText,
  Network,
  Send,
  Settings2,
  Sparkles,
  X,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import GraphView from '../components/GraphView'
import { useConfirm } from '../components/ConfirmDialog'
import { RangeField, SelectField, Toggle } from '../components/ui'
import WorkspaceSwitcher from '../components/WorkspaceSwitcher'
import {
  chatSendStream,
  ChatSettings,
  getChatSession,
  getDocumentChunks,
  getModelBindings,
  getModelConfig,
  listModelProfiles,
  ChatSessionListItem,
  ChatMessage,
  Citation,
  EvidenceChain,
  ModelProfile,
  updateChatSessionSettings,
  WorkspaceInfo,
} from '../api'

type CitationMap = Map<number, Citation[]>
type EvidenceMap = Map<number, EvidenceChain>

const DEFAULT_CHAT_SETTINGS: ChatSettings = {
  answer_profile_id: 'siliconflow-default',
  answer_model: 'Qwen/Qwen2.5-7B-Instruct',
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 4096,
  frequency_penalty: 0.3,
  presence_penalty: 0.2,
  mode: 'mix',
  top_k: 40,
  chunk_top_k: 20,
  enable_rerank: true,
}

function isLikelyChatModel(model: { id: string; type: string }) {
  const type = (model.type || '').toLowerCase()
  if (['chat', 'llm', 'text-generation', 'completion'].includes(type)) return true
  const nonChatType = /(embedding|rerank|image|video|audio|tts|asr|ocr|vision-only)/
  if (nonChatType.test(type)) return false
  return !/(embedding|rerank|bge-|image|kolors|wan2|tts|asr|ocr|sensevoice|cosyvoice)/i.test(model.id)
}

// --- Citation marker helpers (module-level pure functions) ---

const CITATION_RE = /\[(\d+)\]/g

/**
 * Split a text string into plain-text segments and clickable citation
 * superscript elements. Returns the original string unchanged when no
 * [数字] pattern is found (avoids unnecessary Fragment wrappers).
 */
function renderTextWithCitations(
  text: string,
  onCitationClick: (num: number) => void,
  isHighlighted: (num: number) => boolean,
  validCitationNumbers?: Set<number>,
): ReactNode {
  // Fast path: no citation markers in this text segment
  if (!/\[\d+\]/.test(text)) {
    return text
  }

  const parts: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  let keyIdx = 0
  CITATION_RE.lastIndex = 0

  while ((match = CITATION_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(
        <Fragment key={`txt-${keyIdx++}`}>
          {text.slice(lastIndex, match.index)}
        </Fragment>,
      )
    }
    const num = parseInt(match[1], 10)
    if (validCitationNumbers && !validCitationNumbers.has(num)) {
      lastIndex = match.index + match[0].length
      continue
    }
    parts.push(
      <sup
        key={`cite-${keyIdx++}`}
        onClick={(e) => {
          e.stopPropagation()
          onCitationClick(num)
        }}
        className={`inline-flex items-center justify-center min-w-[1.65em] h-[1.25em] px-1 mx-0.5 text-[0.62em] font-bold rounded-full cursor-pointer align-super transition-colors select-none tabular-nums ${
          isHighlighted(num)
            ? 'bg-amber-500 text-white ring-2 ring-amber-300'
            : 'bg-primary-500 text-white hover:bg-primary-600'
        }`}
      >
        [{num}]
      </sup>,
    )
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) {
    parts.push(
      <Fragment key={`txt-${keyIdx++}`}>
        {text.slice(lastIndex)}
      </Fragment>,
    )
  }

  return parts
}

/**
 * Recursively walk React children, replacing [数字] markers inside
 * string nodes with clickable superscript elements. Non-string children
 * (React elements) are returned as-is — they are handled by their own
 * component override if one is registered.
 */
function processChildren(
  children: ReactNode,
  onCitationClick: (num: number) => void,
  isHighlighted: (num: number) => boolean,
  validCitationNumbers?: Set<number>,
): ReactNode {
  if (children == null || typeof children === 'boolean') {
    return children
  }
  if (typeof children === 'string') {
    return renderTextWithCitations(children, onCitationClick, isHighlighted, validCitationNumbers)
  }
  if (typeof children === 'number') {
    return children
  }
  if (Array.isArray(children)) {
    return children.map((child, i) => {
      if (typeof child === 'string') {
        return (
          <Fragment key={`child-${i}`}>
            {renderTextWithCitations(child, onCitationClick, isHighlighted, validCitationNumbers)}
          </Fragment>
        )
      }
      return child
    })
  }
  return children
}

function normalizeCitations(citations: Citation[] | undefined): Citation[] {
  if (!Array.isArray(citations)) return []
  const seen = new Set<string>()
  return citations
    .filter((citation) => citation && typeof citation.index === 'number')
    .filter((citation) => {
      const key = `${citation.doc_name}|${citation.chunk_index}|${citation.excerpt}`
      if (seen.has(key)) return false
      seen.add(key)
      return true
    })
    .sort((a, b) => a.index - b.index)
}

function citationSummary(citations: Citation[]) {
  const docs = Array.from(new Set(citations.map((c) => c.doc_name).filter(Boolean)))
  const firstDoc = docs[0] || 'LightRAG'
  return {
    docCount: docs.length,
    chunkCount: citations.length,
    firstDoc,
  }
}

interface Props {
  workspace: string
  workspaces: WorkspaceInfo[]
  onWorkspaceChange: (workspace: string) => void
  sessions: ChatSessionListItem[]
  activeId: string | null
  setActiveId: (id: string | null) => void
  reloadSessions: () => Promise<void>
}

export default function QAChat({
  workspace,
  workspaces,
  onWorkspaceChange,
  sessions,
  activeId,
  setActiveId,
  reloadSessions,
}: Props) {
  const confirm = useConfirm()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [citationsByIndex, setCitationsByIndex] = useState<CitationMap>(new Map())
  const [evidenceByIndex, setEvidenceByIndex] = useState<EvidenceMap>(new Map())
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [loadingSession, setLoadingSession] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const streamingRef = useRef(false)
  const settingsSaveTimerRef = useRef<number | null>(null)
  const [copiedMsgIndex, setCopiedMsgIndex] = useState<number | null>(null)

  /** Tracks which citation is currently highlighted (after clicking a superscript). */
  const [highlightedCitation, setHighlightedCitation] = useState<{
    msgIndex: number
    num: number
  } | null>(null)

  /** Tracks which assistant messages have their citation list expanded. Default: collapsed. */
  const [expandedCitations, setExpandedCitations] = useState<Set<number>>(new Set())
  const [expandedEvidence, setExpandedEvidence] = useState<Set<number>>(new Set())

  /** Controls the chunk-preview modal (opened by clicking a citation doc name). */
  const [chunkModal, setChunkModal] = useState<{
    docName: string
    chunkIndex: number
    text: string
    loading: boolean
    error: string | null
  } | null>(null)

  const [showSettings, setShowSettings] = useState(false)
  const [chatSettings, setChatSettings] = useState<ChatSettings>(DEFAULT_CHAT_SETTINGS)
  const [modelProfiles, setModelProfiles] = useState<ModelProfile[]>([])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const resetConversationState = useCallback(() => {
    setMessages([])
    setCitationsByIndex(new Map())
    setEvidenceByIndex(new Map())
    setHighlightedCitation(null)
    setExpandedCitations(new Set())
    setExpandedEvidence(new Set())
  }, [])

  const loadActiveSession = useCallback(async (id: string) => {
    setLoadingSession(true)
    try {
      const data = await getChatSession(id, workspace)
      setMessages(data.messages)
      setChatSettings(data.settings || DEFAULT_CHAT_SETTINGS)
      const restoredCitations: CitationMap = new Map()
      const restoredEvidence: EvidenceMap = new Map()
      data.messages.forEach((msg, idx) => {
        if (msg.role === 'assistant' && msg.citations) {
          restoredCitations.set(idx, normalizeCitations(msg.citations))
        }
        if (msg.role === 'assistant' && msg.evidence) {
          restoredEvidence.set(idx, msg.evidence)
        }
      })
      setCitationsByIndex(restoredCitations)
      setEvidenceByIndex(restoredEvidence)
      setHighlightedCitation(null)
      setExpandedCitations(new Set())
      setExpandedEvidence(new Set())
    } catch {
      setMessages([])
    } finally {
      setLoadingSession(false)
    }
  }, [workspace])

  useEffect(() => {
    let cancelled = false
    Promise.all([
      listModelProfiles(),
      getModelBindings(),
      getModelConfig(workspace),
    ]).then(([profiles, bindings, config]) => {
      if (cancelled) return
      setModelProfiles(profiles)
      if (!activeId) {
        setChatSettings({
          answer_profile_id: bindings.chat.profile_id,
          answer_model: bindings.chat.model || config.chat_model,
          temperature: config.chat_temperature,
          top_p: config.chat_top_p,
          max_tokens: config.chat_max_tokens,
          frequency_penalty: config.frequency_penalty,
          presence_penalty: config.presence_penalty,
          mode: 'mix',
          top_k: 40,
          chunk_top_k: 20,
          enable_rerank: bindings.rerank.enabled !== false,
        })
      }
    }).catch(() => {
      if (!cancelled) setModelProfiles([])
    })
    return () => {
      cancelled = true
    }
  }, [workspace, activeId])

  useEffect(() => {
    if (activeId) {
      loadActiveSession(activeId)
    } else {
      resetConversationState()
      setLoadingSession(false)
    }
  }, [activeId, loadActiveSession, resetConversationState])

  useEffect(() => () => {
    if (settingsSaveTimerRef.current !== null) {
      window.clearTimeout(settingsSaveTimerRef.current)
    }
  }, [])

  const updateSettings = useCallback((patch: Partial<ChatSettings>) => {
    setChatSettings((previous) => {
      const next = { ...previous, ...patch }
      if (activeId) {
        if (settingsSaveTimerRef.current !== null) {
          window.clearTimeout(settingsSaveTimerRef.current)
        }
        settingsSaveTimerRef.current = window.setTimeout(() => {
          void updateChatSessionSettings(activeId, next, workspace)
        }, 350)
      }
      return next
    })
  }, [activeId, workspace])

  /**
   * Handle clicking a [数字] superscript in the answer body.
   * Scrolls to the corresponding citation entry and highlights it briefly.
   */
  const handleCitationClick = useCallback((msgIndex: number, num: number) => {
    setHighlightedCitation({ msgIndex, num })
    // Scroll to the citation element after React re-renders
    setTimeout(() => {
      const el = document.querySelector(
        `[data-citation-key="cite-${msgIndex}-${num}"]`,
      )
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 100)
    // Auto-clear highlight after 3 seconds
    setTimeout(() => {
      setHighlightedCitation((prev) =>
        prev && prev.msgIndex === msgIndex && prev.num === num ? null : prev,
      )
    }, 3000)
  }, [])

  /** Toggle the expand/collapse state of a message's citation list. */
  const toggleCitationExpand = useCallback((msgIndex: number) => {
    setExpandedCitations((prev) => {
      const next = new Set(prev)
      if (next.has(msgIndex)) {
        next.delete(msgIndex)
      } else {
        next.add(msgIndex)
      }
      return next
    })
  }, [])

  const toggleEvidenceExpand = useCallback((msgIndex: number) => {
    setExpandedEvidence((prev) => {
      const next = new Set(prev)
      if (next.has(msgIndex)) {
        next.delete(msgIndex)
      } else {
        next.add(msgIndex)
      }
      return next
    })
  }, [])

  const handleCopyAnswer = useCallback(async (msgIndex: number, content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedMsgIndex(msgIndex)
      setTimeout(() => {
        setCopiedMsgIndex((prev) => (prev === msgIndex ? null : prev))
      }, 1500)
    } catch {
      setCopiedMsgIndex(null)
    }
  }, [])

  const handleReuseQuestion = useCallback((msgIndex: number) => {
    const previousUser = [...messages.slice(0, msgIndex)]
      .reverse()
      .find((message) => message.role === 'user')
    if (!previousUser) return
    setInput(previousUser.content)
    setTimeout(() => inputRef.current?.focus(), 0)
  }, [messages])

  /**
   * Open the chunk-preview modal for a specific document + chunk.
   * Fetches the full chunk text via getDocumentChunks and locates the
   * matching chunk_index.
   */
  const handleDocNameClick = useCallback(async (docName: string, chunkIndex: number) => {
    setChunkModal({ docName, chunkIndex, text: '', loading: true, error: null })
    try {
      const data = await getDocumentChunks(docName, workspace)
      const chunk = data.chunks.find((c) => c.chunk_index === chunkIndex)
      if (chunk) {
        setChunkModal({ docName, chunkIndex, text: chunk.text, loading: false, error: null })
      } else {
        setChunkModal({ docName, chunkIndex, text: '', loading: false, error: '未找到该文本块' })
      }
    } catch (e) {
      const rawMessage = (e as Error).message
      const hint =
        rawMessage === 'Not Found'
          ? '\n请确认后端已启动，且端口与前端代理配置一致（当前代理到 :8101）。'
          : ''
      setChunkModal({
        docName,
        chunkIndex,
        text: '',
        loading: false,
        error: `加载失败: ${rawMessage}${hint}`,
      })
    }
  }, [workspace])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    setSending(true)
    streamingRef.current = true

    const now = new Date().toISOString()
    const userMsg: ChatMessage = { role: 'user', content: text, timestamp: now }
    // Placeholder assistant message (will be updated with streamed content)
    const asstIdx = messages.length + 1
    setMessages((prev) => [...prev, userMsg, { role: 'assistant', content: '...', timestamp: now }])

    let fullContent = ''
    let sessionIdFromStream: string | null = null

    try {
      const response = await chatSendStream({
        session_id: activeId,
        workspace,
        message: text,
        settings: chatSettings,
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: '流式请求失败' }))
        throw new Error(err.detail || `HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('无法读取响应流')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // Parse complete SSE event blocks. Events can be split across network
        // reads, so line-by-line parsing with a per-read eventType is unstable.
        buffer = buffer.replace(/\r\n/g, '\n')
        let boundary = buffer.indexOf('\n\n')
        while (boundary !== -1) {
          const rawEvent = buffer.slice(0, boundary)
          buffer = buffer.slice(boundary + 2)

          let eventType = 'token'
          const dataLines: string[] = []
          for (const rawLine of rawEvent.split('\n')) {
            const line = rawLine.trimEnd()
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim()
            } else if (line.startsWith('data:')) {
              dataLines.push(line.slice(5).trimStart())
            }
          }

          if (dataLines.length === 0) {
            boundary = buffer.indexOf('\n\n')
            continue
          }

          try {
            const data = JSON.parse(dataLines.join('\n'))

            if (eventType === 'citations') {
              const cites = normalizeCitations(Array.isArray(data) ? data : (data.citations || []))
              const rawEvidence = Array.isArray(data) ? null : data.evidence
              const evidence: EvidenceChain = {
                nodes: Array.isArray(rawEvidence?.nodes) ? rawEvidence.nodes : [],
                edges: Array.isArray(rawEvidence?.edges) ? rawEvidence.edges : [],
                chunks: Array.isArray(rawEvidence?.chunks) ? rawEvidence.chunks : cites,
              }
              setCitationsByIndex((prev) => {
                const next = new Map(prev)
                next.set(asstIdx, cites)
                return next
              })
              setEvidenceByIndex((prev) => {
                const next = new Map(prev)
                next.set(asstIdx, evidence)
                return next
              })
              setMessages((prev) => {
                const updated = [...prev]
                if (updated[asstIdx]) {
                  updated[asstIdx] = {
                    ...updated[asstIdx],
                    citations: cites,
                    evidence,
                  }
                }
                return updated
              })
              if (evidence.nodes.length > 0 || evidence.edges.length > 0) {
                try {
                  localStorage.setItem(
                    `tdx_latest_evidence_${workspace}`,
                    JSON.stringify({ workspace, evidence, updated_at: new Date().toISOString() }),
                  )
                } catch {/* ignore */}
              }
            } else if (eventType === 'token') {
              const token = data.token || ''
              fullContent += token
              setMessages((prev) => {
                const updated = [...prev]
                if (updated[asstIdx]) {
                  updated[asstIdx] = { ...updated[asstIdx], content: fullContent || '...' }
                }
                return updated
              })
            } else if (eventType === 'done') {
              sessionIdFromStream = data.session_id
              if (data.content) fullContent = data.content
              setMessages((prev) => {
                const updated = [...prev]
                if (updated[asstIdx]) {
                  updated[asstIdx] = { ...updated[asstIdx], content: fullContent, timestamp: now }
                }
                return updated
              })
            } else if (eventType === 'error') {
              fullContent = `[回答生成失败: ${data.error}]`
              setMessages((prev) => {
                const updated = [...prev]
                if (updated[asstIdx]) {
                  updated[asstIdx] = { ...updated[asstIdx], content: fullContent, timestamp: now }
                }
                return updated
              })
            }
          } catch {/* skip malformed JSON */}

          boundary = buffer.indexOf('\n\n')
        }
      }
    } catch (e: unknown) {
      const errContent = `发送失败: ${(e as Error).message}`
      setMessages((prev) => {
        const updated = [...prev]
        if (updated[asstIdx]) {
          updated[asstIdx] = { ...updated[asstIdx], content: errContent, timestamp: now }
        }
        return updated
      })
    } finally {
      streamingRef.current = false
      setSending(false)

      // Refresh sessions after streaming completes
      const finalSessionId = sessionIdFromStream || activeId
      if (!activeId && finalSessionId) {
        setActiveId(finalSessionId)
      }
      await reloadSessions()
    }
  }

  const formatTime = (ts: string) => {
    try {
      const d = new Date(ts)
      return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
    } catch { return '' }
  }

  /**
   * Render the citations section for an assistant message.
   *
   * - `undefined` citations → historical message with unknown citation state → hide
   * - `[]` citations       → RAG retrieved nothing → show "未检索到相关文档" notice
   * - non-empty citations  → always show the full citation list with [数字] badges
   */
  const renderCitations = (msgIndex: number) => {
    const cites = normalizeCitations(citationsByIndex.get(msgIndex))

    // Historical messages (loaded from server) don't carry citation data
    if (!citationsByIndex.has(msgIndex)) return null

    // No references were retrieved for this answer (not collapsible)
    if (cites.length === 0) {
      return (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <p className="text-xs text-gray-400">
            本次回答未找到足够相关的知识库引用
          </p>
        </div>
      )
    }

    const isExpanded = expandedCitations.has(msgIndex)
    const summary = citationSummary(cites)

    return (
      <div className="mt-3 pt-3 border-t border-gray-100">
        <button
          onClick={() => toggleCitationExpand(msgIndex)}
          className="w-full flex items-center gap-2 text-xs font-medium text-gray-500 hover:text-gray-800 transition-colors"
        >
          <FileText size={14} strokeWidth={1.8} />
          <span>引用文档</span>
          <span className="text-gray-400">
            {summary.docCount} 个文档 · {summary.chunkCount} 条片段 · {summary.firstDoc}
          </span>
          <ChevronDown
            size={14}
            className={`ml-auto text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          />
        </button>
        {isExpanded && (
          <div className="space-y-2 max-h-64 overflow-y-auto pr-1 mt-2">
            {cites.map((c) => {
              const num = c.index
              const isHighlighted =
                highlightedCitation?.msgIndex === msgIndex &&
                highlightedCitation?.num === num
              return (
                <div
                  key={`${c.index}-${c.doc_name}-${c.chunk_index}`}
                  data-citation-key={`cite-${msgIndex}-${num}`}
                  className={`rounded-lg px-3 py-2 text-xs transition-all duration-300 border ${
                    isHighlighted
                      ? 'border-amber-400 bg-amber-50 ring-2 ring-amber-200'
                      : 'border-gray-200 bg-white'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className={`inline-flex items-center justify-center w-5 h-5 rounded-full font-mono text-[10px] font-semibold shrink-0 ${
                        isHighlighted
                          ? 'bg-amber-500 text-white'
                          : 'bg-primary-100 text-primary-700'
                      }`}
                    >
                      {num}
                    </span>
                    <button
                      onClick={() => handleDocNameClick(c.doc_name, c.chunk_index)}
                      className="text-gray-700 font-medium truncate hover:text-primary-600 hover:underline cursor-pointer text-left"
                      title={`点击查看 ${c.doc_name} #${c.chunk_index} 的完整文本`}
                    >
                      {c.doc_name}
                    </button>
                    <span className="text-gray-400 shrink-0">#{c.chunk_index}</span>
                  </div>
                  <p className="text-gray-600 leading-relaxed line-clamp-3">{c.excerpt}</p>
                </div>
              )
            })}
          </div>
        )}
      </div>
    )
  }

  const renderEvidence = (msgIndex: number) => {
    const evidence = evidenceByIndex.get(msgIndex)
    if (!evidence || (evidence.nodes.length === 0 && evidence.edges.length === 0)) {
      return null
    }

    const isExpanded = expandedEvidence.has(msgIndex)
    const hitNodes = new Set(evidence.nodes.map((node) => node.id))
    const relationRows = evidence.edges.slice(0, 8)

    return (
      <div className="mt-3 pt-3 border-t border-gray-100">
        <button
          onClick={() => toggleEvidenceExpand(msgIndex)}
          className="w-full flex items-center gap-2 text-xs font-medium text-gray-500 hover:text-gray-800 transition-colors"
        >
          <Network size={14} strokeWidth={1.8} />
          <span>证据链</span>
          <span className="text-gray-400">
            ({evidence.nodes.length} 节点 / {evidence.edges.length} 关系)
          </span>
          <ChevronDown
            size={14}
            className={`ml-auto text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          />
        </button>
        {isExpanded && (
          <div className="mt-2 grid grid-cols-1 lg:grid-cols-[minmax(280px,1fr)_240px] gap-3">
            <div className="h-64 min-w-0 overflow-hidden rounded-lg border border-gray-200 bg-white">
              <GraphView
                nodes={evidence.nodes}
                edges={evidence.edges}
                hitNodes={hitNodes}
                pathNodes={hitNodes}
              />
            </div>
            <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
              <div className="px-3 py-2 border-b border-gray-100 text-xs font-semibold text-gray-700">
                命中关系
              </div>
              <div className="max-h-56 overflow-y-auto divide-y divide-gray-100">
                {relationRows.length > 0 ? (
                  relationRows.map((edge, idx) => (
                    <div key={`${edge.source}-${edge.target}-${idx}`} className="px-3 py-2 text-xs">
                      <div className="font-medium text-gray-800 break-all">
                        {edge.source} <span className="text-gray-400">→</span> {edge.target}
                      </div>
                      <div className="mt-1 text-gray-500 leading-relaxed line-clamp-2">
                        {edge.relation || edge.description || edge.keywords || 'related'}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="px-3 py-4 text-xs text-gray-400">
                    本次只命中实体，未返回关系。
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  const renderAssistantActions = (msg: ChatMessage, msgIndex: number) => {
    const cites = normalizeCitations(citationsByIndex.get(msgIndex))
    const evidence = evidenceByIndex.get(msgIndex)
    const hasEvidence = Boolean(evidence && (evidence.nodes.length > 0 || evidence.edges.length > 0))

    return (
      <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs text-gray-400">
        <button
          onClick={() => handleCopyAnswer(msgIndex, msg.content)}
          className="rounded-lg px-2 py-1 hover:bg-gray-100 hover:text-gray-700"
        >
          {copiedMsgIndex === msgIndex ? '已复制' : '复制'}
        </button>
        <button
          onClick={() => toggleCitationExpand(msgIndex)}
          disabled={cites.length === 0}
          className="rounded-lg px-2 py-1 hover:bg-gray-100 hover:text-gray-700 disabled:cursor-not-allowed disabled:text-gray-300"
        >
          查看上下文
        </button>
        <button
          onClick={() => toggleEvidenceExpand(msgIndex)}
          disabled={!hasEvidence}
          className="rounded-lg px-2 py-1 hover:bg-gray-100 hover:text-gray-700 disabled:cursor-not-allowed disabled:text-gray-300"
        >
          查看证据链
        </button>
        <button
          onClick={() => handleReuseQuestion(msgIndex)}
          className="rounded-lg px-2 py-1 hover:bg-gray-100 hover:text-gray-700"
        >
          重新提问
        </button>
      </div>
    )
  }

  const selectedModelValue = `${chatSettings.answer_profile_id}::${chatSettings.answer_model}`

  const handleModelSelection = (value: string) => {
    const separator = value.indexOf('::')
    if (separator < 0) return
    updateSettings({
      answer_profile_id: value.slice(0, separator),
      answer_model: value.slice(separator + 2),
    })
  }

  const handleWorkspaceSelection = async (nextWorkspace: string) => {
    if (nextWorkspace === workspace) return
    if (messages.length > 0) {
      const confirmed = await confirm({
        title: '切换问答知识库',
        message: `将保留当前对话，并在知识库“${nextWorkspace}”中开始新对话。`,
        confirmLabel: '切换知识库',
      })
      if (!confirmed) return
    }
    setShowSettings(false)
    onWorkspaceChange(nextWorkspace)
  }

  const renderSettingsPanel = () => (
    <div className="absolute bottom-full left-0 right-0 z-30 mb-2 max-h-[68vh] overflow-y-auto rounded-lg border border-gray-200 bg-white p-5 shadow-[var(--ui-shadow)]">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-gray-900">对话设置</h3>
          <p className="mt-1 text-xs text-gray-500">设置按当前对话保存；嵌入模型仍跟随知识库索引。</p>
        </div>
        <button
          onClick={() => setShowSettings(false)}
          className="ui-icon-button"
          title="关闭设置"
          aria-label="关闭设置"
        >
          <X size={16} />
        </button>
      </div>

      <div className="mt-5 grid gap-6 lg:grid-cols-2">
        <section>
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-gray-800">
            <Sparkles size={16} />
            生成参数
          </div>
          <div className="space-y-4">
            <RangeField label="Temperature" value={chatSettings.temperature} min={0} max={2} step={0.05}
              onChange={(value) => updateSettings({ temperature: value })} />
            <RangeField label="Top-P" value={chatSettings.top_p} min={0} max={1} step={0.05}
              onChange={(value) => updateSettings({ top_p: value })} />
            <RangeField label="Max Tokens" value={chatSettings.max_tokens} min={256} max={8192} step={256}
              onChange={(value) => updateSettings({ max_tokens: value })} />
            <RangeField label="Frequency Penalty" value={chatSettings.frequency_penalty} min={-2} max={2} step={0.1}
              onChange={(value) => updateSettings({ frequency_penalty: value })} />
            <RangeField label="Presence Penalty" value={chatSettings.presence_penalty} min={-2} max={2} step={0.1}
              onChange={(value) => updateSettings({ presence_penalty: value })} />
          </div>
        </section>

        <section>
          <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-gray-800">
            <Network size={16} />
            检索参数
          </div>
          <div className="space-y-4">
            <SelectField
              label="LightRAG 模式"
              value={chatSettings.mode}
              onChange={(event) => updateSettings({ mode: event.target.value })}
            >
              <option value="mix">mix - 图谱与文本向量</option>
              <option value="hybrid">hybrid - 局部与全局图谱</option>
              <option value="local">local - 实体局部</option>
              <option value="global">global - 关系全局</option>
              <option value="naive">naive - 仅文本向量</option>
            </SelectField>
            <RangeField label="图谱 Top-K" value={chatSettings.top_k} min={1} max={100}
              onChange={(value) => updateSettings({ top_k: value })} />
            <RangeField label="文本块 Top-K" value={chatSettings.chunk_top_k} min={1} max={100}
              onChange={(value) => updateSettings({ chunk_top_k: value })} />
            <div className="pt-1">
              <Toggle
                checked={chatSettings.enable_rerank}
                onChange={(value) => updateSettings({ enable_rerank: value })}
                label="启用 Rerank"
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  )

  const renderComposerToolbar = () => (
    <div className="mb-2 flex min-h-10 flex-wrap items-center gap-1.5 rounded-lg border border-gray-200 bg-gray-50/70 p-1.5">
      <WorkspaceSwitcher
        workspace={workspace}
        workspaces={workspaces}
        onChange={handleWorkspaceSelection}
        placement="top"
        compact
        className="w-44 shrink-0"
      />

      <span className="h-5 w-px bg-gray-200" />

      <label className="flex h-8 min-w-[180px] flex-1 items-center gap-1.5 rounded-md border border-transparent bg-white px-2 text-xs text-gray-600 shadow-sm hover:border-gray-200">
        <Sparkles size={14} strokeWidth={1.8} />
        <select
          value={selectedModelValue}
          onChange={(event) => handleModelSelection(event.target.value)}
          className="min-w-0 max-w-64 flex-1 appearance-none truncate bg-transparent font-medium text-gray-800 outline-none"
          aria-label="选择回答模型"
        >
          {modelProfiles.map((profile) => {
            const models = (profile.models_cache || []).filter(isLikelyChatModel)
            if (
              profile.id === chatSettings.answer_profile_id
              && chatSettings.answer_model
              && !models.some((model) => model.id === chatSettings.answer_model)
            ) {
              models.unshift({ id: chatSettings.answer_model, type: 'selected' })
            }
            return (
              <optgroup key={profile.id} label={profile.name}>
                {models.map((model) => (
                  <option key={`${profile.id}-${model.id}`} value={`${profile.id}::${model.id}`}>
                    {model.id}
                  </option>
                ))}
              </optgroup>
            )
          })}
          {modelProfiles.length === 0 && (
            <option value={selectedModelValue}>{chatSettings.answer_model}</option>
          )}
        </select>
        <ChevronDown size={13} className="text-gray-400" />
      </label>

      <button
        onClick={() => setShowSettings((visible) => !visible)}
        className={`flex h-8 items-center gap-1.5 rounded-md px-2 text-xs transition ${
          showSettings ? 'bg-gray-100 text-gray-900' : 'text-gray-600 hover:bg-gray-50'
        }`}
        aria-expanded={showSettings}
      >
        <Settings2 size={14} strokeWidth={1.8} />
        <span>{chatSettings.mode}</span>
        <span className="text-gray-300">·</span>
        <span>{chatSettings.chunk_top_k} 块</span>
        <span className={chatSettings.enable_rerank ? 'text-emerald-600' : 'text-gray-400'}>
          Rerank {chatSettings.enable_rerank ? '开' : '关'}
        </span>
      </button>
    </div>
  )

  const isStreamingMsg = (msg: ChatMessage, i: number) =>
    msg.role === 'assistant' && streamingRef.current && i === messages.length - 1

  /**
   * Render an assistant message's markdown content with [数字] citation
   * markers converted to clickable superscripts.
   */
  const renderAssistantContent = (content: string, msgIndex: number) => {
    const onCiteClick = (num: number) => handleCitationClick(msgIndex, num)
    const isCiteHighlighted = (num: number) =>
      highlightedCitation?.msgIndex === msgIndex &&
      highlightedCitation?.num === num
    const validCitationNumbers = new Set(
      normalizeCitations(citationsByIndex.get(msgIndex)).map((citation) => citation.index),
    )

    const process = (children: ReactNode) =>
      processChildren(children, onCiteClick, isCiteHighlighted, validCitationNumbers)

    return (
      <div className="prose prose-sm max-w-none prose-headings:text-gray-800 prose-p:text-gray-700 prose-code:text-primary-700 prose-code:bg-gray-200 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-pre:bg-gray-800 prose-pre:text-gray-100 prose-a:text-primary-600 prose-strong:text-gray-800 prose-li:text-gray-700">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => <p>{process(children)}</p>,
            li: ({ children }) => <li>{process(children)}</li>,
            h1: ({ children }) => <h1>{process(children)}</h1>,
            h2: ({ children }) => <h2>{process(children)}</h2>,
            h3: ({ children }) => <h3>{process(children)}</h3>,
            h4: ({ children }) => <h4>{process(children)}</h4>,
            h5: ({ children }) => <h5>{process(children)}</h5>,
            h6: ({ children }) => <h6>{process(children)}</h6>,
            td: ({ children }) => <td>{process(children)}</td>,
            th: ({ children }) => <th>{process(children)}</th>,
            strong: ({ children }) => <strong>{process(children)}</strong>,
            em: ({ children }) => <em>{process(children)}</em>,
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    )
  }

  return (
    <div className="flex h-full bg-white overflow-hidden">
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex h-12 shrink-0 items-center justify-between border-b border-gray-100 px-5">
          <h3 className="truncate text-sm font-medium text-gray-800">
            {activeId ? sessions.find((s) => s.id === activeId)?.title || '对话' : '新对话'}
          </h3>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          {loadingSession ? (
            <div className="flex items-center justify-center py-12 text-gray-400">
              <div className="animate-spin w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full mr-2" />
              <span className="text-sm">加载对话历史...</span>
            </div>
          ) : messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="mx-auto mb-4 grid h-10 w-10 place-items-center rounded-lg border border-gray-200 bg-gray-50 text-gray-500">
                  <Sparkles size={18} strokeWidth={1.7} />
                </div>
                <p className="text-sm text-gray-600">{activeId ? '选择一条历史对话，或' : ''}开始新的问答</p>
                <p className="text-xs text-gray-400 mt-1">输入问题，系统将基于当前知识库回答</p>
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-7">
              {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`${
                  msg.role === 'user'
                    ? 'max-w-[72%] rounded-3xl bg-gray-100 px-4 py-2.5 text-gray-900'
                    : 'w-full text-gray-900'
                }`}>
                  {msg.role === 'user' ? (
                    <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  ) : (
                    isStreamingMsg(msg, i) && msg.content === '...' ? (
                      <div className="prose prose-sm max-w-none">
                        <span className="inline-flex gap-0.5">
                          <span className="animate-pulse">●</span>
                          <span className="animate-pulse" style={{ animationDelay: '0.2s' }}>●</span>
                          <span className="animate-pulse" style={{ animationDelay: '0.4s' }}>●</span>
                        </span>
                      </div>
                    ) : (
                      renderAssistantContent(msg.content, i)
                    )
                  )}
                  <p className="text-[10px] mt-2 text-gray-400">
                    {formatTime(msg.timestamp)}
                  </p>
                  {msg.role === 'user' && (
                    <button
                      onClick={() => handleCopyAnswer(i, msg.content)}
                      className="mt-1 inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[11px] text-gray-400 hover:bg-white/70 hover:text-gray-700"
                      title="复制提问"
                    >
                      {copiedMsgIndex === i ? <Check size={12} /> : <Copy size={12} />}
                      {copiedMsgIndex === i ? '已复制' : '复制'}
                    </button>
                  )}
                  {msg.role === 'assistant' && !isStreamingMsg(msg, i) && renderAssistantActions(msg, i)}
                  {msg.role === 'assistant' && renderCitations(i)}
                  {msg.role === 'assistant' && renderEvidence(i)}
                </div>
              </div>
              ))}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="shrink-0 px-4 pb-4 pt-2">
          <div className="relative mx-auto max-w-3xl">
            {showSettings && renderSettingsPanel()}
            {renderComposerToolbar()}
            <div className="flex gap-2 rounded-2xl border border-gray-200 bg-white px-3 py-2 shadow-sm focus-within:border-gray-300">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                placeholder="输入问题，按 Enter 发送"
                disabled={sending}
                className="min-w-0 flex-1 bg-transparent px-1 py-2 text-sm outline-none disabled:opacity-50"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || sending}
                className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-gray-900 text-white transition hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-400"
                title="发送"
                aria-label="发送"
              >
                {sending ? (
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                ) : (
                  <Send size={16} strokeWidth={2} />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Chunk preview modal — opened by clicking a citation doc name */}
      {chunkModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
          onClick={() => setChunkModal(null)}
        >
          <div
            className="mx-4 flex max-h-[80vh] w-full max-w-2xl flex-col rounded-lg bg-white shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
              <h3 className="text-sm font-semibold text-gray-800 truncate">
                {chunkModal.docName}
                <span className="text-gray-400 font-normal ml-1">#{chunkModal.chunkIndex}</span>
              </h3>
              <button
                onClick={() => setChunkModal(null)}
                className="ui-icon-button ml-3"
                aria-label="关闭"
              >
                <X size={17} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-5 py-4">
              {chunkModal.loading ? (
                <div className="flex items-center justify-center py-8 text-gray-400">
                  <div className="animate-spin w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full mr-2" />
                  <span className="text-sm">加载文本块...</span>
                </div>
              ) : chunkModal.error ? (
                <p className="text-sm text-gray-500 text-center py-8">{chunkModal.error}</p>
              ) : (
                <pre className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed font-sans">
                  {chunkModal.text}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
