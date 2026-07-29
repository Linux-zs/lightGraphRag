import { Fragment, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import GraphView from '../components/GraphView'
import {
  chatSendStream,
  getChatSession,
  getDocumentChunks,
  ChatSessionListItem,
  ChatMessage,
  Citation,
  EvidenceChain,
} from '../api'

type CitationMap = Map<number, Citation[]>
type EvidenceMap = Map<number, EvidenceChain>

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
  sessions: ChatSessionListItem[]
  activeId: string | null
  setActiveId: (id: string | null) => void
  reloadSessions: () => Promise<void>
}

export default function QAChat({ workspace, sessions, activeId, setActiveId, reloadSessions }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [citationsByIndex, setCitationsByIndex] = useState<CitationMap>(new Map())
  const [evidenceByIndex, setEvidenceByIndex] = useState<EvidenceMap>(new Map())
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [loadingSession, setLoadingSession] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const streamingRef = useRef(false)
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

  // Retrieval params (F3)
  const [showSettings, setShowSettings] = useState(false)
  const [mode, setMode] = useState('mix')
  const [topK, setTopK] = useState(40)
  const [chunkTopK, setChunkTopK] = useState(20)
  const [enableRerank, setEnableRerank] = useState(true)

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
    if (activeId) {
      loadActiveSession(activeId)
    } else {
      resetConversationState()
      setLoadingSession(false)
    }
  }, [activeId, loadActiveSession, resetConversationState])

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
        mode,
        top_k: topK,
        chunk_top_k: chunkTopK,
        enable_rerank: enableRerank,
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
          <span className="inline-flex h-4 w-4 items-center justify-center rounded border border-gray-300 text-[10px] text-gray-400">
            #
          </span>
          <span>引用文档</span>
          <span className="text-gray-400">
            {summary.docCount} 个文档 · {summary.chunkCount} 条片段 · {summary.firstDoc}
          </span>
          <span className="ml-auto text-gray-400">{isExpanded ? '▲' : '▼'}</span>
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
          <span className="inline-block h-3 w-3 rotate-45 border border-gray-400" />
          <span>证据链</span>
          <span className="text-gray-400">
            ({evidence.nodes.length} 节点 / {evidence.edges.length} 关系)
          </span>
          <span className="ml-auto text-gray-400">{isExpanded ? '▲' : '▼'}</span>
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

  const renderSettingsPanel = () => (
    <div className="px-6 py-3 border-t border-gray-100 bg-gray-50 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">检索设置</span>
        <button
          onClick={() => setShowSettings(false)}
          className="text-xs text-gray-400 hover:text-gray-600"
        >
          收起
        </button>
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <div>
          <label className="block text-xs text-gray-500 mb-1">mode</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            className="w-full border border-gray-200 rounded-lg px-2 py-1.5 text-xs bg-white"
          >
            <option value="mix">mix</option>
            <option value="hybrid">hybrid</option>
            <option value="local">local</option>
            <option value="global">global</option>
            <option value="naive">naive</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">top_k: <span className="text-primary-600 font-medium">{topK}</span></label>
          <input type="range" min={1} max={100} value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="w-full accent-primary-500" />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">chunk_top_k: <span className="text-primary-600 font-medium">{chunkTopK}</span></label>
          <input type="range" min={1} max={100} value={chunkTopK}
            onChange={(e) => setChunkTopK(parseInt(e.target.value))}
            className="w-full accent-primary-500" />
        </div>
        <div className="flex items-end">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={enableRerank}
              onChange={(e) => setEnableRerank(e.target.checked)}
              className="w-4 h-4 rounded accent-primary-500" />
            <span className="text-xs text-gray-600">启用重排序</span>
          </label>
        </div>
      </div>
    </div>
  )

  const renderRetrievalStatusBar = () => (
    <div className="px-4 pt-2 shrink-0">
      <button
        onClick={() => setShowSettings(true)}
        className="mx-auto flex max-w-3xl flex-wrap items-center gap-2 rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-500 shadow-sm transition-colors hover:border-gray-300 hover:text-gray-800"
      >
        <span className="font-medium text-gray-700">检索</span>
        <span>知识库: {workspace}</span>
        <span className="text-gray-300">·</span>
        <span>mode {mode}</span>
        <span className="text-gray-300">·</span>
        <span>topK {topK}</span>
        <span className="text-gray-300">·</span>
        <span>chunk {chunkTopK}</span>
        <span className="text-gray-300">·</span>
        <span className={enableRerank ? 'text-emerald-600' : 'text-gray-400'}>
          rerank {enableRerank ? '开' : '关'}
        </span>
        <span className="ml-auto text-gray-400">设置</span>
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
    <div className="flex h-screen bg-white overflow-hidden">
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="h-14 px-5 border-b border-gray-100 shrink-0 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-gray-800">
              {activeId ? sessions.find((s) => s.id === activeId)?.title || '对话' : '新对话'}
            </h3>
            <p className="text-xs text-gray-400">
              当前知识库: {workspace}
            </p>
          </div>
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
                <div className="mx-auto mb-4 h-10 w-10 rounded-full border border-gray-300 flex items-center justify-center">
                  <span className="h-4 w-4 rounded-full border border-gray-500" />
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

        {/* Settings panel (F3) */}
        {showSettings ? (
          renderSettingsPanel()
        ) : (
          renderRetrievalStatusBar()
        )}

        {/* Input */}
        <div className="px-4 py-4 shrink-0">
          <div className="mx-auto max-w-3xl flex gap-3 rounded-3xl border border-gray-200 bg-white px-3 py-2 shadow-sm">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="输入问题，按 Enter 发送..."
              disabled={sending}
              className="flex-1 bg-transparent px-2 py-2 text-sm outline-none disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || sending}
              className="h-9 min-w-9 px-4 text-sm font-medium rounded-full bg-gray-900 text-white hover:bg-gray-700 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {sending ? (
                <span className="inline-flex items-center gap-1">
                  <div className="animate-spin w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full" />
                  发送
                </span>
              ) : '发送'}
            </button>
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
            className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
              <h3 className="text-sm font-semibold text-gray-800 truncate">
                {chunkModal.docName}
                <span className="text-gray-400 font-normal ml-1">#{chunkModal.chunkIndex}</span>
              </h3>
              <button
                onClick={() => setChunkModal(null)}
                className="text-gray-400 hover:text-gray-600 text-lg leading-none ml-3 shrink-0"
              >
                ×
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
