import { createContext, useContext, useState } from 'react'
import type { Element, Root, RootContent } from 'hast'
import type { Citation } from '../api'
import { StoredImage } from './Multimodal'

export const AnswerImagesContext = createContext<{ workspace: string; citations: Citation[] }>({ workspace: '', citations: [] })

/** Attach only verified image citations, after the Markdown block citing them. */
export function inlineAnswerImages(citations: Citation[]) {
  return () => (tree: Root) => {
    const byIndex = new Map(citations.filter(c => c.image_id).map(c => [c.index, c]))
    const seen = new Set<string>()
    const text = (node: RootContent): string => {
      if (node.type === 'text') return node.value
      if (node.type !== 'element' || ['code', 'pre', 'img'].includes(node.tagName)) return ''
      return node.children.map(text).join('')
    }
    const children: RootContent[] = []
    for (const block of tree.children) {
      children.push(block)
      for (const match of text(block).matchAll(/\[(\d+)\]/g)) {
        const citation = byIndex.get(Number(match[1]))
        if (!citation?.image_id || seen.has(citation.image_id) || seen.size >= 3) continue
        seen.add(citation.image_id)
        children.push({ type: 'element', tagName: 'figure',
          properties: { 'data-image-citation': citation.index }, children: [] })
      }
    }
    tree.children = children
  }
}

// Stable renderer identity preserves the collapse state during streamed updates.
export function AnswerImageFigure({ node }: { node?: Element }) {
  const { workspace, citations } = useContext(AnswerImagesContext)
  const [collapsed, setCollapsed] = useState(false)
  const index = Number(node?.properties['data-image-citation'])
  const citation = citations.find(c => c.index === index && c.image_id)
  if (!citation) return null
  const location = citation.location?.page ? `第 ${citation.location.page} 页`
    : citation.location?.paragraph ? `段落 ${citation.location.paragraph}` : '原图'
  return <figure className="not-prose my-4 overflow-hidden rounded-xl border border-gray-200 bg-gray-50">
    <figcaption className="flex items-start justify-between gap-3 px-3 py-2 text-xs text-gray-600">
      <span className="min-w-0 break-words"><span className="font-medium text-gray-800">图片 [{citation.index}]</span> · {citation.doc_name} · {location}</span>
      <button type="button" aria-expanded={!collapsed} onClick={() => setCollapsed(value => !value)}
        className="shrink-0 rounded px-1 text-gray-600 underline hover:text-gray-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500">
        {collapsed ? '展开图片' : '收起图片'}
      </button>
    </figcaption>
    {!collapsed && <div className="border-t border-gray-200 bg-white p-3">
      <StoredImage path={`/kb/images/${citation.image_id}?workspace=${encodeURIComponent(workspace)}${citation.doc_id ? `&doc_id=${encodeURIComponent(citation.doc_id)}` : ''}`}
        imageClassName="mx-auto max-h-96 max-w-full rounded-lg object-contain"
        alt={`${citation.doc_name} · ${location} · 图片引用 ${citation.index}`} />
    </div>}
  </figure>
}
