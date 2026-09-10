import type { IndexTaskResult } from '../api'

export default function ExtractionPolicySummary({ results }: { results: IndexTaskResult[] }) {
  const filtered = results.filter(result => Object.values(result.kg_policy_rejections || {}).some(count => count > 0))
  if (!filtered.length) return null
  return <div className="mt-2 rounded-md bg-slate-50 px-3 py-2 text-xs text-slate-700">
    <p className="font-medium">抽取规则过滤（记录数，非去重实体数）</p>
    {filtered.map(result => <p key={`${result.doc_name}-policy`} className="mt-1">{result.doc_name}：
      实体类型不匹配 {result.kg_policy_rejections?.entity_type_not_allowed || 0}；
      关系端点未通过 {result.kg_policy_rejections?.relation_endpoint_not_allowed || 0}；
      关系类型不匹配 {result.kg_policy_rejections?.relation_type_not_allowed || 0}
    </p>)}
  </div>
}
