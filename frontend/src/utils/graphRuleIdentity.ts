import type { GraphRuleTemplate } from '../api'

export interface GraphRuleDraft {
  entityTypes: string[]
  relationTypes: string[]
  entityExclusionRules: string[]
  relationExclusionRules: string[]
  aliasesText: string
  extractionPrompt: string
}

const sameList = (left: string[], right: string[]) => (
  left.length === right.length && left.every((value, index) => value === right[index])
)

export function appendUniqueRuleText(current: string, rule: string): string {
  const candidate = rule.trim()
  if (!candidate) return current
  const existing = current
    .split(/\r?\n/)
    .map((value) => value.trim())
    .filter(Boolean)
  if (existing.some((value) => value.toLowerCase() === candidate.toLowerCase())) {
    return current
  }
  return [...existing, candidate].join('\n')
}

export function resolveGraphRuleIdentity(
  draft: GraphRuleDraft,
  template: GraphRuleTemplate | undefined,
): { id: string; name: string } {
  const matchesTemplate = Boolean(template)
    && sameList(draft.entityTypes, template?.entity_types || [])
    && sameList(draft.relationTypes, template?.relation_types || [])
    && sameList(draft.entityExclusionRules, template?.entity_exclusion_rules || [])
    && sameList(draft.relationExclusionRules, template?.relation_exclusion_rules || [])
    && draft.aliasesText === (template?.aliases_text || '')
    && draft.extractionPrompt === (template?.extraction_prompt || '')

  return matchesTemplate && template
    ? { id: template.id, name: template.name }
    : { id: '', name: '当前知识库自定义规则' }
}
