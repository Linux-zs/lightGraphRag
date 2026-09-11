import { describe, expect, it } from 'vitest'
import type { GraphRuleTemplate } from '../api'
import { appendUniqueRuleText, resolveGraphRuleIdentity } from './graphRuleIdentity'

const template: GraphRuleTemplate = {
  id: 'general',
  name: '通用知识库',
  description: '',
  entity_types: ['人物'],
  relation_types: ['属于'],
  entity_exclusion_rules: [],
  relation_exclusion_rules: [],
  aliases_text: '',
  extraction_prompt: '保守抽取',
  built_in: true,
}

const draft = {
  entityTypes: ['人物'],
  relationTypes: ['属于'],
  entityExclusionRules: [] as string[],
  relationExclusionRules: [] as string[],
  aliasesText: '',
  extractionPrompt: '保守抽取',
}

describe('resolveGraphRuleIdentity', () => {
  it('retains template identity while editable content still matches', () => {
    expect(resolveGraphRuleIdentity(draft, template)).toEqual({
      id: 'general', name: '通用知识库',
    })
  })

  it('marks edited exclusions as a workspace custom rule', () => {
    expect(resolveGraphRuleIdentity({
      ...draft,
      entityExclusionRules: ['contains:/tmp/'],
    }, template)).toEqual({ id: '', name: '当前知识库自定义规则' })
  })

  it('does not borrow identity from an un-applied template selection', () => {
    expect(resolveGraphRuleIdentity(draft, undefined)).toEqual({
      id: '', name: '当前知识库自定义规则',
    })
  })
})

describe('appendUniqueRuleText', () => {
  it('appends a trimmed exact rule while preserving existing order', () => {
    expect(appendUniqueRuleText('unknown\ncontains:/tmp/', ' server.log ')).toBe(
      'unknown\ncontains:/tmp/\nserver.log',
    )
  })

  it('does not append a case-insensitive duplicate', () => {
    expect(appendUniqueRuleText('Unknown', 'unknown')).toBe('Unknown')
  })
})
