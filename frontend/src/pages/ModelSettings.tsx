import { useEffect, useMemo, useState } from 'react'
import {
  discoverModels,
  discoverProfileModels,
  DiscoveredModel,
  deleteModelProfile,
  getModelBindings,
  getModelConfig,
  listModelProfiles,
  ModelBinding,
  ModelBindings,
  ModelConfig,
  ModelProfile,
  saveModelProfile,
  testChatModel,
  testEmbeddingModel,
  testRerankModel,
  updateModelBindings,
  updateModelConfig,
} from '../api'

const DEFAULT_ANSWER_SYSTEM_PROMPT =
  '你是通达信系统技术支持知识库助手。必须使用简体中文回答。' +
  '只依据给定参考资料回答；资料不足时明确说“知识库上下文不足”。' +
  '参考资料只作为依据，不要把原文逐条搬运成答案。' +
  '你需要先理解用户问题，再综合多条资料，按原因、链路、排查步骤或结论组织成通顺、有逻辑的说明。' +
  '回答要结构清晰，使用正常的 Markdown 标题和有序列表。' +
  '不要输出 References/引用文档列表，不要输出 assistant/user 角色名，' +
  '不要复述大段原始脚本，不要输出孤立的编号或字母。' +
  '引用资料时在相关句子末尾使用 [数字] 标记，数字必须来自参考资料编号。'

const DEFAULT_CONFIG: ModelConfig = {
  embed_model: 'BAAI/bge-large-zh-v1.5',
  embed_base_url: 'https://api.siliconflow.cn/v1',
  rerank_model: 'BAAI/bge-reranker-v2-m3',
  chat_model: 'Qwen/Qwen2.5-7B-Instruct',
  chat_temperature: 0.7,
  chat_top_p: 0.9,
  chat_max_tokens: 4096,
  answer_system_prompt: DEFAULT_ANSWER_SYSTEM_PROMPT,
}

const DEFAULT_BINDINGS: ModelBindings = {
  chat: { profile_id: 'siliconflow-default', model: 'Qwen/Qwen2.5-7B-Instruct' },
  embedding: {
    profile_id: 'siliconflow-default',
    model: 'BAAI/bge-large-zh-v1.5',
    embed_dim: 1024,
    embed_max_chars: 700,
  },
  rerank: { profile_id: 'siliconflow-default', model: 'BAAI/bge-reranker-v2-m3', enabled: true },
}

function modelOptions(profile: ModelProfile | undefined, manualModel: string): DiscoveredModel[] {
  const models = profile?.models_cache || []
  if (manualModel && !models.some((item) => item.id === manualModel)) {
    return [{ id: manualModel, type: 'manual' }, ...models]
  }
  return models
}

function isLocalApiBase(apiBase: string): boolean {
  return /localhost|127\.0\.0\.1|0\.0\.0\.0/.test(apiBase)
}

function needsSavedApiKey(profile: ModelProfile | undefined): boolean {
  return Boolean(profile && !profile.has_api_key && !isLocalApiBase(profile.api_base))
}

function friendlyError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error)
  if (message.includes('Illegal header value')) {
    return '当前连接没有有效 API Key，请编辑连接档案并保存 API Key 后再测试。'
  }
  if (message.includes('401') || message.includes('Unauthorized')) {
    return '接口返回 401，请确认 API Key 已保存且有权限访问该 API 地址。'
  }
  return message
}

export default function ModelSettings() {
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [bindings, setBindings] = useState<ModelBindings>(DEFAULT_BINDINGS)
  const [config, setConfig] = useState<ModelConfig>(DEFAULT_CONFIG)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [testing, setTesting] = useState('')

  const [profileDraft, setProfileDraft] = useState({
    id: '',
    name: 'SiliconFlow',
    api_base: 'https://api.siliconflow.cn/v1',
    api_key: '',
  })
  const [draftModels, setDraftModels] = useState<DiscoveredModel[]>([])
  const [discovering, setDiscovering] = useState(false)

  const profileById = useMemo(() => {
    const map = new Map<string, ModelProfile>()
    profiles.forEach((profile) => map.set(profile.id, profile))
    return map
  }, [profiles])

  const loadAll = async () => {
    setLoading(true)
    setMessage('')
    try {
      const [profileData, bindingData, modelConfig] = await Promise.all([
        listModelProfiles(),
        getModelBindings(),
        getModelConfig(),
      ])
      setProfiles(profileData)
      setBindings(bindingData)
      setConfig(modelConfig)
    } catch (e) {
      setMessage(`加载失败: ${(e as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAll()
  }, [])

  const updateBinding = (purpose: keyof ModelBindings, patch: Partial<ModelBinding>) => {
    setBindings((prev) => ({
      ...prev,
      [purpose]: { ...prev[purpose], ...patch },
    }))
  }

  const handleDiscoverDraft = async () => {
    if (!profileDraft.api_base.trim()) return
    if (!profileDraft.api_key.trim() && !isLocalApiBase(profileDraft.api_base)) {
      setMessage('远端模型服务通常需要 API Key；请先输入 API Key，再发现模型。')
      return
    }
    setDiscovering(true)
    setMessage('')
    try {
      const result = await discoverModels(profileDraft.api_base.trim(), profileDraft.api_key.trim())
      setDraftModels(result.models)
      setMessage(`发现 ${result.models.length} 个模型`)
    } catch (e) {
      setMessage(`发现模型失败: ${friendlyError(e)}`)
    } finally {
      setDiscovering(false)
    }
  }

  const handleSaveProfile = async () => {
    if (!profileDraft.name.trim() || !profileDraft.api_base.trim()) return
    setSaving(true)
    setMessage('')
    try {
      const saved = await saveModelProfile({
        id: profileDraft.id || undefined,
        name: profileDraft.name.trim(),
        api_base: profileDraft.api_base.trim(),
        api_key: profileDraft.api_key.trim() || undefined,
        api_type: 'openai_compatible',
      })
      setProfiles((prev) => [saved, ...prev.filter((item) => item.id !== saved.id)])
      setProfileDraft({ id: '', name: 'SiliconFlow', api_base: 'https://api.siliconflow.cn/v1', api_key: '' })
      setDraftModels([])
      setMessage('连接档案已保存，API Key 不会回显')
    } catch (e) {
      setMessage(`保存连接失败: ${(e as Error).message}`)
    } finally {
      setSaving(false)
    }
  }

  const refreshProfileModels = async (profileId: string) => {
    const profile = profileById.get(profileId)
    if (needsSavedApiKey(profile)) {
      setMessage('该连接还没有保存 API Key，请先编辑连接档案并保存 API Key，再刷新模型列表。')
      return
    }
    setTesting(`discover-${profileId}`)
    setMessage('')
    try {
      const result = await discoverProfileModels(profileId)
      setProfiles((prev) =>
        prev.map((profile) =>
          profile.id === profileId ? { ...profile, models_cache: result.models } : profile,
        ),
      )
      setMessage(`已刷新模型列表: ${result.models.length} 个`)
    } catch (e) {
      setMessage(`刷新模型失败: ${friendlyError(e)}`)
    } finally {
      setTesting('')
    }
  }

  const editProfile = (profile: ModelProfile) => {
    setProfileDraft({
      id: profile.id,
      name: profile.name,
      api_base: profile.api_base,
      api_key: '',
    })
    setMessage('已载入连接档案；如需替换 API Key，请重新输入后保存')
  }

  const removeProfile = async (profile: ModelProfile) => {
    const ok = window.confirm(`确认删除连接档案“${profile.name}”？相关绑定会回退到默认连接。`)
    if (!ok) return
    setSaving(true)
    setMessage('')
    try {
      await deleteModelProfile(profile.id)
      await loadAll()
      setMessage('连接档案已删除')
    } catch (e) {
      setMessage(`删除连接失败: ${(e as Error).message}`)
    } finally {
      setSaving(false)
    }
  }

  const saveBindings = async () => {
    const ok =
      !bindings.embedding.embed_dim ||
      window.confirm('如果嵌入模型或维度发生变化，已有 LightRAG 索引可能不可复用，需要清空并重建知识库。确认保存？')
    if (!ok) return
    setSaving(true)
    setMessage('')
    try {
      const result = await updateModelBindings(bindings)
      setBindings(result.bindings)
      setMessage(result.embedding_changed ? '绑定已保存；嵌入模型/维度已变化，请重建知识库索引。' : '模型绑定已保存')
    } catch (e) {
      setMessage(`保存绑定失败: ${(e as Error).message}`)
    } finally {
      setSaving(false)
    }
  }

  const saveGenerationConfig = async () => {
    setSaving(true)
    setMessage('')
    try {
      await updateModelConfig(config)
      setMessage('生成参数和问答提示词已保存')
    } catch (e) {
      setMessage(`保存失败: ${(e as Error).message}`)
    } finally {
      setSaving(false)
    }
  }

  const testPurpose = async (purpose: keyof ModelBindings) => {
    const binding = bindings[purpose]
    if (!binding.profile_id || !binding.model) return
    const profile = profileById.get(binding.profile_id)
    if (needsSavedApiKey(profile)) {
      setMessage('当前连接没有保存 API Key，请先编辑连接档案并保存 API Key 后再测试。')
      return
    }
    setTesting(purpose)
    setMessage('')
    try {
      if (purpose === 'chat') {
        await testChatModel(binding.profile_id, binding.model)
        setMessage('大语言模型测试通过')
      } else if (purpose === 'embedding') {
        const result = await testEmbeddingModel(binding.profile_id, binding.model)
        updateBinding('embedding', { embed_dim: result.dimensions })
        setMessage(`嵌入模型测试通过，维度 ${result.dimensions}`)
      } else {
        await testRerankModel(binding.profile_id, binding.model)
        setMessage('Rerank 模型测试通过')
      }
    } catch (e) {
      setMessage(`模型测试失败: ${friendlyError(e)}`)
    } finally {
      setTesting('')
    }
  }

  const renderBinding = (purpose: keyof ModelBindings, title: string, hint: string) => {
    const binding = bindings[purpose]
    const profile = profileById.get(binding.profile_id)
    const options = modelOptions(profile, binding.model)

    return (
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-gray-800">{title}</h3>
            <p className="mt-1 text-xs text-gray-500">{hint}</p>
          </div>
          {purpose === 'rerank' && (
            <label className="flex items-center gap-2 text-xs text-gray-600">
              <input
                type="checkbox"
                checked={binding.enabled !== false}
                onChange={(e) => updateBinding('rerank', { enabled: e.target.checked })}
              />
              启用
            </label>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)_auto] gap-3">
          <select
            value={binding.profile_id}
            onChange={(e) => updateBinding(purpose, { profile_id: e.target.value })}
            className="rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white"
          >
            {profiles.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>

          <div className="flex gap-2">
            <select
              value={binding.model}
              onChange={(e) => updateBinding(purpose, { model: e.target.value })}
              className="min-w-0 flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white"
            >
              {options.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.id}
                </option>
              ))}
              {options.length === 0 && <option value={binding.model}>{binding.model || '请手动输入模型名'}</option>}
            </select>
            <input
              value={binding.model}
              onChange={(e) => updateBinding(purpose, { model: e.target.value })}
              className="min-w-0 flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm"
              placeholder="也可手动输入模型名"
            />
          </div>

          <button
            onClick={() => testPurpose(purpose)}
            disabled={testing === purpose || !binding.profile_id || !binding.model}
            className="rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 disabled:text-gray-300"
          >
            {testing === purpose ? '测试中...' : '测试'}
          </button>
        </div>

        {purpose === 'embedding' && (
          <div className="mt-3 grid grid-cols-2 gap-3">
            <label className="text-xs text-gray-500">
              向量维度
              <input
                type="number"
                value={binding.embed_dim || 1024}
                onChange={(e) => updateBinding('embedding', { embed_dim: Number(e.target.value) })}
                className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
              />
            </label>
            <label className="text-xs text-gray-500">
              单条嵌入最大字符
              <input
                type="number"
                value={binding.embed_max_chars || 700}
                onChange={(e) => updateBinding('embedding', { embed_max_chars: Number(e.target.value) })}
                className="mt-1 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
              />
            </label>
          </div>
        )}
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-gray-400">
        <div className="animate-spin w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full mr-2" />
        <span className="text-sm">加载配置中...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-800">模型设置</h2>
        <p className="text-sm text-gray-500 mt-1">
          管理 OpenAI-compatible 连接档案，并分别绑定大语言、嵌入和 Rerank 模型。
        </p>
      </div>

      {message && (
        <div className={`rounded-lg border px-4 py-3 text-sm ${
          message.includes('失败')
            ? 'border-red-200 bg-red-50 text-red-700'
            : 'border-emerald-200 bg-emerald-50 text-emerald-700'
        }`}>
          {message}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[380px_minmax(0,1fr)] gap-5">
        <section className="space-y-4">
          <div className="rounded-lg border border-gray-200 bg-white p-4">
            <h3 className="text-sm font-semibold text-gray-800">连接档案</h3>
            <div className="mt-3 space-y-3">
              {profiles.map((profile) => (
                <div key={profile.id} className="rounded-lg border border-gray-200 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium text-gray-800">{profile.name}</div>
                      <div className="mt-1 truncate text-xs text-gray-500">{profile.api_base}</div>
                      <div className="mt-1 text-xs text-gray-400">
                        Key: {profile.has_api_key ? profile.api_key_preview : '未保存'} · 模型 {profile.models_cache?.length || 0}
                      </div>
                    </div>
                    <div className="shrink-0 flex gap-1">
                      <button
                        onClick={() => refreshProfileModels(profile.id)}
                        className="rounded border border-gray-300 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50"
                      >
                        刷新
                      </button>
                      <button
                        onClick={() => editProfile(profile)}
                        className="rounded border border-gray-300 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50"
                      >
                        编辑
                      </button>
                      <button
                        onClick={() => removeProfile(profile)}
                        className="rounded border border-red-200 px-2 py-1 text-xs text-red-600 hover:bg-red-50"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-gray-200 bg-white p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-800">
                {profileDraft.id ? '更新连接' : '新增连接'}
              </h3>
              {profileDraft.id && (
                <button
                  onClick={() => setProfileDraft({ id: '', name: 'SiliconFlow', api_base: 'https://api.siliconflow.cn/v1', api_key: '' })}
                  className="text-xs text-gray-500 hover:text-gray-800"
                >
                  取消编辑
                </button>
              )}
            </div>
            <div className="mt-3 space-y-3">
              <input
                value={profileDraft.name}
                onChange={(e) => setProfileDraft({ ...profileDraft, name: e.target.value })}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                placeholder="连接名称"
              />
              <input
                value={profileDraft.api_base}
                onChange={(e) => setProfileDraft({ ...profileDraft, api_base: e.target.value })}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono"
                placeholder="https://api.example.com/v1"
              />
              <input
                type="password"
                value={profileDraft.api_key}
                onChange={(e) => setProfileDraft({ ...profileDraft, api_key: e.target.value })}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono"
                placeholder="API Key，保存后不回显"
              />
              <div className="flex gap-2">
                <button
                  onClick={handleDiscoverDraft}
                  disabled={discovering}
                  className="rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
                >
                  {discovering ? '发现中...' : '发现模型'}
                </button>
                <button
                  onClick={handleSaveProfile}
                  disabled={saving}
                  className="rounded-lg bg-gray-900 px-3 py-2 text-sm text-white hover:bg-gray-700 disabled:bg-gray-300"
                >
                  保存连接
                </button>
              </div>
              {draftModels.length > 0 && (
                <div className="max-h-40 overflow-y-auto rounded-lg bg-gray-50 p-2 text-xs text-gray-600">
                  {draftModels.map((model) => (
                    <div key={model.id} className="truncate py-1">{model.id}</div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="space-y-4">
          {renderBinding('chat', '大语言模型', '用于问答生成、图谱修正建议和 LightRAG 查询。')}
          {renderBinding('embedding', '嵌入模型', '用于索引和检索向量。模型或维度变化后需要重建索引。')}
          {renderBinding('rerank', 'Rerank 模型', '用于召回结果重排序；如果供应商不支持，可以关闭。')}

          <div className="flex justify-end">
            <button
              onClick={saveBindings}
              disabled={saving}
              className="rounded-lg bg-primary-600 px-5 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:bg-gray-300"
            >
              保存模型绑定
            </button>
          </div>
        </section>
      </div>

      <section className="bg-white border border-gray-200 rounded-lg p-5">
        <h3 className="text-sm font-semibold text-gray-800">生成参数</h3>
        <div className="mt-4 grid grid-cols-3 gap-4">
          <label className="text-sm text-gray-600">
            Temperature: {config.chat_temperature}
            <input type="range" min={0} max={2} step={0.05} value={config.chat_temperature}
              onChange={(e) => setConfig({ ...config, chat_temperature: parseFloat(e.target.value) })}
              className="mt-2 w-full accent-primary-500" />
          </label>
          <label className="text-sm text-gray-600">
            Top-P: {config.chat_top_p}
            <input type="range" min={0} max={1} step={0.05} value={config.chat_top_p}
              onChange={(e) => setConfig({ ...config, chat_top_p: parseFloat(e.target.value) })}
              className="mt-2 w-full accent-primary-500" />
          </label>
          <label className="text-sm text-gray-600">
            Max Tokens: {config.chat_max_tokens}
            <input type="range" min={256} max={8192} step={256} value={config.chat_max_tokens}
              onChange={(e) => setConfig({ ...config, chat_max_tokens: parseInt(e.target.value) })}
              className="mt-2 w-full accent-primary-500" />
          </label>
        </div>
      </section>

      <section className="bg-white border border-gray-200 rounded-lg p-5">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-800">问答提示词</h3>
            <p className="text-sm text-gray-500 mt-1">
              控制最终回答风格；参考资料会另行拼接到用户消息中。
            </p>
          </div>
          <button
            onClick={() => setConfig({ ...config, answer_system_prompt: DEFAULT_ANSWER_SYSTEM_PROMPT })}
            className="shrink-0 px-3 py-1.5 text-xs rounded-lg border border-gray-300 bg-white text-gray-600 hover:bg-gray-50"
          >
            恢复推荐模板
          </button>
        </div>
        <textarea
          value={config.answer_system_prompt}
          onChange={(e) => setConfig({ ...config, answer_system_prompt: e.target.value })}
          rows={10}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm leading-relaxed outline-none font-mono"
        />
        <div className="mt-3 flex items-center justify-between">
          <span className="text-xs text-gray-400">{config.answer_system_prompt.length} 字符</span>
          <button
            onClick={saveGenerationConfig}
            disabled={saving}
            className="rounded-lg bg-gray-900 px-4 py-2 text-sm text-white hover:bg-gray-700 disabled:bg-gray-300"
          >
            保存生成参数和提示词
          </button>
        </div>
      </section>
    </div>
  )
}
