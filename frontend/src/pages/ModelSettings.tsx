import { useEffect, useState } from 'react'
import {
  getModelConfig,
  updateModelConfig,
  testEmbed,
  ModelConfig,
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

export default function ModelSettings() {
  const [config, setConfig] = useState<ModelConfig>(DEFAULT_CONFIG)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState('')

  // Embed test
  const [testText, setTestText] = useState('你好，这是一段测试文本')
  const [embedResult, setEmbedResult] = useState<{
    dimensions: number
    preview: number[]
  } | null>(null)
  const [embedTesting, setEmbedTesting] = useState(false)
  const [embedError, setEmbedError] = useState('')

  useEffect(() => {
    loadConfig()
  }, [])

  const loadConfig = async () => {
    try {
      const data = await getModelConfig()
      setConfig(data)
    } catch {
      // use defaults
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    setSaveMsg('')
    try {
      await updateModelConfig(config)
      setSaveMsg('配置已保存到 config/default.yaml')
    } catch (e: unknown) {
      setSaveMsg(`保存失败: ${(e as Error).message}`)
    } finally {
      setSaving(false)
    }
  }

  const handleTestEmbed = async () => {
    if (!testText.trim()) return
    setEmbedTesting(true)
    setEmbedError('')
    setEmbedResult(null)
    try {
      const data = await testEmbed(testText.trim())
      setEmbedResult(data)
    } catch (e: unknown) {
      setEmbedError((e as Error).message || 'Embed test failed')
    } finally {
      setEmbedTesting(false)
    }
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
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-bold text-gray-800">模型设置</h2>
        <p className="text-sm text-gray-500 mt-1">
          配置嵌入模型、重排序模型和大语言模型参数，修改后保存到 YAML 配置文件。
        </p>
      </div>

      {/* Model config form */}
      <section className="bg-white border border-gray-200 rounded-xl p-6">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-5">
          模型配置
        </h3>

        <div className="space-y-5">
          {/* Embed model */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Embed Model
              </label>
              <input
                type="text"
                value={config.embed_model}
                onChange={(e) => setConfig({ ...config, embed_model: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-200 focus:border-primary-400 outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Embed API Base URL
              </label>
              <input
                type="text"
                value={config.embed_base_url}
                onChange={(e) =>
                  setConfig({ ...config, embed_base_url: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono focus:ring-2 focus:ring-primary-200 focus:border-primary-400 outline-none"
              />
            </div>
          </div>

          {/* Rerank model */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Rerank Model
            </label>
            <input
              type="text"
              value={config.rerank_model}
              onChange={(e) => setConfig({ ...config, rerank_model: e.target.value })}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-200 focus:border-primary-400 outline-none"
            />
          </div>

          {/* Chat model */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Chat Model
            </label>
            <input
              type="text"
              value={config.chat_model}
              onChange={(e) => setConfig({ ...config, chat_model: e.target.value })}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-200 focus:border-primary-400 outline-none"
            />
          </div>

          {/* Chat params */}
          <div className="grid grid-cols-3 gap-4 pt-2 border-t border-gray-100">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Temperature: {config.chat_temperature}
              </label>
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={config.chat_temperature}
                onChange={(e) =>
                  setConfig({ ...config, chat_temperature: parseFloat(e.target.value) })
                }
                className="w-full accent-primary-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Top-P: {config.chat_top_p}
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={config.chat_top_p}
                onChange={(e) =>
                  setConfig({ ...config, chat_top_p: parseFloat(e.target.value) })
                }
                className="w-full accent-primary-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Tokens: {config.chat_max_tokens}
              </label>
              <input
                type="range"
                min={256}
                max={8192}
                step={256}
                value={config.chat_max_tokens}
                onChange={(e) =>
                  setConfig({ ...config, chat_max_tokens: parseInt(e.target.value) })
                }
                className="w-full accent-primary-500"
              />
            </div>
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-5 py-2 text-sm font-medium rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {saving ? '保存中...' : '保存配置'}
          </button>
          {saveMsg && (
            <span
              className={`ml-3 text-sm ${
                saveMsg.includes('失败') ? 'text-red-500' : 'text-green-600'
              }`}
            >
              {saveMsg}
            </span>
          )}
        </div>
      </section>

      <section className="bg-white border border-gray-200 rounded-xl p-6">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide">
              问答提示词
            </h3>
            <p className="text-sm text-gray-500 mt-1">
              这里控制智能问答的系统提示词；参考资料会另行拼接到用户消息中。
            </p>
          </div>
          <button
            onClick={() =>
              setConfig({ ...config, answer_system_prompt: DEFAULT_ANSWER_SYSTEM_PROMPT })
            }
            className="shrink-0 px-3 py-1.5 text-xs rounded-lg border border-gray-300 bg-white text-gray-600 hover:bg-gray-50"
          >
            恢复推荐模板
          </button>
        </div>

        <textarea
          value={config.answer_system_prompt}
          onChange={(e) =>
            setConfig({ ...config, answer_system_prompt: e.target.value })
          }
          rows={10}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm leading-relaxed focus:ring-2 focus:ring-primary-200 focus:border-primary-400 outline-none font-mono"
        />
        <div className="mt-3 flex items-center justify-between text-xs text-gray-400">
          <span>保存后新问题立即使用；已生成的历史回答不会自动重写。</span>
          <span>{config.answer_system_prompt.length} 字符</span>
        </div>
      </section>

      {/* Embed test */}
      <section className="bg-white border border-gray-200 rounded-xl p-6">
        <h3 className="text-sm font-semibold text-gray-600 uppercase tracking-wide mb-4">
          嵌入测试
        </h3>
        <p className="text-sm text-gray-500 mb-4">
          输入一段文本，查看当前嵌入模型的输出向量维度和前 10 个值。
        </p>

        <div className="flex gap-3">
          <input
            type="text"
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleTestEmbed()}
            placeholder="输入测试文本..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-primary-200 focus:border-primary-400 outline-none"
          />
          <button
            onClick={handleTestEmbed}
            disabled={embedTesting || !testText.trim()}
            className="px-5 py-2.5 text-sm font-medium rounded-lg bg-white border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {embedTesting ? '计算中...' : '测试'}
          </button>
        </div>

        {embedError && (
          <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {embedError}
          </div>
        )}

        {embedResult && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="flex items-center gap-4 mb-3">
              <span className="text-sm text-gray-500">
                维度: <span className="font-bold text-gray-800">{embedResult.dimensions}</span>
              </span>
              <span className="text-sm text-gray-500">
                模型: <span className="font-mono">{config.embed_model}</span>
              </span>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">前 10 个值:</p>
              <div className="flex flex-wrap gap-1.5">
                {embedResult.preview.map((v, i) => (
                  <span
                    key={i}
                    className="text-[11px] font-mono bg-white border border-gray-200 rounded px-1.5 py-0.5"
                  >
                    {v.toFixed(6)}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
