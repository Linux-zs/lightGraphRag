import { Component, type ReactNode } from 'react'

/** Keep navigation usable if a page chunk fails to load after deployment. */
export default class PageBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    if (this.state.failed) {
      return <div role="alert" className="p-6 space-y-3">
        <p>页面加载或运行失败。可以切换其他页面，或刷新后重试。</p>
        <p className="text-sm text-gray-600">刷新会丢失当前未提交的输入。</p>
        <button className="ui-button-secondary" onClick={() => window.location.reload()}>刷新页面</button>
      </div>
    }
    return this.props.children
  }
}
