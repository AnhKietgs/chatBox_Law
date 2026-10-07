import { Component, type ErrorInfo, type ReactNode } from 'react'

type Props = { children: ReactNode; fallbackLabel?: string }
type State = { hasError: boolean }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Keep diagnostics available without exposing internals in the UI.
    console.error('UI rendering error', error, info)
  }

  render() {
    if (this.state.hasError) {
      return <section className="component-error" role="alert">
        <strong>{this.props.fallbackLabel || 'Không thể hiển thị phần nội dung này.'}</strong>
        <button type="button" onClick={() => this.setState({ hasError: false })}>Thử lại</button>
      </section>
    }
    return this.props.children
  }
}
