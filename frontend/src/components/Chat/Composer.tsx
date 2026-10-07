import type { FormEvent } from 'react'

type Props = {
  question: string
  loading: boolean
  disabled: boolean
  onQuestionChange: (value: string) => void
  onSubmit: () => void
}

export function Composer({ question, loading, disabled, onQuestionChange, onSubmit }: Props) {
  const submit = (event: FormEvent) => { event.preventDefault(); onSubmit() }
  return <form className="ask composer" onSubmit={submit}>
    <label className="sr-only" htmlFor="legal-question">Câu hỏi pháp lý</label>
    <textarea id="legal-question" aria-describedby="composer-help" value={question} onChange={e => onQuestionChange(e.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); onSubmit() } }} minLength={8} required placeholder="Nhập câu hỏi pháp lý của bạn…" />
    <div className="composer-actions"><small id="composer-help">Áp dụng quy định hiện hành · Enter để gửi · Shift + Enter để xuống dòng</small><button disabled={disabled || question.trim().length < 8}>{loading ? 'Đang trả lời…' : 'Gửi câu hỏi'}</button></div>
  </form>
}
