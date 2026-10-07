import { useEffect, useRef, useState } from 'react'
import { ask, deleteConversation } from '../../api'
import type { ChatTurn, HistoryEntry } from '../../types'
import { ErrorBoundary } from '../ErrorBoundary'
import { AssistantTurn } from './AssistantTurn'
import { Composer } from './Composer'

const HISTORY_KEY = 'lawrag_chat_history_v1'
const ACTIVE_CONVERSATION_KEY = 'lawrag_active_conversation_id'
const HISTORY_VISIBLE_LIMIT = 20
const TRANSCRIPT_TURN_LIMIT = 20
const SUGGESTED_QUESTIONS = [
  'Mức phạt vi phạm hợp đồng mua bán hàng hóa tối đa là bao nhiêu?',
  'Khi nào được miễn trách nhiệm do vi phạm hợp đồng thương mại?',
  'Hợp đồng mua bán hàng hóa cần có những nội dung gì?',
]

function loadHistory(): HistoryEntry[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    if (!Array.isArray(value)) return []
    const grouped = new Map<string, HistoryEntry>()
    const ordered = (value as HistoryEntry[]).sort((a, b) => (a.createdAt || '').localeCompare(b.createdAt || ''))
    for (const item of ordered) {
      const key = item.conversationId || `legacy-${item.id}`
      const legacyTurn: ChatTurn = { id: `legacy-${item.id}`, question: item.lastQuestion || item.question, asOfDate: item.asOfDate || '', answer: item.answer, endToEndMs: item.endToEndMs || 0, createdAt: item.updatedAt || item.createdAt }
      const normalized: HistoryEntry = { ...item, lastQuestion: item.lastQuestion || item.question, updatedAt: item.updatedAt || item.createdAt, turnCount: item.turnCount || 1, questions: item.questions?.length ? item.questions : [item.question], turns: item.turns?.length ? item.turns.slice(-TRANSCRIPT_TURN_LIMIT) : [legacyTurn] }
      const existing = grouped.get(key)
      grouped.set(key, existing ? { ...existing, lastQuestion: normalized.lastQuestion, asOfDate: normalized.asOfDate, answer: normalized.answer, endToEndMs: normalized.endToEndMs, updatedAt: normalized.updatedAt, turnCount: existing.turnCount + normalized.turnCount, questions: [...existing.questions, ...normalized.questions], turns: [...(existing.turns || []), ...(normalized.turns || [])].slice(-TRANSCRIPT_TURN_LIMIT) } : normalized)
    }
    // Keep all conversations in local storage. The sidebar only progressively
    // discloses them, instead of silently deleting anything after 20 records.
    return [...grouped.values()].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  } catch { return [] }
}

export function ChatShell() {
  const [question, setQuestion] = useState('')
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory)
  const [selectedHistoryId, setSelectedHistoryId] = useState('')
  const [conversationId, setConversationId] = useState(() => localStorage.getItem(ACTIVE_CONVERSATION_KEY) || '')
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [pendingQuestion, setPendingQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [clearingConversation, setClearingConversation] = useState(false)
  const [error, setError] = useState('')
  const [showAllHistory, setShowAllHistory] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)) }, [history])
  useEffect(() => { if (conversationId) localStorage.setItem(ACTIVE_CONVERSATION_KEY, conversationId); else localStorage.removeItem(ACTIVE_CONVERSATION_KEY) }, [conversationId])
  useEffect(() => {
    if (!conversationId || turns.length) return
    const active = history.find(entry => entry.conversationId === conversationId)
    if (active?.turns?.length) { setTurns(active.turns); setSelectedHistoryId(active.id) }
  }, [conversationId, history, turns.length])
  useEffect(() => {
    if (!bottomRef.current || (!pendingQuestion && turns.length === 0)) return
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    bottomRef.current.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'end' })
  }, [pendingQuestion, turns.length, loading])

  const startNewChat = () => { setQuestion(''); setTurns([]); setPendingQuestion(''); setError(''); setSelectedHistoryId(''); setConversationId('') }
  const submit = async () => {
    const submittedQuestion = question.trim()
    if (clearingConversation || loading || submittedQuestion.length < 8) return
    setLoading(true); setError(''); setPendingQuestion(submittedQuestion); setQuestion('')
    try {
      const result = await ask(submittedQuestion, undefined, conversationId)
      const nextConversationId = result.answer.conversation_id || conversationId
      setConversationId(nextConversationId)
      const now = new Date().toISOString()
      const nextTurn: ChatTurn = { id: crypto.randomUUID(), question: submittedQuestion, asOfDate: '', answer: result.answer, endToEndMs: result.endToEndMs, createdAt: now }
      setTurns(current => [...current, nextTurn].slice(-TRANSCRIPT_TURN_LIMIT))
      const existingEntry = history.find(item => item.conversationId === nextConversationId)
      const entryId = existingEntry?.id || crypto.randomUUID()
      setHistory(items => {
        const existing = items.find(item => item.conversationId === nextConversationId)
        const entry: HistoryEntry = existing
          ? { ...existing, lastQuestion: submittedQuestion, asOfDate: '', answer: result.answer, endToEndMs: result.endToEndMs, updatedAt: now, turnCount: existing.turnCount + 1, questions: [...existing.questions, submittedQuestion], turns: [...(existing.turns || []), nextTurn].slice(-TRANSCRIPT_TURN_LIMIT) }
          : { id: entryId, question: submittedQuestion, lastQuestion: submittedQuestion, asOfDate: '', answer: result.answer, endToEndMs: result.endToEndMs, createdAt: now, updatedAt: now, turnCount: 1, questions: [submittedQuestion], turns: [nextTurn], conversationId: nextConversationId || undefined }
        return [entry, ...items.filter(item => item.id !== entry.id)]
      })
      setSelectedHistoryId(entryId)
    } catch (err) { setError(err instanceof Error ? err.message : 'Đã có lỗi xảy ra') }
    finally { setLoading(false); setPendingQuestion('') }
  }
  const openHistory = (entry: HistoryEntry) => {
    setTurns(entry.turns?.length ? entry.turns : [{ id: `legacy-${entry.id}`, question: entry.lastQuestion || entry.question, asOfDate: entry.asOfDate, answer: entry.answer, endToEndMs: entry.endToEndMs, createdAt: entry.updatedAt }])
    setSelectedHistoryId(entry.id); setConversationId(entry.conversationId || ''); setError(''); setQuestion('')
  }
  const deleteHistory = async (entry: HistoryEntry) => {
    if (!window.confirm(`Xóa đoạn chat “${entry.question}”? Ngữ cảnh liên quan cũng sẽ bị xóa.`)) return
    const removesActiveConversation = conversationId === entry.conversationId || selectedHistoryId === entry.id
    if (removesActiveConversation) startNewChat()
    setClearingConversation(true)
    try {
      if (entry.conversationId) await deleteConversation(entry.conversationId)
      setHistory(items => items.filter(item => entry.conversationId ? item.conversationId !== entry.conversationId : item.id !== entry.id))
    } catch (err) { setError(err instanceof Error ? err.message : 'Không thể xóa đoạn chat') }
    finally { setClearingConversation(false) }
  }
  const clearHistory = async () => {
    if (!window.confirm(`Xóa toàn bộ ${history.length} đoạn chat? Thao tác này không thể hoàn tác.`)) return
    startNewChat(); setClearingConversation(true)
    try {
      const ids = [...new Set(history.map(entry => entry.conversationId).filter((id): id is string => Boolean(id)))]
      await Promise.all(ids.map(deleteConversation)); setHistory([]); setShowAllHistory(false)
    } catch (err) { setError(err instanceof Error ? err.message : 'Không thể xóa lịch sử') }
    finally { setClearingConversation(false) }
  }
  const displayedHistory = showAllHistory ? history : history.slice(0, HISTORY_VISIBLE_LIMIT)
  const statusMessage = error ? `Lỗi: ${error}` : loading ? 'Đang truy xuất, xếp hạng và kiểm tra căn cứ pháp lý.' : pendingQuestion ? 'Đã gửi câu hỏi.' : ''

  return <div className="chat-shell">
    <a className="skip-link" href="#chat-main">Bỏ qua lịch sử, tới phần hỏi đáp</a>
    <p className="sr-only" aria-live="polite" aria-atomic="true">{statusMessage}</p>
    <aside className="chat-history" aria-label="Lịch sử câu hỏi">
      <div className="history-heading"><h2>Lịch sử hỏi đáp</h2><div className="history-actions"><button className="history-new" type="button" disabled={clearingConversation} onClick={startNewChat}>Đoạn chat mới</button>{history.length > 0 && <button className="history-clear" type="button" disabled={clearingConversation} onClick={() => void clearHistory()}>Xóa tất cả</button>}</div></div>
      {history.length === 0 ? <p className="history-empty">Chưa có đoạn chat nào.</p> : <ul>{displayedHistory.map(entry => <li key={entry.id} className={entry.id === selectedHistoryId ? 'selected' : ''}><button className="history-open" type="button" disabled={clearingConversation} onClick={() => openHistory(entry)}><span>{entry.question}</span><small>{entry.answer.status === 'grounded' ? 'Có căn cứ' : 'Chưa đủ căn cứ'} · {entry.turnCount} lượt hỏi</small></button><button className="history-delete" type="button" disabled={clearingConversation} aria-label={`Xóa đoạn chat: ${entry.question}`} onClick={() => void deleteHistory(entry)}>×</button><details className="history-questions"><summary>Xem các câu hỏi</summary><ol>{entry.questions.map((askedQuestion, index) => <li key={`${entry.id}-${index}`}>{askedQuestion}</li>)}</ol></details></li>)}</ul>}
      {history.length > HISTORY_VISIBLE_LIMIT && <button className="history-more" type="button" onClick={() => setShowAllHistory(value => !value)} aria-expanded={showAllHistory}>{showAllHistory ? 'Thu gọn lịch sử' : `Hiển thị thêm ${history.length - HISTORY_VISIBLE_LIMIT} đoạn chat`}</button>}
      <p className="history-note">Xóa một mục sẽ xóa ngữ cảnh đoạn chat đó.</p>
    </aside>
    <main className="chat-page" id="chat-main">
      <nav aria-label="Điều hướng chính"><a className="brand" href="#/"><span className="brand-mark" aria-hidden="true">§</span><span>ChatBot Luật</span></a><a className="admin-link" href="#admin">Quản trị kho luật <span aria-hidden="true">→</span></a></nav>
      <section className="chat-transcript" aria-busy={loading} aria-label="Nội dung đoạn chat">
        {turns.length === 0 && !pendingQuestion && <div className="chat-welcome"><p>Tra cứu Luật Thương mại 2005 và Bộ luật Dân sự 2015.</p><div className="suggestions" aria-label="Câu hỏi gợi ý">{SUGGESTED_QUESTIONS.map(item => <button type="button" key={item} onClick={() => setQuestion(item)}>{item}</button>)}</div></div>}
        {turns.map(turn => <div className="turn" key={turn.id}><div className="message user-message"><span>Bạn</span><p>{turn.question}</p></div><ErrorBoundary fallbackLabel="Không thể hiển thị phản hồi này."><AssistantTurn turn={turn} /></ErrorBoundary></div>)}
        {pendingQuestion && <div className="turn"><div className="message user-message"><span>Bạn</span><p>{pendingQuestion}</p></div><div className="message assistant-message pending-message" role="status"><span><i className="loading-dot" aria-hidden="true" />Đang đối chiếu nguồn</span><p>Đang truy xuất, xếp hạng và kiểm tra căn cứ pháp lý…</p></div></div>}
        <div ref={bottomRef} aria-hidden="true" />
      </section>
      {error && <p className="error" role="alert">{error}</p>}
      <Composer question={question} loading={loading} disabled={clearingConversation || loading} onQuestionChange={setQuestion} onSubmit={() => void submit()} />
      <footer>Thông tin tra cứu mang tính tham khảo, không thay thế tư vấn luật sư cho tình huống cụ thể.</footer>
    </main>
  </div>
}
