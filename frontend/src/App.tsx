import { FormEvent, useEffect, useState } from 'react'
import { ask, deleteConversation, job, login, publish, reviewVersion, uploadVersion, versions, withdraw } from './api'
import type { Answer, Citation, Job, Review, Version } from './types'

const HISTORY_KEY = 'lawrag_chat_history_v1'
const ACTIVE_CONVERSATION_KEY = 'lawrag_active_conversation_id'
const HISTORY_LIMIT = 20

type HistoryEntry = {
  id: string
  question: string
  asOfDate: string
  answer: Answer
  endToEndMs: number
  createdAt: string
  conversationId?: string
}

function loadHistory(): HistoryEntry[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    return Array.isArray(value) ? value as HistoryEntry[] : []
  } catch {
    return []
  }
}

function Locator({ citation }: { citation: Citation }) {
  return <span>Điều {citation.article}{citation.clause ? ` · Khoản ${citation.clause}` : ''}{citation.point ? ` · Điểm ${citation.point}` : ''}</span>
}

function CitationCard({ citation }: { citation: Citation }) {
  return <article className="citation">
    <div className="citation-top"><strong><Locator citation={citation} /></strong><span>{citation.document_code}</span></div>
    <p>{citation.excerpt}</p>
    <a href={citation.official_url} target="_blank" rel="noreferrer">Mở văn bản gốc — {citation.document_title}</a>
  </article>
}

function Chat() {
  const [question, setQuestion] = useState('')
  const [asOfDate, setAsOfDate] = useState('')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [endToEndMs, setEndToEndMs] = useState<number | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory)
  const [selectedHistoryId, setSelectedHistoryId] = useState('')
  const [conversationId, setConversationId] = useState(() => localStorage.getItem(ACTIVE_CONVERSATION_KEY) || '')
  const [loading, setLoading] = useState(false)
  const [clearingConversation, setClearingConversation] = useState(false)
  const [error, setError] = useState('')
  // Citations are already server-validated. Keep this filter in the UI as a
  // second guard so the user only sees sources that support a generated claim.
  const citedIds = answer ? new Set(answer.claims.flatMap(claim => claim.citation_ids)) : new Set<string>()
  const supportingCitations = answer ? answer.citations.filter(citation => citedIds.has(citation.id)) : []
  useEffect(() => { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)) }, [history])
  useEffect(() => { if (conversationId) localStorage.setItem(ACTIVE_CONVERSATION_KEY, conversationId); else localStorage.removeItem(ACTIVE_CONVERSATION_KEY) }, [conversationId])
  const submit = async () => {
    if (clearingConversation) return
    setLoading(true); setError(''); setAnswer(null); setEndToEndMs(null)
    try {
      const result = await ask(question, asOfDate, conversationId)
      setAnswer(result.answer)
      setEndToEndMs(result.endToEndMs)
      const nextConversationId = result.answer.conversation_id || conversationId
      setConversationId(nextConversationId)
      const entry: HistoryEntry = {
        id: crypto.randomUUID(), question: question.trim(), asOfDate, answer: result.answer,
        endToEndMs: result.endToEndMs, createdAt: new Date().toISOString(), conversationId: nextConversationId || undefined,
      }
      setHistory(items => [entry, ...items].slice(0, HISTORY_LIMIT))
      setSelectedHistoryId(entry.id)
      setQuestion('')
    } catch (err) { setError(err instanceof Error ? err.message : 'Đã có lỗi xảy ra') } finally { setLoading(false) }
  }
  const openHistory = (entry: HistoryEntry) => {
    setAnswer(entry.answer); setEndToEndMs(entry.endToEndMs); setSelectedHistoryId(entry.id); setConversationId(entry.conversationId || ''); setError('')
  }
  const deleteHistory = async (entry: HistoryEntry) => {
    const removesActiveConversation = conversationId === entry.conversationId || selectedHistoryId === entry.id
    if (removesActiveConversation) startNewChat()
    setClearingConversation(true)
    try {
      if (entry.conversationId) await deleteConversation(entry.conversationId)
      setHistory(items => items.filter(item => entry.conversationId ? item.conversationId !== entry.conversationId : item.id !== entry.id))
    } catch (err) { setError(err instanceof Error ? err.message : 'Không thể xóa đoạn chat') }
    finally { setClearingConversation(false) }
  }
  const startNewChat = () => {
    setQuestion(''); setAsOfDate(''); setAnswer(null); setEndToEndMs(null); setError(''); setSelectedHistoryId(''); setConversationId('')
  }
  const clearHistory = async () => {
    startNewChat()
    setClearingConversation(true)
    try {
      const ids = [...new Set(history.map(entry => entry.conversationId).filter((id): id is string => Boolean(id)))]
      await Promise.all(ids.map(deleteConversation))
      setHistory([])
    } catch (err) { setError(err instanceof Error ? err.message : 'Không thể xóa lịch sử') }
    finally { setClearingConversation(false) }
  }
  return <div className="chat-shell">
    <aside className="chat-history" aria-label="Lịch sử câu hỏi">
      <div className="history-heading"><h2>Lịch sử hỏi đáp</h2><div className="history-actions"><button className="history-new" type="button" disabled={clearingConversation} onClick={startNewChat}>Đoạn chat mới</button>{history.length > 0 && <button className="history-clear" type="button" disabled={clearingConversation} onClick={() => void clearHistory()}>Xóa tất cả</button>}</div></div>
      {history.length === 0 ? <p className="history-empty">Chưa có câu hỏi nào.</p> : <ul>{history.map(entry => <li key={entry.id} className={entry.id === selectedHistoryId ? 'selected' : ''}><button className="history-open" type="button" disabled={clearingConversation} onClick={() => openHistory(entry)}><span>{entry.question}</span><small>{entry.answer.status === 'grounded' ? 'Có căn cứ' : 'Chưa đủ căn cứ'}</small></button><button className="history-delete" type="button" disabled={clearingConversation} aria-label={`Xóa đoạn chat: ${entry.question}`} onClick={() => void deleteHistory(entry)}>×</button></li>)}</ul>}
      <p className="history-note">Xóa một mục sẽ xóa ngữ cảnh đoạn chat đó.</p>
    </aside>
    <main className="chat-page">
    <nav><span className="brand">ChatBot Luật RAG</span><a href="#admin">Quản trị kho luật</a></nav>
    <form className="ask" onSubmit={event => { event.preventDefault(); void submit() }}>
      <label>Câu hỏi pháp lý<textarea value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (!loading && !clearingConversation && question.trim().length >= 8) void submit() } }} minLength={8} required placeholder="Ví dụ: Mức phạt vi phạm hợp đồng mua bán hàng hóa tối đa là bao nhiêu?" /></label>
      <small>Enter để hỏi · Shift + Enter để xuống dòng</small>
      <button disabled={loading || clearingConversation}>{loading ? 'Đang đối chiếu nguồn…' : clearingConversation ? 'Đang xóa đoạn chat…' : 'Hỏi pháp luật'}</button>
    </form>
    {error && <p className="error">{error}</p>}
    {answer && <section className={`answer ${answer.status}`}>
      <div className="answer-state">{answer.status === 'grounded' ? `Có căn cứ · áp dụng ${answer.applied_as_of_date}` : 'Chưa đủ căn cứ'}</div>
      {endToEndMs !== null && <p className="latency">Phản hồi trong {(endToEndMs / 1000).toFixed(2)} giây · Server xử lý {((answer.latency?.total_ms ?? 0) / 1000).toFixed(2)} giây</p>}
      <p className="answer-text">{answer.answer}</p>
      {supportingCitations.length > 0 && <><h2>Căn cứ pháp lý</h2><div className="citations">{supportingCitations.map(c => <div id={`cite-${c.id}`} key={c.id}><CitationCard citation={c} /></div>)}</div></>}
      {answer.warnings.map((warning, i) => <p className="warning" key={i}>{warning}</p>)}
      {answer.latency && <details className="latency-breakdown"><summary>Chi tiết độ trễ máy chủ</summary><ul><li>Retrieval: {answer.latency.retrieval_ms.toFixed(0)} ms</li><li>Rerank: {answer.latency.rerank_ms.toFixed(0)} ms</li><li>LLM: {answer.latency.llm_ms.toFixed(0)} ms</li><li>Kiểm tra citation: {answer.latency.citation_validation_ms.toFixed(0)} ms</li></ul></details>}
    </section>}
    <footer>Thông tin tra cứu mang tính tham khảo, không thay thế tư vấn luật sư cho tình huống cụ thể.</footer>
    </main>
  </div>
}

function Admin() {
  const [token, setToken] = useState(sessionStorage.getItem('lawrag_admin_token') || '')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [items, setItems] = useState<Version[]>([])
  const [notice, setNotice] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [activeJob, setActiveJob] = useState<Job | null>(null)
  const [reviews, setReviews] = useState<Record<string, Review>>({})
  const [reviewed, setReviewed] = useState<Record<string, boolean>>({})
  const [publishingId, setPublishingId] = useState('')
  const [fields, setFields] = useState({ document_code: '', title: '', version_label: '', official_url: '', effective_from: '', effective_to: '' })
  const refresh = () => { if (token) void versions(token).then(setItems).catch(e => setNotice(e.message)) }
  useEffect(refresh, [token])
  const auth = async (event: FormEvent) => {
    event.preventDefault()
    try { const value = await login(email, password); sessionStorage.setItem('lawrag_admin_token', value); setToken(value) } catch (e) { setNotice(e instanceof Error ? e.message : 'Lỗi') }
  }
  useEffect(() => {
    if (!activeJob || !['QUEUED', 'RUNNING'].includes(activeJob.status)) return
    const timer = window.setInterval(async () => {
      try { const next = await job(token, activeJob.id); setActiveJob(next); if (!['QUEUED', 'RUNNING'].includes(next.status)) refresh() } catch { /* retain the previous visible state */ }
    }, 2000)
    return () => clearInterval(timer)
  }, [activeJob, token])
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!file) return setNotice('Chọn tệp PDF hoặc DOCX')
    try { const created = await uploadVersion(token, fields, file); setActiveJob(created); setNotice(`Đã xếp hàng xử lý ${created.id}.`); setFile(null); refresh() } catch (e) { setNotice(e instanceof Error ? e.message : 'Lỗi') }
  }
  const inspect = async (item: Version) => {
    try { const value = await reviewVersion(token, item.id); setReviews({ ...reviews, [item.id]: value }); setReviewed({ ...reviewed, [item.id]: false }) } catch (e) { setNotice(e instanceof Error ? e.message : 'Lỗi') }
  }
  const publishReviewed = async (item: Version) => {
    const review = reviews[item.id]
    if (!review || !reviewed[item.id]) {
      setNotice('Hãy mở cấu trúc và tick xác nhận trước khi xuất bản.')
      return
    }
    setPublishingId(item.id)
    setNotice('Đang gửi yêu cầu xuất bản và lập chỉ mục…')
    try { await publish(token, item.id, review.structure_hash); setNotice('Đã xuất bản phiên bản đã duyệt.'); refresh() } catch (e) { setNotice(e instanceof Error ? e.message : 'Lỗi') } finally { setPublishingId('') }
  }
  const withdrawPublished = async (item: Version) => {
    if (!window.confirm('Thu hồi phiên bản này khỏi chatbot để kiểm tra lại?')) return
    try { await withdraw(token, item.id); setNotice('Đã thu hồi phiên bản; chatbot sẽ không còn dùng dữ liệu này.'); refresh() } catch (e) { setNotice(e instanceof Error ? e.message : 'Lỗi') }
  }
  if (!token) return <main className="admin-page"><a href="#/">← Tra cứu</a><h1>Quản trị kho luật</h1><form className="panel" onSubmit={auth}><input value={email} onChange={e => setEmail(e.target.value)} placeholder="Email admin" type="email" required /><input value={password} onChange={e => setPassword(e.target.value)} placeholder="Mật khẩu" type="password" required /><button>Đăng nhập</button></form>{notice && <p className="error">{notice}</p>}</main>
  return <main className="admin-page">
    <nav><a href="#/">← Tra cứu</a><button className="link-button" onClick={() => { sessionStorage.removeItem('lawrag_admin_token'); setToken('') }}>Đăng xuất</button></nav>
    <h1>Kho văn bản pháp luật</h1><p className="muted">Chỉ xuất bản sau khi kiểm tra cấu trúc Điều/Khoản/Điểm và URL nguồn chính thức.</p>
    <form className="panel upload" onSubmit={submit}>{Object.entries(fields).map(([key, value]) => <label key={key}>{key === 'document_code' ? 'Mã văn bản' : key === 'title' ? 'Tên văn bản' : key === 'version_label' ? 'Nhãn phiên bản' : key === 'official_url' ? 'URL chính thức' : key === 'effective_from' ? 'Hiệu lực từ' : 'Hiệu lực đến'}<input required={key !== 'effective_to'} type={key.includes('effective') ? 'date' : key === 'official_url' ? 'url' : 'text'} value={value} onChange={e => setFields({ ...fields, [key]: e.target.value })} /></label>)}<label>Tệp PDF/DOCX<input required type="file" accept=".pdf,.docx" onChange={e => setFile(e.target.files?.[0] || null)} /></label><button>Đưa vào hàng đợi xử lý</button></form>
    {activeJob && <p className="notice">Xử lý: {activeJob.status} · {activeJob.progress}% {activeJob.message || ''}</p>}{notice && <p className="notice">{notice}</p>}<button type="button" onClick={refresh}>Làm mới</button>
    <section className="version-list">{items.map(item => {
      const review = reviews[item.id]
      return <article className="version" key={item.id}><div><strong>{item.document_code} — {item.version_label}</strong><p>{item.document_title}</p><small>Hiệu lực: {item.effective_from} {item.effective_to ? `đến ${item.effective_to}` : 'đến nay'}</small>
        {review && <details className="review" open><summary>Đã tải {review.provision_count} đơn vị để kiểm tra</summary>{review.provisions.map(p => <div className="provision" key={p.id}><strong>Điều {p.article_no}{p.clause_no ? ` · Khoản ${p.clause_no}` : ''}{p.point_label ? ` · Điểm ${p.point_label}` : ''}</strong>{p.heading && <span> — {p.heading}</span>}<p>{p.content}</p></div>)}<label className="review-check"><input type="checkbox" checked={Boolean(reviewed[item.id])} onChange={e => setReviewed({ ...reviewed, [item.id]: e.target.checked })} /> Tôi đã kiểm tra cấu trúc và đoạn trích phía trên.</label></details>}
      </div><div><span className={`badge ${item.status.toLowerCase()}`}>{item.status}</span>{item.status === 'PENDING_REVIEW' && <><button type="button" onClick={() => void inspect(item)}>Kiểm tra cấu trúc</button>{review && <button type="button" disabled={!reviewed[item.id] || publishingId === item.id} onClick={() => void publishReviewed(item)}>{publishingId === item.id ? 'Đang xuất bản…' : 'Duyệt & xuất bản'}</button>}</>}{item.status === 'PUBLISHED' && <button type="button" onClick={() => void withdrawPublished(item)}>Thu hồi</button>}</div></article>
    })}</section>
  </main>
}

export default function App() {
  const [hash, setHash] = useState(window.location.hash)

  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  return hash.startsWith('#admin') ? <Admin /> : <Chat />
}
