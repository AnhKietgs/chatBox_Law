import { FormEvent, useEffect, useState } from 'react'
import { ask, deleteConversation, job, login, publish, publishedChunks, reviewVersion, uploadVersion, versions, withdraw } from './api'
import type { Answer, Citation, Job, ProvisionPage, Review, Version } from './types'

const HISTORY_KEY = 'lawrag_chat_history_v1'
const ACTIVE_CONVERSATION_KEY = 'lawrag_active_conversation_id'
const HISTORY_LIMIT = 20
const TRANSCRIPT_TURN_LIMIT = 20

type ChatTurn = {
  id: string
  question: string
  asOfDate: string
  answer: Answer
  endToEndMs: number
  createdAt: string
}

type HistoryEntry = {
  id: string
  // The first question names a conversation; later questions update this same entry.
  question: string
  lastQuestion: string
  asOfDate: string
  answer: Answer
  endToEndMs: number
  createdAt: string
  updatedAt: string
  turnCount: number
  questions: string[]
  conversationId?: string
  turns?: ChatTurn[]
}

function loadHistory(): HistoryEntry[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
    if (!Array.isArray(value)) return []
    // Upgrade the old per-question local history into one item per conversation.
    const grouped = new Map<string, HistoryEntry>()
    const ordered = (value as HistoryEntry[]).sort((a, b) => (a.createdAt || '').localeCompare(b.createdAt || ''))
    for (const item of ordered) {
      const key = item.conversationId || `legacy-${item.id}`
      const legacyTurn: ChatTurn = {
        id: `legacy-${item.id}`,
        question: item.lastQuestion || item.question,
        asOfDate: item.asOfDate || '',
        answer: item.answer,
        endToEndMs: item.endToEndMs || 0,
        createdAt: item.updatedAt || item.createdAt,
      }
      const normalized: HistoryEntry = {
        ...item,
        lastQuestion: item.lastQuestion || item.question,
        updatedAt: item.updatedAt || item.createdAt,
        turnCount: item.turnCount || 1,
        questions: item.questions?.length ? item.questions : [item.question],
        turns: item.turns?.length ? item.turns.slice(-TRANSCRIPT_TURN_LIMIT) : [legacyTurn],
      }
      const existing = grouped.get(key)
      grouped.set(key, existing ? {
        ...existing,
        lastQuestion: normalized.lastQuestion,
        asOfDate: normalized.asOfDate,
        answer: normalized.answer,
        endToEndMs: normalized.endToEndMs,
        updatedAt: normalized.updatedAt,
        turnCount: existing.turnCount + normalized.turnCount,
        questions: [...existing.questions, ...normalized.questions],
        turns: [...(existing.turns || []), ...(normalized.turns || [])].slice(-TRANSCRIPT_TURN_LIMIT),
      } : normalized)
    }
    return [...grouped.values()].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).slice(0, HISTORY_LIMIT)
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
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory)
  const [selectedHistoryId, setSelectedHistoryId] = useState('')
  const [conversationId, setConversationId] = useState(() => localStorage.getItem(ACTIVE_CONVERSATION_KEY) || '')
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [pendingQuestion, setPendingQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [clearingConversation, setClearingConversation] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)) }, [history])
  useEffect(() => { if (conversationId) localStorage.setItem(ACTIVE_CONVERSATION_KEY, conversationId); else localStorage.removeItem(ACTIVE_CONVERSATION_KEY) }, [conversationId])
  useEffect(() => {
    if (!conversationId || turns.length) return
    const active = history.find(entry => entry.conversationId === conversationId)
    if (active?.turns?.length) { setTurns(active.turns); setSelectedHistoryId(active.id) }
  }, [conversationId, history, turns.length])

  const submit = async () => {
    const submittedQuestion = question.trim()
    if (clearingConversation || loading || submittedQuestion.length < 8) return
    setLoading(true); setError(''); setPendingQuestion(submittedQuestion); setQuestion('')
    try {
      const result = await ask(submittedQuestion, asOfDate, conversationId)
      const nextConversationId = result.answer.conversation_id || conversationId
      setConversationId(nextConversationId)
      const now = new Date().toISOString()
      const nextTurn: ChatTurn = { id: crypto.randomUUID(), question: submittedQuestion, asOfDate, answer: result.answer, endToEndMs: result.endToEndMs, createdAt: now }
      setTurns(current => [...current, nextTurn].slice(-TRANSCRIPT_TURN_LIMIT))
      const existingEntry = history.find(item => item.conversationId === nextConversationId)
      const entryId = existingEntry?.id || crypto.randomUUID()
      setHistory(items => {
        const existing = items.find(item => item.conversationId === nextConversationId)
        const entry: HistoryEntry = existing
          ? { ...existing, lastQuestion: submittedQuestion, asOfDate, answer: result.answer, endToEndMs: result.endToEndMs, updatedAt: now, turnCount: existing.turnCount + 1, questions: [...existing.questions, submittedQuestion], turns: [...(existing.turns || []), nextTurn].slice(-TRANSCRIPT_TURN_LIMIT) }
          : { id: entryId, question: submittedQuestion, lastQuestion: submittedQuestion, asOfDate, answer: result.answer, endToEndMs: result.endToEndMs, createdAt: now, updatedAt: now, turnCount: 1, questions: [submittedQuestion], turns: [nextTurn], conversationId: nextConversationId || undefined }
        return [entry, ...items.filter(item => item.id !== entry.id)].slice(0, HISTORY_LIMIT)
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
    setQuestion(''); setAsOfDate(''); setTurns([]); setPendingQuestion(''); setError(''); setSelectedHistoryId(''); setConversationId('')
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
  const AssistantTurn = ({ turn }: { turn: ChatTurn }) => {
    const citedIds = new Set(turn.answer.claims.flatMap(claim => claim.citation_ids))
    const citations = turn.answer.citations.filter(citation => citedIds.has(citation.id))
    return <div className={`message assistant-message ${turn.answer.status}`}>
      <span>{turn.answer.status === 'grounded' ? `Có căn cứ · áp dụng ${turn.answer.applied_as_of_date}` : 'Chưa đủ căn cứ'}</span>
      <p className="answer-text">{turn.answer.answer}</p>
      {citations.length > 0 && <details className="message-sources"><summary>Căn cứ pháp lý ({citations.length})</summary><div className="citations">{citations.map(citation => <CitationCard citation={citation} key={citation.id} />)}</div></details>}
      {turn.answer.warnings.map((warning, index) => <p className="warning" key={index}>{warning}</p>)}
      <p className="latency">Phản hồi trong {(turn.endToEndMs / 1000).toFixed(2)} giây</p>
      {turn.answer.latency && <details className="latency-breakdown"><summary>Chi tiết độ trễ</summary><ul><li>Contextualize: {turn.answer.latency.contextualization_ms.toFixed(0)} ms</li><li>HyDE + MultiQuery: {turn.answer.latency.query_augmentation_ms.toFixed(0)} ms</li><li>Retrieval: {turn.answer.latency.retrieval_ms.toFixed(0)} ms</li><li>Rerank: {turn.answer.latency.rerank_ms.toFixed(0)} ms</li><li>LLM: {turn.answer.latency.llm_ms.toFixed(0)} ms</li></ul></details>}
    </div>
  }

  return <div className="chat-shell">
    <aside className="chat-history" aria-label="Lịch sử câu hỏi">
      <div className="history-heading"><h2>Lịch sử hỏi đáp</h2><div className="history-actions"><button className="history-new" type="button" disabled={clearingConversation} onClick={startNewChat}>Đoạn chat mới</button>{history.length > 0 && <button className="history-clear" type="button" disabled={clearingConversation} onClick={() => void clearHistory()}>Xóa tất cả</button>}</div></div>
      {history.length === 0 ? <p className="history-empty">Chưa có đoạn chat nào.</p> : <ul>{history.map(entry => <li key={entry.id} className={entry.id === selectedHistoryId ? 'selected' : ''}><button className="history-open" type="button" disabled={clearingConversation} onClick={() => openHistory(entry)}><span>{entry.question}</span><small>{entry.answer.status === 'grounded' ? 'Có căn cứ' : 'Chưa đủ căn cứ'} · {entry.turnCount} lượt hỏi</small></button><button className="history-delete" type="button" disabled={clearingConversation} aria-label={`Xóa đoạn chat: ${entry.question}`} onClick={() => void deleteHistory(entry)}>×</button><details className="history-questions"><summary>Xem các câu hỏi</summary><ol>{entry.questions.map((askedQuestion, index) => <li key={`${entry.id}-${index}`}>{askedQuestion}</li>)}</ol></details></li>)}</ul>}
      <p className="history-note">Xóa một mục sẽ xóa ngữ cảnh đoạn chat đó.</p>
    </aside>
    <main className="chat-page">
      <nav><span className="brand">ChatBot Luật RAG</span><a href="#admin">Quản trị kho luật</a></nav>
      <section className="chat-transcript" aria-live="polite">
        {turns.length === 0 && !pendingQuestion && <div className="chat-welcome"><h1>Xin chào</h1></div>}
        {turns.map(turn => <div className="turn" key={turn.id}><div className="message user-message"><span>Bạn</span><p>{turn.question}</p></div><AssistantTurn turn={turn} /></div>)}
        {pendingQuestion && <div className="turn"><div className="message user-message"><span>Bạn</span><p>{pendingQuestion}</p></div><div className="message assistant-message pending-message"><span>ChatBot Luật</span><p>Đang đối chiếu nguồn pháp lý…</p></div></div>}
      </section>
      {error && <p className="error">{error}</p>}
      <form className="ask composer" onSubmit={event => { event.preventDefault(); void submit() }}>
        <textarea aria-label="Câu hỏi pháp lý" value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void submit() } }} minLength={8} required placeholder="Nhập câu hỏi pháp lý của bạn…" />
        <div className="composer-actions"><small>Enter để gửi · Shift + Enter để xuống dòng</small><input aria-label="Thời điểm áp dụng" title="Để trống để dùng quy định hiện hành" type="date" value={asOfDate} onChange={e => setAsOfDate(e.target.value)} /><button disabled={loading || clearingConversation || question.trim().length < 8}>{loading ? 'Đang trả lời…' : 'Gửi'}</button></div>
      </form>
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
  const [chunkPages, setChunkPages] = useState<Record<string, ProvisionPage>>({})
  const [reviewed, setReviewed] = useState<Record<string, boolean>>({})
  const [publishingId, setPublishingId] = useState('')
  const [loadingChunksId, setLoadingChunksId] = useState('')
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
  const togglePublishedChunks = async (item: Version, offset = 0) => {
    if (offset === 0 && chunkPages[item.id]) {
      setChunkPages(current => {
        const next = { ...current }
        delete next[item.id]
        return next
      })
      return
    }
    setLoadingChunksId(item.id)
    try {
      const page = await publishedChunks(token, item.id, offset)
      setChunkPages(current => ({ ...current, [item.id]: page }))
    } catch (e) { setNotice(e instanceof Error ? e.message : 'Không thể tải chunk') }
    finally { setLoadingChunksId('') }
  }
  if (!token) return <main className="admin-page"><a href="#/">← Tra cứu</a><h1>Quản trị kho luật</h1><form className="panel" onSubmit={auth}><input value={email} onChange={e => setEmail(e.target.value)} placeholder="Email admin" type="email" required /><input value={password} onChange={e => setPassword(e.target.value)} placeholder="Mật khẩu" type="password" required /><button>Đăng nhập</button></form>{notice && <p className="error">{notice}</p>}</main>
  return <main className="admin-page">
    <nav><a href="#/">← Tra cứu</a><button className="link-button" onClick={() => { sessionStorage.removeItem('lawrag_admin_token'); setToken('') }}>Đăng xuất</button></nav>
    <h1>Kho văn bản pháp luật</h1><p className="muted">Chỉ xuất bản sau khi kiểm tra cấu trúc Điều/Khoản/Điểm và URL nguồn chính thức.</p>
    <form className="panel upload" onSubmit={submit}>{Object.entries(fields).map(([key, value]) => <label key={key}>{key === 'document_code' ? 'Mã văn bản' : key === 'title' ? 'Tên văn bản' : key === 'version_label' ? 'Nhãn phiên bản' : key === 'official_url' ? 'URL chính thức' : key === 'effective_from' ? 'Hiệu lực từ' : 'Hiệu lực đến'}<input required={key !== 'effective_to'} type={key.includes('effective') ? 'date' : key === 'official_url' ? 'url' : 'text'} value={value} onChange={e => setFields({ ...fields, [key]: e.target.value })} /></label>)}<label>Tệp PDF/DOCX<input required type="file" accept=".pdf,.docx" onChange={e => setFile(e.target.files?.[0] || null)} /></label><button>Đưa vào hàng đợi xử lý</button></form>
    {activeJob && <p className="notice">Xử lý: {activeJob.status} · {activeJob.progress}% {activeJob.message || ''}</p>}{notice && <p className="notice">{notice}</p>}<button type="button" onClick={refresh}>Làm mới</button>
    <section className="version-list">{items.map(item => {
      const review = reviews[item.id]
      const chunkPage = chunkPages[item.id]
      return <article className="version" key={item.id}><div><strong>{item.document_code} — {item.version_label}</strong><p>{item.document_title}</p><small>Hiệu lực: {item.effective_from} {item.effective_to ? `đến ${item.effective_to}` : 'đến nay'}</small>
        {review && <details className="review" open><summary>Đã tải {review.provision_count} đơn vị để kiểm tra</summary>{review.provisions.map(p => <div className="provision" key={p.id}><strong>Điều {p.article_no}{p.clause_no ? ` · Khoản ${p.clause_no}` : ''}{p.point_label ? ` · Điểm ${p.point_label}` : ''}</strong>{p.heading && <span> — {p.heading}</span>}<p>{p.content}</p></div>)}<label className="review-check"><input type="checkbox" checked={Boolean(reviewed[item.id])} onChange={e => setReviewed({ ...reviewed, [item.id]: e.target.checked })} /> Tôi đã kiểm tra cấu trúc và đoạn trích phía trên.</label></details>}
        {chunkPage && <section className="published-chunks"><h3>Chunk đã lập chỉ mục ({chunkPage.total})</h3><p className="muted">Đang xem {chunkPage.total === 0 ? 0 : chunkPage.offset + 1}–{Math.min(chunkPage.offset + chunkPage.provisions.length, chunkPage.total)}. Mỗi mục là một đơn vị Điều/Khoản/Điểm đã gửi vào Qdrant.</p>{chunkPage.provisions.map(p => <div className="provision" key={p.id}><strong>#{p.ordinal} · Điều {p.article_no}{p.clause_no ? ` · Khoản ${p.clause_no}` : ''}{p.point_label ? ` · Điểm ${p.point_label}` : ''}</strong>{p.heading && <span> — {p.heading}</span>}<p>{p.content}</p></div>)}<div className="chunk-pagination"><button type="button" disabled={loadingChunksId === item.id || chunkPage.offset === 0} onClick={() => void togglePublishedChunks(item, Math.max(0, chunkPage.offset - chunkPage.limit))}>← Trước</button><button type="button" disabled={loadingChunksId === item.id || chunkPage.offset + chunkPage.provisions.length >= chunkPage.total} onClick={() => void togglePublishedChunks(item, chunkPage.offset + chunkPage.limit)}>Sau →</button></div></section>}
      </div><div><span className={`badge ${item.status.toLowerCase()}`}>{item.status}</span>{item.status === 'PENDING_REVIEW' && <><button type="button" onClick={() => void inspect(item)}>Kiểm tra cấu trúc</button>{review && <button type="button" disabled={!reviewed[item.id] || publishingId === item.id} onClick={() => void publishReviewed(item)}>{publishingId === item.id ? 'Đang xuất bản…' : 'Duyệt & xuất bản'}</button>}</>}{item.status === 'PUBLISHED' && <><button type="button" disabled={loadingChunksId === item.id} onClick={() => void togglePublishedChunks(item)}>{loadingChunksId === item.id ? 'Đang tải chunk…' : chunkPage ? 'Ẩn chunk' : 'Xem chunk đã lập chỉ mục'}</button><button type="button" onClick={() => void withdrawPublished(item)}>Thu hồi</button></>}</div></article>
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
