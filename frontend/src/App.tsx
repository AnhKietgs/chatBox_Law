import { FormEvent, useEffect, useState } from 'react'
import { ask, job, login, publish, reviewVersion, uploadVersion, versions, withdraw } from './api'
import type { Answer, Citation, Job, Review, Version } from './types'

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
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const submit = async () => {
    setLoading(true); setError(''); setAnswer(null); setEndToEndMs(null)
    try {
      const result = await ask(question, asOfDate)
      setAnswer(result.answer)
      setEndToEndMs(result.endToEndMs)
      setQuestion('')
    } catch (err) { setError(err instanceof Error ? err.message : 'Đã có lỗi xảy ra') } finally { setLoading(false) }
  }
  return <main className="chat-page">
    <nav><span className="brand">Luật Thương mại RAG</span><a href="#admin">Quản trị kho luật</a></nav>
    <section className="hero"><p className="eyebrow">Tra cứu có căn cứ</p><h1>Chatbot hỏi đáp về hợp đồng thương mại, mua bán hàng hóa và phạt vi phạm.</h1></section>
    <form className="ask" onSubmit={event => { event.preventDefault(); void submit() }}>
      <label>Câu hỏi pháp lý<textarea value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (!loading && question.trim().length >= 8) void submit() } }} minLength={8} required placeholder="Ví dụ: Mức phạt vi phạm hợp đồng mua bán hàng hóa tối đa là bao nhiêu?" /></label>
      <small>Enter để hỏi · Shift + Enter để xuống dòng</small>
      <label>Thời điểm áp dụng <small>Để trống để dùng quy định hiện hành</small><input type="date" value={asOfDate} onChange={e => setAsOfDate(e.target.value)} /></label>
      <button disabled={loading}>{loading ? 'Đang đối chiếu nguồn…' : 'Hỏi pháp luật'}</button>
    </form>
    {error && <p className="error">{error}</p>}
    {answer && <section className={`answer ${answer.status}`}>
      <div className="answer-state">{answer.status === 'grounded' ? `Có căn cứ · áp dụng ${answer.applied_as_of_date}` : 'Chưa đủ căn cứ'}</div>
      {endToEndMs !== null && <p className="latency">Phản hồi trong {(endToEndMs / 1000).toFixed(2)} giây · Server xử lý {((answer.latency?.total_ms ?? 0) / 1000).toFixed(2)} giây</p>}
      <p className="answer-text">{answer.answer}</p>
      {answer.claims.map((claim, i) => <div className="claim" key={i}><span>{claim.text}</span><div>{claim.citation_ids.map(id => { const c = answer.citations.find(item => item.id === id); return c && <a className="chip" href={`#cite-${id}`} key={id}><Locator citation={c} /></a> })}</div></div>)}
      {answer.citations.length > 0 && <><h2>Căn cứ pháp lý</h2><div className="citations">{answer.citations.map(c => <div id={`cite-${c.id}`} key={c.id}><CitationCard citation={c} /></div>)}</div></>}
      {answer.warnings.map((warning, i) => <p className="warning" key={i}>{warning}</p>)}
      {answer.latency && <details className="latency-breakdown"><summary>Chi tiết độ trễ máy chủ</summary><ul><li>Retrieval: {answer.latency.retrieval_ms.toFixed(0)} ms</li><li>Rerank: {answer.latency.rerank_ms.toFixed(0)} ms</li><li>LLM: {answer.latency.llm_ms.toFixed(0)} ms</li><li>Kiểm tra citation: {answer.latency.citation_validation_ms.toFixed(0)} ms</li></ul></details>}
    </section>}
    <footer>Thông tin tra cứu mang tính tham khảo, không thay thế tư vấn luật sư cho tình huống cụ thể.</footer>
  </main>
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
