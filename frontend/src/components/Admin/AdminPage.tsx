import { type FormEvent, useEffect, useState } from 'react'
import { job, login, publish, publishedChunks, reindex, reviewVersion, uploadVersion, versions, withdraw } from '../../api'
import type { Job, Notice, ProvisionPage, Review, Version } from '../../types'
import { UploadForm, type UploadFields } from './UploadForm'
import { VersionList } from './VersionList'

const emptyFields: UploadFields = { document_code: '', title: '', version_label: '', official_url: '', effective_from: '', effective_to: '', domain: 'commercial', category: 'general' }

export function AdminPage() {
  const [token, setToken] = useState(sessionStorage.getItem('lawrag_admin_token') || '')
  const [email, setEmail] = useState(''); const [password, setPassword] = useState('')
  const [authenticating, setAuthenticating] = useState(false)
  const [items, setItems] = useState<Version[]>([]); const [notice, setNotice] = useState<Notice | null>(null)
  const [file, setFile] = useState<File | null>(null); const [activeJob, setActiveJob] = useState<Job | null>(null)
  const [reviews, setReviews] = useState<Record<string, Review>>({}); const [chunkPages, setChunkPages] = useState<Record<string, ProvisionPage>>({})
  const [reviewed, setReviewed] = useState<Record<string, boolean>>({}); const [publishingId, setPublishingId] = useState(''); const [reindexingId, setReindexingId] = useState(''); const [loadingChunksId, setLoadingChunksId] = useState('')
  const [fields, setFields] = useState<UploadFields>(emptyFields); const [submitting, setSubmitting] = useState(false)
  const reportError = (error: unknown, fallback = 'Đã có lỗi xảy ra') => setNotice({ tone: 'error', message: error instanceof Error ? error.message : fallback })
  const refresh = () => { if (token) void versions(token).then(setItems).catch(error => reportError(error, 'Không thể tải danh sách văn bản')) }
  useEffect(refresh, [token])
  useEffect(() => {
    if (!activeJob || !['QUEUED', 'RUNNING'].includes(activeJob.status)) return
    const timer = window.setInterval(async () => { try { const next = await job(token, activeJob.id); setActiveJob(next); if (!['QUEUED', 'RUNNING'].includes(next.status)) { setReindexingId(''); setNotice({ tone: next.status === 'SUCCEEDED' ? 'success' : 'error', message: next.message || (next.status === 'SUCCEEDED' ? 'Tác vụ đã hoàn tất.' : 'Tác vụ xử lý thất bại.') }); refresh() } } catch { setNotice({ tone: 'error', message: 'Không thể cập nhật tiến trình xử lý.' }) } }, 2000)
    return () => clearInterval(timer)
  }, [activeJob, token])
  const auth = async (event: FormEvent) => {
    event.preventDefault()
    setAuthenticating(true)
    setNotice(null)
    try {
      const value = await login(email, password)
      sessionStorage.setItem('lawrag_admin_token', value)
      setToken(value)
      setNotice({ tone: 'success', message: 'Đăng nhập quản trị thành công.' })
    } catch (error) {
      reportError(error, 'Đăng nhập không thành công')
    } finally {
      setAuthenticating(false)
    }
  }
  const submit = async (event: FormEvent) => {
    event.preventDefault(); if (!file) { setNotice({ tone: 'error', message: 'Chọn tệp PDF hoặc DOCX trước khi gửi.' }); return }
    setSubmitting(true); setNotice({ tone: 'info', message: 'Đang tải tệp và tạo job xử lý…' })
    try { const created = await uploadVersion(token, fields, file); setActiveJob(created); setFile(null); setNotice({ tone: 'success', message: `Đã xếp hàng xử lý ${created.id}.` }); refresh() } catch (error) { reportError(error, 'Không thể tạo phiên bản') } finally { setSubmitting(false) }
  }
  const inspect = async (item: Version) => { try { const value = await reviewVersion(token, item.id); setReviews(current => ({ ...current, [item.id]: value })); setReviewed(current => ({ ...current, [item.id]: false })); setNotice({ tone: 'success', message: `Đã tải ${value.provision_count} đơn vị để kiểm tra.` }) } catch (error) { reportError(error, 'Không thể tải cấu trúc văn bản') } }
  const publishReviewed = async (item: Version) => { const review = reviews[item.id]; if (!review || !reviewed[item.id]) { setNotice({ tone: 'error', message: 'Hãy mở cấu trúc và tick xác nhận trước khi xuất bản.' }); return }; setPublishingId(item.id); setNotice({ tone: 'info', message: 'Đang gửi yêu cầu xuất bản và lập chỉ mục…' }); try { await publish(token, item.id, review.structure_hash); setNotice({ tone: 'success', message: 'Đã xuất bản phiên bản đã duyệt.' }); refresh() } catch (error) { reportError(error, 'Không thể xuất bản') } finally { setPublishingId('') } }
  const withdrawPublished = async (item: Version) => { if (!window.confirm('Thu hồi phiên bản này khỏi chatbot để kiểm tra lại?')) return; try { await withdraw(token, item.id); setNotice({ tone: 'success', message: 'Đã thu hồi phiên bản; chatbot sẽ không còn dùng dữ liệu này.' }); refresh() } catch (error) { reportError(error, 'Không thể thu hồi') } }
  const reindexPublished = async (item: Version) => { setReindexingId(item.id); setNotice({ tone: 'info', message: `Đang gửi yêu cầu lập chỉ mục lại ${item.domain}/${item.category}…` }); try { const created = await reindex(token, item.id); setActiveJob(created); setNotice({ tone: 'info', message: 'Đã xếp hàng lập chỉ mục lại. Bạn có thể theo dõi tiến trình trên trang này.' }) } catch (error) { setReindexingId(''); reportError(error, 'Không thể lập chỉ mục lại') } }
  const togglePublishedChunks = async (item: Version, offset = 0) => { if (offset === 0 && chunkPages[item.id]) { setChunkPages(current => { const next = { ...current }; delete next[item.id]; return next }); return }; setLoadingChunksId(item.id); try { const page = await publishedChunks(token, item.id, offset); setChunkPages(current => ({ ...current, [item.id]: page })) } catch (error) { reportError(error, 'Không thể tải chunk') } finally { setLoadingChunksId('') } }
  if (!token) return <main className="admin-page admin-login-page">
    <nav className="admin-login-nav" aria-label="Điều hướng quản trị">
      <a className="admin-back-link" href="#/"><span aria-hidden="true">←</span> Tra cứu pháp luật</a>
    </nav>
    <section className="login-shell" aria-labelledby="admin-login-title">
      <header className="login-intro">
        <p className="eyebrow">KHO VĂN BẢN PHÁP LUẬT</p>
        <h1 id="admin-login-title">Quản trị kho luật</h1>
        <p>Đăng nhập để tải lên, kiểm tra cấu trúc và xuất bản các văn bản được chatbot sử dụng.</p>
      </header>
      <div className="login-panel-wrap">
        <form className="panel login-form" onSubmit={auth} aria-describedby={notice ? 'admin-login-notice' : undefined}>
          <div className="login-form-heading">
            <h2>Đăng nhập quản trị</h2>
            <p>Sử dụng tài khoản quản trị đã cấu hình trong hệ thống.</p>
          </div>
          <label htmlFor="admin-email">Email quản trị
            <input id="admin-email" value={email} onChange={e => setEmail(e.target.value)} placeholder="admin@example.com" type="email" autoComplete="username" inputMode="email" aria-invalid={notice?.tone === 'error'} required autoFocus />
          </label>
          <label htmlFor="admin-password">Mật khẩu
            <input id="admin-password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Nhập mật khẩu" type="password" autoComplete="current-password" aria-invalid={notice?.tone === 'error'} required />
          </label>
          <button type="submit" disabled={authenticating} aria-busy={authenticating}>{authenticating ? 'Đang đăng nhập…' : 'Đăng nhập'}</button>
        </form>
        {notice && <p id="admin-login-notice" className={`notice ${notice.tone}`} role={notice.tone === 'error' ? 'alert' : 'status'}>{notice.message}</p>}
      </div>
    </section>
  </main>
  return <main className="admin-page"><p className="sr-only" aria-live="polite" aria-atomic="true">{notice?.message || (activeJob ? `Xử lý ${activeJob.status}, ${activeJob.progress} phần trăm.` : '')}</p><nav><a href="#/">← Tra cứu</a><button className="link-button" onClick={() => { sessionStorage.removeItem('lawrag_admin_token'); setToken('') }}>Đăng xuất</button></nav><h1>Kho văn bản pháp luật</h1><p className="muted">Chỉ xuất bản sau khi kiểm tra cấu trúc Điều/Khoản/Điểm và URL nguồn chính thức.</p>
    <UploadForm fields={fields} file={file} submitting={submitting} fieldError={notice?.tone === 'error' ? notice.message : undefined} onFieldsChange={setFields} onFileChange={setFile} onSubmit={submit} />
    {activeJob && <p className="notice info" role="status">Xử lý: {activeJob.status} · {activeJob.progress}% {activeJob.message || ''}</p>}{notice && <p className={`notice ${notice.tone}`} role={notice.tone === 'error' ? 'alert' : 'status'}>{notice.message}</p>}<button type="button" onClick={refresh}>Làm mới</button>
    <VersionList items={items} reviews={reviews} chunkPages={chunkPages} reviewed={reviewed} publishingId={publishingId} reindexingId={reindexingId} loadingChunksId={loadingChunksId} onInspect={item => void inspect(item)} onReviewedChange={(id, checked) => setReviewed(current => ({ ...current, [id]: checked }))} onPublish={item => void publishReviewed(item)} onReindex={item => void reindexPublished(item)} onChunks={(item, offset) => void togglePublishedChunks(item, offset)} onWithdraw={item => void withdrawPublished(item)} />
  </main>
}
