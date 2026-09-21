import type { Answer, Job, Review, Version } from './types'
const jsonHeaders = { 'Content-Type': 'application/json' }

type ValidationIssue = { loc?: Array<string | number>; msg?: string }
const fieldNames: Record<string, string> = {
  document_code: 'Mã văn bản', title: 'Tên văn bản', version_label: 'Nhãn phiên bản',
  official_url: 'URL chính thức', effective_from: 'Hiệu lực từ', effective_to: 'Hiệu lực đến', file: 'Tệp PDF/DOCX',
}

async function responseError(response: Response, fallback: string): Promise<Error> {
  try {
    const payload = await response.json()
    if (typeof payload.detail === 'string') return new Error(payload.detail)
    if (Array.isArray(payload.detail)) {
      const message = payload.detail.map((issue: ValidationIssue) => {
        const key = String(issue.loc?.at(-1) ?? '')
        const displayName = fieldNames[key] ?? (key || 'Dữ liệu')
        return `${displayName}: ${issue.msg ?? 'không hợp lệ'}`
      }).join('; ')
      return new Error(message || fallback)
    }
  } catch { /* A non-JSON error response uses the fallback below. */ }
  return new Error(fallback)
}

export type AskedAnswer = { answer: Answer; endToEndMs: number }

export async function ask(question: string, asOfDate?: string): Promise<AskedAnswer> {
  const started = performance.now()
  const response = await fetch('/api/v1/chat/query', { method: 'POST', headers: jsonHeaders, body: JSON.stringify({ question, as_of_date: asOfDate || null }) })
  if (!response.ok) throw await responseError(response, 'Không thể gửi câu hỏi')
  return { answer: await response.json(), endToEndMs: performance.now() - started }
}
export async function login(email: string, password: string): Promise<string> {
  const response = await fetch('/api/v1/admin/token', { method: 'POST', headers: jsonHeaders, body: JSON.stringify({ email, password }) })
  if (!response.ok) throw new Error('Đăng nhập không thành công')
  return (await response.json()).access_token
}
export async function versions(token: string): Promise<Version[]> {
  const response = await fetch('/api/v1/admin/versions', { headers: { Authorization: `Bearer ${token}` } })
  if (!response.ok) throw new Error('Không thể tải danh sách văn bản')
  return response.json()
}
export async function uploadVersion(token: string, fields: Record<string, string>, file: File): Promise<Job> {
  const data = new FormData(); Object.entries(fields).forEach(([key, value]) => data.append(key, value)); data.append('file', file)
  const response = await fetch('/api/v1/admin/versions', { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: data })
  if (!response.ok) throw await responseError(response, 'Không thể tạo phiên bản')
  return response.json()
}
export async function job(token: string, id: string): Promise<Job> {
  const response = await fetch(`/api/v1/admin/jobs/${id}`, { headers: { Authorization: `Bearer ${token}` } })
  if (!response.ok) throw new Error('Không thể xem trạng thái xử lý')
  return response.json()
}
export async function reviewVersion(token: string, id: string): Promise<Review> {
  const response = await fetch(`/api/v1/admin/versions/${id}/review`, { headers: { Authorization: `Bearer ${token}` } })
  if (!response.ok) throw await responseError(response, 'Không thể tải cấu trúc văn bản')
  return response.json()
}
export async function publish(token: string, id: string, reviewedStructureHash: string) {
  const response = await fetch(`/api/v1/admin/versions/${id}/publish`, { method: 'POST', headers: { ...jsonHeaders, Authorization: `Bearer ${token}` }, body: JSON.stringify({ reviewed_structure_hash: reviewedStructureHash }) })
  if (!response.ok) throw await responseError(response, 'Không thể xuất bản')
}
export async function withdraw(token: string, id: string) {
  const response = await fetch(`/api/v1/admin/versions/${id}/withdraw`, { method: 'POST', headers: { Authorization: `Bearer ${token}` } })
  if (!response.ok) throw await responseError(response, 'Không thể thu hồi phiên bản')
}
