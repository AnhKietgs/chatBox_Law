import type { FormEvent } from 'react'

export type UploadFields = {
  document_code: string; title: string; version_label: string; official_url: string
  effective_from: string; effective_to: string; domain: string; category: string
}

type Props = {
  fields: UploadFields
  file: File | null
  submitting: boolean
  fieldError?: string
  onFieldsChange: (next: UploadFields) => void
  onFileChange: (file: File | null) => void
  onSubmit: (event: FormEvent) => void
}

const domains = [
  ['commercial', 'Thương mại'], ['civil', 'Dân sự'], ['labor', 'Lao động'], ['enterprise', 'Doanh nghiệp'],
  ['tax', 'Thuế'], ['family', 'Hôn nhân & gia đình'], ['criminal', 'Hình sự'], ['general', 'Khác/chưa phân loại'],
]

export function UploadForm({ fields, file, submitting, fieldError, onFieldsChange, onFileChange, onSubmit }: Props) {
  const update = (key: keyof UploadFields, value: string) => onFieldsChange({ ...fields, [key]: value })
  return <form className="panel upload" onSubmit={onSubmit} noValidate>
    <label htmlFor="document-code">Mã văn bản <span aria-hidden="true">*</span><input id="document-code" value={fields.document_code} onChange={e => update('document_code', e.target.value)} placeholder="Ví dụ: LTM-2005" required /></label>
    <label htmlFor="document-title">Tên văn bản <span aria-hidden="true">*</span><input id="document-title" value={fields.title} onChange={e => update('title', e.target.value)} placeholder="Ví dụ: Luật Thương mại 2005" required /></label>
    <label htmlFor="version-label">Nhãn phiên bản <span aria-hidden="true">*</span><input id="version-label" value={fields.version_label} onChange={e => update('version_label', e.target.value)} placeholder="Ví dụ: Bản gốc" required /></label>
    <label htmlFor="official-url">URL nguồn chính thức <span aria-hidden="true">*</span><input id="official-url" type="url" value={fields.official_url} onChange={e => update('official_url', e.target.value)} placeholder="https://…" required /></label>
    <label htmlFor="effective-from">Hiệu lực từ <span aria-hidden="true">*</span><input id="effective-from" type="date" value={fields.effective_from} onChange={e => update('effective_from', e.target.value)} required /></label>
    <label htmlFor="effective-to">Hiệu lực đến <small>Để trống nếu còn hiệu lực</small><input id="effective-to" type="date" value={fields.effective_to} onChange={e => update('effective_to', e.target.value)} /></label>
    <label htmlFor="document-domain">Lĩnh vực pháp luật <select id="document-domain" value={fields.domain} onChange={e => update('domain', e.target.value)}>{domains.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label>
    <label htmlFor="document-category">Nhóm nghiệp vụ <input id="document-category" value={fields.category} onChange={e => update('category', e.target.value)} placeholder="Ví dụ: hợp đồng" /></label>
    <label className="upload-file" htmlFor="source-file">Tệp PDF/DOCX <span aria-hidden="true">*</span><input id="source-file" required type="file" accept=".pdf,.docx" onChange={e => onFileChange(e.target.files?.[0] || null)} aria-describedby="source-file-help" /><small id="source-file-help">Tối đa 50 MB. Hệ thống chỉ chấp nhận PDF/DOCX.</small>{file && <span className="selected-file">Đã chọn: {file.name}</span>}</label>
    {fieldError && <p className="form-error" role="alert">{fieldError}</p>}
    <button disabled={submitting}>{submitting ? 'Đang gửi tệp…' : 'Đưa vào hàng đợi xử lý'}</button>
  </form>
}
