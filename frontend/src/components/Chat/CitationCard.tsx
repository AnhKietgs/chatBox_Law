import type { Citation } from '../../types'

export function Locator({ citation }: { citation: Citation }) {
  return <span>Điều {citation.article}{citation.clause ? ` · Khoản ${citation.clause}` : ''}{citation.point ? ` · Điểm ${citation.point}` : ''}</span>
}

export function CitationCard({ citation }: { citation: Citation }) {
  return <article className="citation">
    <div className="citation-top"><strong><Locator citation={citation} /></strong><span>{citation.document_code}</span></div>
    <p>{citation.excerpt}</p>
    <a href={citation.official_url} target="_blank" rel="noreferrer">Mở văn bản gốc — {citation.document_title}</a>
  </article>
}
