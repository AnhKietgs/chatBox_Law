import type { ChatTurn } from '../../types'
import { CitationCard } from './CitationCard'

export function AssistantTurn({ turn }: { turn: ChatTurn }) {
  const citedIds = new Set(turn.answer.claims.flatMap(claim => claim.citation_ids))
  const citations = turn.answer.citations.filter(citation => citedIds.has(citation.id))
  const isWebFallback = turn.answer.source_mode === 'web_search'
  return <div className={`message assistant-message ${turn.answer.status}`}>
    <span>{turn.answer.status === 'grounded' ? `Có căn cứ · áp dụng ${turn.answer.applied_as_of_date}` : isWebFallback ? 'Chưa đủ căn cứ pháp lý · thông tin web tham khảo' : 'Chưa đủ căn cứ'}</span>
    <p className="answer-text">{turn.answer.answer}</p>
    {citations.length > 0 && <details className="message-sources"><summary>Căn cứ pháp lý ({citations.length})</summary><div className="citations">{citations.map(citation => <CitationCard citation={citation} key={citation.id} />)}</div></details>}
    {isWebFallback && turn.answer.web_sources.length > 0 && <details className="message-sources web-sources"><summary>Nguồn web chưa xác minh ({turn.answer.web_sources.length})</summary><div className="citations">{turn.answer.web_sources.map(source => <article className="citation" key={source.id}><strong>{source.title}</strong><p>{source.snippet}</p><a href={source.url} target="_blank" rel="noreferrer">Mở nguồn web</a></article>)}</div></details>}
    {turn.answer.warnings.map((warning, index) => <p className="warning" key={index}>{warning}</p>)}
    <p className="latency">Phản hồi trong {(turn.endToEndMs / 1000).toFixed(2)} giây</p>
    {turn.answer.latency && <details className="latency-breakdown"><summary>Chi tiết độ trễ</summary><ul><li>Contextualize: {turn.answer.latency.contextualization_ms.toFixed(0)} ms</li><li>HyDE + MultiQuery: {turn.answer.latency.query_augmentation_ms.toFixed(0)} ms</li><li>Retrieval: {turn.answer.latency.retrieval_ms.toFixed(0)} ms</li><li>Rerank: {turn.answer.latency.rerank_ms.toFixed(0)} ms</li>{turn.answer.latency.web_search_ms > 0 && <li>Web search: {turn.answer.latency.web_search_ms.toFixed(0)} ms</li>}<li>LLM: {turn.answer.latency.llm_ms.toFixed(0)} ms</li></ul></details>}
  </div>
}
