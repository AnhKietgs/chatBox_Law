export type Citation = { id: string; document_code: string; document_title: string; version_label: string; article: string; clause?: string; point?: string; excerpt: string; official_url: string; relevance_score?: number }
export type Claim = { text: string; citation_ids: string[] }
export type Latency = { contextualization_ms: number; query_augmentation_ms: number; retrieval_ms: number; rerank_ms: number; llm_ms: number; citation_validation_ms: number; total_ms: number }
export type Answer = { status: 'grounded' | 'abstained'; answer: string; claims: Claim[]; citations: Citation[]; warnings: string[]; applied_as_of_date: string; latency?: Latency; conversation_id?: string }
export type Version = { id: string; document_code: string; document_title: string; version_label: string; effective_from: string; effective_to?: string; status: string; official_url: string; domain: string; category: string }
export type Job = { id: string; version_id: string; status: string; progress: number; message?: string }
export type Provision = { id: string; article_no: string; clause_no?: string; point_label?: string; heading?: string; content: string; ordinal: number; chunk_index: number }
export type ProvisionPage = { version_id: string; total: number; offset: number; limit: number; provisions: Provision[] }
export type Review = { version_id: string; provision_count: number; structure_hash: string; provisions: Provision[] }

export type ChatTurn = {
  id: string
  question: string
  asOfDate: string
  answer: Answer
  endToEndMs: number
  createdAt: string
}

export type HistoryEntry = {
  id: string
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

export type Notice = { tone: 'info' | 'success' | 'error'; message: string }
