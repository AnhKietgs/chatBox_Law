from datetime import date, datetime
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field, HttpUrl


class TokenRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class ChatQuery(BaseModel):
    question: str = Field(min_length=8, max_length=4000)
    as_of_date: date | None = None
    conversation_id: UUID | None = None


class Citation(BaseModel):
    id: UUID
    document_code: str
    document_title: str
    version_label: str
    article: str
    clause: str | None = None
    point: str | None = None
    excerpt: str
    official_url: HttpUrl
    relevance_score: float | None = None


class Claim(BaseModel):
    text: str
    citation_ids: list[UUID] = Field(min_length=1)


class LatencyBreakdown(BaseModel):
    """Server-side timings in milliseconds for one chat request."""
    retrieval_ms: float = 0
    rerank_ms: float = 0
    llm_ms: float = 0
    citation_validation_ms: float = 0
    total_ms: float = 0


class ChatResponse(BaseModel):
    status: Literal["grounded", "abstained"]
    answer: str
    claims: list[Claim] = []
    citations: list[Citation] = []
    warnings: list[str] = []
    applied_as_of_date: date
    latency: LatencyBreakdown | None = None
    conversation_id: UUID | None = None


class VersionSummary(BaseModel):
    id: UUID
    document_code: str
    document_title: str
    version_label: str
    effective_from: date
    effective_to: date | None
    status: str
    official_url: str


class ProvisionSummary(BaseModel):
    id: UUID
    article_no: str
    clause_no: str | None
    point_label: str | None
    heading: str | None
    content: str
    ordinal: int


class ReviewSummary(BaseModel):
    version_id: UUID
    provision_count: int
    structure_hash: str
    provisions: list[ProvisionSummary]


class PublishRequest(BaseModel):
    reviewed_structure_hash: str = Field(min_length=64, max_length=64)


class JobSummary(BaseModel):
    id: UUID
    version_id: UUID
    status: str
    progress: int
    message: str | None
    created_at: datetime
