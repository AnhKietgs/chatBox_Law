from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
from time import perf_counter
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import LegalProvision, LegalVersion, LegalDocument, VersionStatus
from ..config import get_settings
from ..versioning import is_effective
from .vector_store import HybridVectorStore


@dataclass(frozen=True)
class RetrievedProvision:
    provision: LegalProvision
    version: LegalVersion
    document: LegalDocument
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    provisions: list[RetrievedProvision]
    retrieval_ms: float
    rerank_ms: float


class LegalRetriever:
    def __init__(self, db: Session):
        self.db = db

    def retrieve(self, question: str, as_of: date, limit: int = 8) -> list[RetrievedProvision]:
        return self.retrieve_with_metrics(question, as_of, limit).provisions

    def retrieve_with_metrics(self, question: str, as_of: date, limit: int = 8) -> RetrievalResult:
        retrieval_started = perf_counter()
        points = HybridVectorStore().search(question, as_of.isoformat(), limit=limit * 2)
        ids = [UUID(str(point.id)) for point in points]
        scores = {UUID(str(point.id)): float(point.score) for point in points}
        if not ids:
            return RetrievalResult([], (perf_counter() - retrieval_started) * 1000, 0)
        rows = self.db.execute(
            select(LegalProvision, LegalVersion, LegalDocument)
            .join(LegalVersion, LegalProvision.version_id == LegalVersion.id)
            .join(LegalDocument, LegalVersion.document_id == LegalDocument.id)
            .where(LegalProvision.id.in_(ids), LegalVersion.status == VersionStatus.published)
        ).all()
        valid = [
            RetrievedProvision(provision, version, document, scores[provision.id])
            for provision, version, document in rows
            if is_effective(version.effective_from, version.effective_to, as_of)
        ]
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        # bge-reranker-v2-m3 is deliberately applied after hybrid recall.
        rerank_started = perf_counter()
        provisions = Reranker().rerank(question, valid)[:limit]
        return RetrievalResult(provisions, retrieval_ms, (perf_counter() - rerank_started) * 1000)


class Reranker:
    def rerank(self, question: str, candidates: list[RetrievedProvision]) -> list[RetrievedProvision]:
        if not candidates:
            return []
        passages = [
            f"Điều {item.provision.article_no}. {item.provision.heading or ''}\n{item.provision.content}"
            for item in candidates
        ]
        # FlagEmbedding returns raw logits by default. Normalizing makes the
        # configured threshold interpretable on a stable 0..1 scale.
        scores = get_reranker_model().compute_score(
            [[question, passage] for passage in passages], normalize=True
        )
        if isinstance(scores, float):
            scores = [scores]
        threshold = get_settings().confidence_threshold
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [replace(item, score=float(score)) for score, item in ranked if float(score) >= threshold]


@lru_cache(maxsize=1)
def get_reranker_model():
    """Load bge-reranker-v2-m3 once per API process."""
    import torch
    from FlagEmbedding import FlagReranker
    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=torch.cuda.is_available())
