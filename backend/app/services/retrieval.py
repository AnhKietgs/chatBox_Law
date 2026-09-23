from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
import logging
from time import perf_counter
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..models import LegalProvision, LegalVersion, LegalDocument, VersionStatus
from ..config import get_settings
from ..versioning import is_effective
from .query_expansion import expand_commercial_query
from .conversation import contextual_retrieval_query
from .vector_store import HybridVectorStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def log_top_k(stage: str, question: str, candidates: list[RetrievedProvision], score_label: str) -> None:
    """Development-only, human-readable retrieval trace for relevance debugging."""
    settings = get_settings()
    if not getattr(settings, "retrieval_debug_logs", False):
        return
    top_k = max(1, getattr(settings, "retrieval_debug_top_k", 8))
    logger.info("Retrieval %s: query=%r, candidates=%d", stage, question[:500], len(candidates))
    for rank, item in enumerate(candidates[:top_k], start=1):
        excerpt = " ".join(item.provision.content.split())
        if len(excerpt) > 280:
            excerpt = f"{excerpt[:277]}..."
        logger.info(
            "Retrieval %s #%d %s=%.4f | %s | %s | Điều %s, Khoản %s, Điểm %s | %s",
            stage,
            rank,
            score_label,
            item.score,
            item.document.code,
            item.version.version_label,
            item.provision.article_no,
            item.provision.clause_no or "-",
            item.provision.point_label or "-",
            excerpt,
        )


class LegalRetriever:
    def __init__(self, db: Session):
        self.db = db

    def retrieve(self, question: str, as_of: date, limit: int = 8, conversation_context: str = "") -> list[RetrievedProvision]:
        return self.retrieve_with_metrics(question, as_of, limit, conversation_context).provisions

    def retrieve_with_metrics(self, question: str, as_of: date, limit: int = 8, conversation_context: str = "") -> RetrievalResult:
        retrieval_started = perf_counter()
        retrieval_query = contextual_retrieval_query(question, conversation_context)
        expanded = expand_commercial_query(retrieval_query)
        if expanded.applied and get_settings().retrieval_debug_logs:
            logger.info("Retrieval query expansion: original=%r expanded=%r", expanded.original, expanded.retrieval_query)
        points = HybridVectorStore().search(
            retrieval_query,
            as_of.isoformat(),
            limit=limit * 2,
            expansion_query=expanded.retrieval_query if expanded.applied else None,
        )
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
        log_top_k("hybrid-recall", question, valid, "rrf_score")
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        # bge-reranker-v2-m3 is deliberately applied after hybrid recall.
        rerank_started = perf_counter()
        provisions = Reranker().rerank(retrieval_query, valid)[:limit]
        log_top_k("reranked", question, provisions, "reranker_score")
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
        scored = [replace(item, score=float(score)) for score, item in ranked]
        log_top_k("reranker-raw", question, scored, "reranker_score")
        return [item for item in scored if item.score >= threshold]


@lru_cache(maxsize=1)
def get_reranker_model():
    """Load bge-reranker-v2-m3 once per API process."""
    import torch
    from FlagEmbedding import FlagReranker
    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=torch.cuda.is_available())
