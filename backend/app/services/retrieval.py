from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
import logging
import re
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


def score_percentile(scores: list[float], percentile: float) -> float:
    """Return a NumPy-compatible linear percentile without a new dependency."""
    if not scores:
        raise ValueError("Cannot calculate a percentile from an empty score list")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(score) for score in scores)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


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
    minimum_score: float


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

    def retrieve(self, question: str, as_of: date, limit: int = 8, conversation_context: str = "", standalone_query: str | None = None, additional_queries: list[str] | None = None, hyde_query: str | None = None) -> list[RetrievedProvision]:
        return self.retrieve_with_metrics(
            question, as_of,
            conversation_context=conversation_context,
            standalone_query=standalone_query,
            additional_queries=additional_queries,
            hyde_query=hyde_query,
        ).provisions

    def retrieve_with_metrics(self, question: str, as_of: date, limit: int = 8, inject_limit: int | None = None, conversation_context: str = "", standalone_query: str | None = None, additional_queries: list[str] | None = None, hyde_query: str | None = None) -> RetrievalResult:
        settings = get_settings()
        # inject_limit controls how many provisions reach the prompt;
        # recall_pool controls how many candidates the vector store fetches for reranking.
        effective_inject_limit = inject_limit if inject_limit is not None else settings.retrieval_inject_limit
        recall_pool = settings.retrieval_recall_pool
        retrieval_started = perf_counter()
        # `standalone_query` comes from the contextualizer. Keep the old
        # bounded-history composition only as a resilient compatibility fallback.
        retrieval_query = standalone_query or contextual_retrieval_query(question, conversation_context)
        expanded = expand_commercial_query(retrieval_query)
        if expanded.applied and settings.retrieval_debug_logs:
            logger.info("Retrieval query expansion: original=%r expanded=%r", expanded.original, expanded.retrieval_query)
        points = HybridVectorStore().search(
            retrieval_query,
            as_of.isoformat(),
            limit=recall_pool,
            expansion_query=expanded.retrieval_query if expanded.applied else None,
            additional_queries=additional_queries,
            hyde_query=hyde_query,
        )
        ids = [UUID(str(point.id)) for point in points]
        scores = {UUID(str(point.id)): float(point.score) for point in points}
        if not ids:
            return RetrievalResult([], (perf_counter() - retrieval_started) * 1000, 0, settings.confidence_threshold)
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
        # Short follow-ups have less direct lexical overlap. Keep a slightly
        # broader percentile band after contextualization, while still selecting
        # relative to the scores returned for this particular query.
        is_short_follow_up = bool(conversation_context) and len(re.findall(r"\w+", question)) <= 7
        score_percentile_value = (
            settings.conversation_rerank_score_percentile
            if is_short_follow_up
            else settings.rerank_score_percentile
        )
        # ── Domain coherence filter (chỉ áp dụng cho short follow-up) ────────
        # Khi follow-up ngắn có đại từ ("mức đó", "điều này"...), reranker
        # hay bị literal match sang văn bản sai domain (VD: "vượt quá" →
        # BLDS Điều 143 thay vì LTM Điều 301).
        # Tín hiệu đáng tin cậy hơn là hybrid recall: nếu top-K recall đa số
        # từ một văn bản (≥ 60%), giữ lại provisions cùng văn bản đó sau rerank.
        dominant_doc_code: str | None = None
        if is_short_follow_up and valid:
            from collections import Counter
            top_recall = valid[:10]
            doc_counts = Counter(p.document.code for p in top_recall)
            top_code, top_count = doc_counts.most_common(1)[0]
            if top_count >= len(top_recall) * 0.6:
                dominant_doc_code = top_code
                logger.info(
                    "Domain coherence: dominant=%s (%d/%d recall hits) — will filter cross-domain after rerank",
                    dominant_doc_code, top_count, len(top_recall),
                )

        # Reranker dùng retrieval_query (contextualized) để score chính xác
        # hơn về domain; domain coherence filter ở trên lo phần lọc sai văn bản.
        provisions, minimum_score = Reranker().rerank_with_threshold(
            retrieval_query,
            valid,
            percentile=score_percentile_value,
        )
        logger.info(
            "Retrieval reranker percentile=%.1f threshold=%.4f mode=%s candidates=%d",
            score_percentile_value,
            minimum_score,
            "contextual-follow-up" if is_short_follow_up else "standalone",
            len(valid),
        )

        # Áp dụng domain filter sau rerank: loại bỏ provisions không cùng
        # văn bản dominant. Nếu sau lọc không còn gì → answering.py abstain
        # qua reranker_minimum_absolute_score gate hoặc empty-list gate.
        if dominant_doc_code and provisions:
            # Exception: nếu reranker top-1 là từ domain KHÁC với dominant
            # và score cao hơn đáng kể (≥1.5x) → user đang topic-switch sang
            # chủ thể pháp lý mới (VD: "hợp đồng ủy quyền" sau "hợp đồng đại lý")
            # → bỏ qua domain filter để không chặn domain mới.
            top_prov = provisions[0]
            if top_prov.document.code != dominant_doc_code:
                dominant_top_score = max(
                    (p.score for p in provisions if p.document.code == dominant_doc_code),
                    default=0.0,
                )
                if top_prov.score >= dominant_top_score * 1.5:
                    logger.info(
                        "Domain coherence filter: skipped — cross-domain top-1 %s (score=%.3f) outranks dominant %s (%.3f)",
                        top_prov.document.code, top_prov.score,
                        dominant_doc_code, dominant_top_score,
                    )
                    dominant_doc_code = None  # skip filter below

        if dominant_doc_code:
            same_domain = [p for p in provisions if p.document.code == dominant_doc_code]
            if same_domain:
                logger.info(
                    "Domain coherence filter: kept %d %s provisions, dropped %d cross-domain",
                    len(same_domain), dominant_doc_code,
                    len(provisions) - len(same_domain),
                )
                provisions = same_domain
            else:
                # Không có cùng domain sau rerank → trả empty để abstain
                logger.info("Domain coherence filter: no %s provision survived rerank → will abstain", dominant_doc_code)
                provisions = []

        provisions = provisions[:effective_inject_limit]
        log_top_k("reranked", question, provisions, "reranker_score")
        return RetrievalResult(provisions, retrieval_ms, (perf_counter() - rerank_started) * 1000, minimum_score)


class Reranker:
    def rerank_with_threshold(
        self,
        question: str,
        candidates: list[RetrievedProvision],
        minimum_score: float | None = None,
        percentile: float | None = None,
    ) -> tuple[list[RetrievedProvision], float]:
        if not candidates:
            return [], minimum_score if minimum_score is not None else get_settings().confidence_threshold
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
        numeric_scores = [float(score) for score in scores]
        threshold = score_percentile(numeric_scores, percentile) if percentile is not None else (minimum_score if minimum_score is not None else get_settings().confidence_threshold)
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        scored = [replace(item, score=float(score)) for score, item in ranked]
        log_top_k("reranker-raw", question, scored, "reranker_score")
        return [item for item in scored if item.score >= threshold], threshold

    def rerank(self, question: str, candidates: list[RetrievedProvision], minimum_score: float | None = None, percentile: float | None = None) -> list[RetrievedProvision]:
        """Compatibility helper for callers that only need filtered candidates."""
        return self.rerank_with_threshold(question, candidates, minimum_score, percentile)[0]


@lru_cache(maxsize=1)
def get_reranker_model():
    """Load bge-reranker-v2-m3 once per API process."""
    import torch
    from FlagEmbedding import FlagReranker
    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=torch.cuda.is_available())
