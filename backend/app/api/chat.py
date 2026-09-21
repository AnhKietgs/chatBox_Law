from datetime import date
import logging
from time import perf_counter
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from ..database import get_db
from ..rate_limit import enforce_public_rate_limit
from ..schemas import ChatQuery, ChatResponse, LatencyBreakdown
from ..services.answering import GroundedAnswerService
from ..services.retrieval import LegalRetriever

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/query", response_model=ChatResponse)
async def query(payload: ChatQuery, request: Request, db: Session = Depends(get_db)) -> ChatResponse:
    started = perf_counter()
    enforce_public_rate_limit(request)
    applied_date = payload.as_of_date or date.today()
    retrieval = LegalRetriever(db).retrieve_with_metrics(payload.question, applied_date)
    logger.info("Chat retrieval returned %d grounded provisions for %s", len(retrieval.provisions), applied_date.isoformat())
    generation = await GroundedAnswerService().generate(payload.question, retrieval.provisions, applied_date)
    generation.answer.latency = LatencyBreakdown(
        retrieval_ms=round(retrieval.retrieval_ms, 1),
        rerank_ms=round(retrieval.rerank_ms, 1),
        llm_ms=round(generation.llm_ms, 1),
        citation_validation_ms=round(generation.citation_validation_ms, 1),
        total_ms=round((perf_counter() - started) * 1000, 1),
    )
    logger.info(
        "Chat latency total=%.1fms retrieval=%.1fms rerank=%.1fms llm=%.1fms citation=%.1fms",
        generation.answer.latency.total_ms,
        generation.answer.latency.retrieval_ms,
        generation.answer.latency.rerank_ms,
        generation.answer.latency.llm_ms,
        generation.answer.latency.citation_validation_ms,
    )
    return generation.answer
