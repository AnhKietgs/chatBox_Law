from datetime import date
import logging
from time import perf_counter
from uuid import UUID
from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy import delete as sql_delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from ..database import get_db
from ..config import get_settings
from ..rate_limit import enforce_public_rate_limit
from ..schemas import ChatQuery, ChatResponse, LatencyBreakdown
from ..services.answering import GroundedAnswerService
from ..services.conversation import build_conversation_context
from ..services.retrieval import LegalRetriever
from ..models import ChatMessage, Conversation

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_turn(db: Session, conversation: Conversation, question: str, answer: str) -> UUID:
    """Persist a turn; recover if the client deleted this anonymous chat concurrently."""
    # Preserve the scalar before commit/rollback expires ORM attributes. In the
    # race handled below the database row may already be gone after rollback.
    conversation_id = conversation.id
    try:
        db.add_all([
            ChatMessage(conversation_id=conversation.id, role="user", content=question),
            ChatMessage(conversation_id=conversation.id, role="assistant", content=answer),
        ])
        db.commit()
        return conversation_id
    except IntegrityError:
        db.rollback()
        # A Delete request can win the race after the conversation was read but
        # before these messages were committed.  The answer is still valid, so
        # preserve it in a fresh anonymous conversation instead of returning 500.
        logger.info("Conversation %s was deleted during a chat request; creating a replacement", conversation_id)
        replacement = Conversation()
        db.add(replacement)
        db.flush()
        replacement_id = replacement.id
        db.add_all([
            ChatMessage(conversation_id=replacement.id, role="user", content=question),
            ChatMessage(conversation_id=replacement.id, role="assistant", content=answer),
        ])
        db.commit()
        return replacement_id


@router.post("/query", response_model=ChatResponse)
async def query(payload: ChatQuery, request: Request, db: Session = Depends(get_db)) -> ChatResponse:
    started = perf_counter()
    enforce_public_rate_limit(request)
    applied_date = payload.as_of_date or date.today()
    conversation = db.get(Conversation, payload.conversation_id) if payload.conversation_id else None
    if conversation is None:
        conversation = Conversation()
        db.add(conversation)
        db.flush()
    settings = get_settings()
    prior_messages = list(db.scalars(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation.id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(settings.conversation_context_messages)
    ))
    conversation_context = build_conversation_context(reversed(prior_messages), settings.conversation_context_max_chars)
    retrieval = LegalRetriever(db).retrieve_with_metrics(payload.question, applied_date, conversation_context=conversation_context)
    logger.info("Chat retrieval returned %d grounded provisions for %s", len(retrieval.provisions), applied_date.isoformat())
    generation = await GroundedAnswerService().generate(
        payload.question,
        retrieval.provisions,
        applied_date,
        conversation_context,
        fallback_minimum_score=retrieval.minimum_score,
    )
    generation.answer.conversation_id = save_turn(db, conversation, payload.question, generation.answer.answer)
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


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: UUID, db: Session = Depends(get_db)) -> Response:
    """Delete an anonymous conversation and its messages on the user's request."""
    # Explicit child-first bulk deletes make this endpoint idempotent when a
    # browser retries or two tabs try to delete the same conversation.
    db.execute(sql_delete(ChatMessage).where(ChatMessage.conversation_id == conversation_id))
    db.execute(sql_delete(Conversation).where(Conversation.id == conversation_id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
