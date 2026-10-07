from datetime import date
import logging
import re
from time import perf_counter
from uuid import UUID
from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy import delete as sql_delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from ..database import get_db
from ..config import get_settings
from ..rate_limit import enforce_public_rate_limit
from ..schemas import ChatQuery, ChatResponse, LatencyBreakdown
from ..services.answering import GroundedAnswerService, abstain
from ..services.conversation import build_conversation_context
from ..services.contextualization import contextualize_for_retrieval
from ..services.memory import summarize_history
from ..services.query_augmentation import (
    augment_for_retrieval,
    clarification_response,
    detect_mixed_legal_domains,
    detect_intent_shift,
    mixed_domain_clarification_response,
    needs_clarification,
)
from ..services.retrieval import LegalRetriever
from ..services.web_search import WebSearchAnswerService, WebSearchService, should_use_web_fallback
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

    # Do this before contextualization and HyDE/MultiQuery.  Those retrieval
    # helpers should not choose one branch of an explicitly mixed question and
    # silently suppress the other (for example: labour + enterprise governance).
    mixed_domains = detect_mixed_legal_domains(payload.question)
    if settings.mixed_domain_clarification_enabled and mixed_domains.is_mixed:
        answer = abstain(applied_date)
        answer.answer = mixed_domain_clarification_response(mixed_domains.domains)
        answer.warnings = [
            "Không dùng HyDE/MultiQuery để tự chọn một lĩnh vực khi câu hỏi kết hợp nhiều lĩnh vực pháp lý."
        ]
        answer.conversation_id = save_turn(db, conversation, payload.question, answer.answer)
        answer.latency = LatencyBreakdown(total_ms=round((perf_counter() - started) * 1000, 1))
        logger.info("Chat abstained before retrieval because mixed legal domains were detected: %s", mixed_domains.domains)
        return answer

    # ------------------------------------------------------------------ #
    # Strategy 4: Sliding window — only fetch the N recent turns          #
    # ------------------------------------------------------------------ #
    window_msg_count = settings.history_window_turns * 2  # user + assistant per turn
    prior_messages = list(db.scalars(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation.id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(window_msg_count)
    ))
    conversation_context = build_conversation_context(
        reversed(prior_messages),
        settings.conversation_context_max_chars,
        memory_summary=conversation.memory_summary,
        window_turns=settings.history_window_turns,
    )
    prior_user_questions = [
        str(message.content) for message in reversed(prior_messages)
        if getattr(message, "role", "") == "user" and getattr(message, "content", None)
    ]
    intent_shift = detect_intent_shift(payload.question, prior_user_questions)
    retrieval_context = conversation_context
    if settings.intent_shift_detection_enabled and intent_shift.shifted:
        # Keep the transcript for the UI/audit trail, but isolate this request
        # from old context for contextualization, retrieval and generation.
        retrieval_context = ""
        logger.info(
            "Chat intent shift detected: previous_domains=%s current_domains=%s; ignoring prior context for this turn",
            intent_shift.previous_domains, intent_shift.current_domains,
        )
    if needs_clarification(payload.question, bool(conversation_context)):
        answer = abstain(applied_date)
        answer.answer = clarification_response()
        answer.warnings = ["Không suy đoán chủ thể hoặc nội dung pháp lý khi câu hỏi chưa đủ rõ."]
        answer.conversation_id = save_turn(db, conversation, payload.question, answer.answer)
        answer.latency = LatencyBreakdown(total_ms=round((perf_counter() - started) * 1000, 1))
        logger.info("Chat abstained before retrieval because the question has an unresolved reference")
        return answer
    contextualized = await contextualize_for_retrieval(payload.question, retrieval_context)
    augmentation = await augment_for_retrieval(contextualized.retrieval_query)
    retrieval = LegalRetriever(db).retrieve_with_metrics(
        payload.question,
        applied_date,
        inject_limit=settings.retrieval_inject_limit,
        conversation_context=retrieval_context,
        standalone_query=contextualized.retrieval_query,
        additional_queries=augmentation.query_variants,
        hyde_query=augmentation.hypothetical_document,
    )
    logger.info("Chat retrieval returned %d grounded provisions for %s", len(retrieval.provisions), applied_date.isoformat())

    # ------------------------------------------------------------------ #
    # Ambiguous standalone query gate                                      #
    # Câu hỏi cực ngắn + không có context + retrieval trả về nhiều văn   #
    # bản khác nhau (mixed domain) → câu hỏi mơ hồ, nên hỏi lại thay vì #
    # đoán domain. Điều kiện:                                             #
    #   1. Không có conversation context (standalone)                     #
    #   2. ≤ 6 từ trong câu hỏi gốc                                      #
    #   3. Không có số điều cụ thể / mã luật trong câu hỏi               #
    #   4. Retrieval trả về ≥ 2 văn bản khác nhau (mixed domain)         #
    # ------------------------------------------------------------------ #
    _HAS_SPECIFIC_REF = re.compile(
        r"\b(điều\s+\d+|khoản\s+\d+|ltm|blds|blhs|bllđ|blds|luật\s+\w+)\b",
        re.IGNORECASE | re.UNICODE,
    )
    if (
        not retrieval_context
        and len(re.findall(r"\w+", payload.question)) <= 6
        and not _HAS_SPECIFIC_REF.search(payload.question)
        and retrieval.provisions
        and len({p.document.code for p in retrieval.provisions}) > 1
    ):
        logger.info(
            "Chat ambiguous standalone query (short=%d words, mixed domains=%s) → clarification",
            len(re.findall(r"\w+", payload.question)),
            {p.document.code for p in retrieval.provisions},
        )
        answer = abstain(applied_date)
        answer.answer = (
            "Câu hỏi chưa xác định rõ lĩnh vực pháp lý. "
            "Vui lòng cho biết thêm bối cảnh — ví dụ: loại hợp đồng, đối tượng áp dụng, "
            "hoặc tên luật bạn muốn tra cứu."
        )
        answer.warnings = ["Không suy đoán domain khi câu hỏi ngắn có thể áp dụng nhiều văn bản khác nhau."]
        answer.conversation_id = save_turn(db, conversation, payload.question, answer.answer)
        answer.latency = LatencyBreakdown(total_ms=round((perf_counter() - started) * 1000, 1))
        return answer

    # RAG is never skipped: only after hybrid recall + reranking has completed
    # do we evaluate its absolute score for the optional web-search fallback.
    # This avoids a "lazy RAG" path where a broad web result replaces a valid
    # provision already present in the approved legal corpus.
    top_rag_score = retrieval.provisions[0].score if retrieval.provisions else None
    if should_use_web_fallback(
        provision_count=len(retrieval.provisions),
        top_score=top_rag_score,
        threshold=settings.web_search_rag_threshold,
    ):
        logger.info(
            "RAG score gate triggered web fallback: provisions=%d top_score=%s threshold=%.4f",
            len(retrieval.provisions),
            f"{top_rag_score:.4f}" if top_rag_score is not None else "none",
            settings.web_search_rag_threshold,
        )
        web_result = await WebSearchService().search(contextualized.retrieval_query)
        if web_result.sources:
            web_summary, used_web_sources, web_llm_ms = await WebSearchAnswerService().generate(
                payload.question,
                web_result.sources,
            )
            if web_summary and used_web_sources:
                answer = abstain(applied_date)
                answer.source_mode = "web_search"
                answer.answer = (
                    "Tôi không có đủ thông tin để cung cấp câu trả lời có căn cứ pháp lý từ kho văn bản đã duyệt. "
                    "Tuy nhiên, dưới đây là thông tin tôi tìm được trên web (chưa được xác minh):\n\n"
                    f"{web_summary}"
                )
                answer.warnings = [
                    "Kết quả dưới đây lấy từ web, chưa phải căn cứ pháp lý đã được quản trị viên duyệt.",
                    "Hãy kiểm tra nguồn gốc hoặc bổ sung văn bản chính thức vào kho trước khi dùng cho quyết định pháp lý.",
                ]
                answer.web_sources = used_web_sources
                answer.conversation_id = save_turn(db, conversation, payload.question, answer.answer)
                answer.latency = LatencyBreakdown(
                    contextualization_ms=round(contextualized.llm_ms, 1),
                    query_augmentation_ms=round(augmentation.llm_ms, 1),
                    retrieval_ms=round(retrieval.retrieval_ms, 1),
                    rerank_ms=round(retrieval.rerank_ms, 1),
                    web_search_ms=round(web_result.elapsed_ms, 1),
                    llm_ms=round(web_llm_ms, 1),
                    total_ms=round((perf_counter() - started) * 1000, 1),
                )
                logger.info(
                    "Chat returned unverified web fallback: sources=%d total=%.1fms",
                    len(used_web_sources), answer.latency.total_ms,
                )
                return answer
        logger.info("Web fallback yielded no usable summary; continuing to safe RAG abstention path")

    generation = await GroundedAnswerService().generate(
        payload.question,
        retrieval.provisions,
        applied_date,
        retrieval_context,
        fallback_minimum_score=retrieval.minimum_score,
        retrieval_query=contextualized.retrieval_query,
    )
    if intent_shift.shifted:
        generation.answer.warnings.append(
            "Đã nhận diện chuyển chủ đề; câu trả lời này không dùng ngữ cảnh pháp lý của lượt chat trước."
        )
    generation.answer.conversation_id = save_turn(db, conversation, payload.question, generation.answer.answer)

    # ------------------------------------------------------------------ #
    # Strategy 4 (cont.): Memory summary — condense turns outside window  #
    # ------------------------------------------------------------------ #
    if settings.memory_summary_enabled:
        total_messages = db.scalar(
            select(func.count(ChatMessage.id))
            .where(ChatMessage.conversation_id == conversation.id)
        )
        if total_messages is not None and total_messages > window_msg_count:
            old_messages = list(db.scalars(
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conversation.id)
                .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
                .limit(total_messages - window_msg_count)
            ))
            try:
                memory_update = await summarize_history(old_messages, conversation.memory_summary)
                if memory_update.summary:
                    conversation.memory_summary = memory_update.summary
                    db.commit()
                    logger.info(
                        "Memory summary updated: %d old turns condensed (%.0fms)",
                        memory_update.turns_summarized,
                        memory_update.llm_ms,
                    )
            except Exception as exc:
                logger.warning("Memory summarization failed (non-fatal): %s", exc)

    generation.answer.latency = LatencyBreakdown(
        contextualization_ms=round(contextualized.llm_ms, 1),
        query_augmentation_ms=round(augmentation.llm_ms, 1),
        retrieval_ms=round(retrieval.retrieval_ms, 1),
        rerank_ms=round(retrieval.rerank_ms, 1),
        llm_ms=round(generation.llm_ms, 1),
        citation_validation_ms=round(generation.citation_validation_ms, 1),
        token_budget_ms=round(generation.token_budget_ms, 1),
        map_reduce_ms=round(generation.map_reduce_ms, 1),
        total_ms=round((perf_counter() - started) * 1000, 1),
    )
    logger.info(
        "Chat latency total=%.1fms contextualize=%.1fms augment=%.1fms "
        "retrieval=%.1fms rerank=%.1fms llm=%.1fms citation=%.1fms "
        "token_budget=%.1fms map_reduce=%.1fms",
        generation.answer.latency.total_ms,
        generation.answer.latency.contextualization_ms,
        generation.answer.latency.query_augmentation_ms,
        generation.answer.latency.retrieval_ms,
        generation.answer.latency.rerank_ms,
        generation.answer.latency.llm_ms,
        generation.answer.latency.citation_validation_ms,
        generation.answer.latency.token_budget_ms,
        generation.answer.latency.map_reduce_ms,
    )
    return generation.answer


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    """Delete an anonymous conversation and its messages on the user's request."""
    enforce_public_rate_limit(request)
    # Explicit child-first bulk deletes make this endpoint idempotent when a
    # browser retries or two tabs try to delete the same conversation.
    db.execute(sql_delete(ChatMessage).where(ChatMessage.conversation_id == conversation_id))
    db.execute(sql_delete(Conversation).where(Conversation.id == conversation_id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
