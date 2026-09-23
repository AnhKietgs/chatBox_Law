from __future__ import annotations
import json
import logging
import re
from datetime import date
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID
from pydantic import BaseModel, Field, ValidationError
from ..config import get_settings
from ..schemas import ChatResponse, Citation, Claim
if TYPE_CHECKING:
    from .retrieval import RetrievedProvision

ABSTENTION = "Tôi chưa có đủ thông tin để trả lời câu hỏi này."
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def remove_internal_source_labels(text: str) -> str:
    """Source IDs are for backend citation validation, never for end users."""
    return re.sub(r"\s*[\(\[]\s*S\d+(?:\s*[,;]\s*S\d+)*\s*[\)\]]", "", text).strip()


def is_question_echo(answer: str, question: str) -> bool:
    """A model repeating the question is not a grounded legal answer."""
    normalized_answer = re.sub(r"[^\w]+", "", answer.casefold())
    normalized_question = re.sub(r"[^\w]+", "", question.casefold())
    return len(normalized_question) >= 8 and normalized_answer == normalized_question


class ModelClaim(BaseModel):
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class ModelAnswer(BaseModel):
    answer: str = Field(min_length=1)
    claims: list[ModelClaim] = []
    warnings: list[str] = []
    abstain: bool = False


@dataclass(frozen=True)
class AnswerGeneration:
    answer: ChatResponse
    llm_ms: float
    citation_validation_ms: float


def citation_from(item: RetrievedProvision) -> Citation:
    provision, version, document = item.provision, item.version, item.document
    return Citation(
        id=provision.id,
        document_code=document.code,
        document_title=document.title,
        version_label=version.version_label,
        article=provision.article_no,
        clause=provision.clause_no,
        point=provision.point_label,
        excerpt=provision.content,
        official_url=version.official_url,
        relevance_score=item.score,
    )


def validate_model_answer(raw: str, retrieved: list[RetrievedProvision], as_of: date, question: str | None = None) -> ChatResponse:
    """Reject an answer unless every generated claim cites a retrieved provision."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].removesuffix("```").strip()
        payload: object = json.loads(cleaned)
        # Small models sometimes serialize an assistant message as JSON before
        # serializing the requested JSON payload.  Accept that harmless wrapper,
        # but only after extracting and validating its nested content below.
        for _ in range(2):
            if not isinstance(payload, dict) or "answer" in payload or "content" not in payload:
                break
            nested = payload["content"]
            payload = json.loads(nested) if isinstance(nested, str) else nested
        proposed = ModelAnswer.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Rejected LLM output: invalid JSON or answer schema (%s)", exc)
        return abstain(as_of)
    if question and is_question_echo(proposed.answer, question):
        logger.warning("Rejected LLM output: answer merely repeats the question")
        return abstain(as_of)
    if proposed.abstain or not proposed.claims:
        logger.info("LLM abstained because the retrieved sources did not directly support the question")
        return abstain(as_of)
    available = {f"S{index}": item for index, item in enumerate(retrieved, start=1)}
    cited_source_ids = {citation_id.upper() for claim in proposed.claims for citation_id in claim.citation_ids}
    if not cited_source_ids or not cited_source_ids.issubset(available):
        logger.warning("Rejected LLM output: citations are absent or outside retrieved provisions")
        return abstain(as_of)
    citations = [citation_from(available[source_id]) for source_id in cited_source_ids]
    return ChatResponse(
        status="grounded",
        answer=remove_internal_source_labels(proposed.answer),
        claims=[Claim(text=claim.text, citation_ids=[available[source_id.upper()].provision.id for source_id in claim.citation_ids]) for claim in proposed.claims],
        citations=citations,
        warnings=[*proposed.warnings, "Thông tin có tính tham khảo, không thay thế tư vấn pháp lý cho tình huống cụ thể."],
        applied_as_of_date=as_of,
    )


def abstain(as_of: date) -> ChatResponse:
    return ChatResponse(status="abstained", answer=ABSTENTION, warnings=["Không suy đoán khi không có căn cứ Điều/Khoản/Điểm."], applied_as_of_date=as_of)


def claims_are_supported(answer: ChatResponse, retrieved: list[RetrievedProvision]) -> bool:
    """Require each LLM claim to be semantically supported by one cited excerpt."""
    source_by_id = {item.provision.id: item for item in retrieved}
    pairs: list[list[str]] = []
    claim_ranges: list[tuple[int, int]] = []
    for claim in answer.claims:
        start = len(pairs)
        for citation_id in claim.citation_ids:
            item = source_by_id.get(citation_id)
            if item is None:
                return False
            passage = f"Điều {item.provision.article_no}. {item.provision.heading or ''}\n{item.provision.content}"
            pairs.append([claim.text, passage])
        claim_ranges.append((start, len(pairs)))
    if not pairs:
        return False
    try:
        from .retrieval import get_reranker_model
        scores = get_reranker_model().compute_score(pairs, normalize=True)
    except Exception as exc:
        logger.warning("Claim-to-citation verification failed: %s", exc)
        return False
    if isinstance(scores, float):
        scores = [scores]
    threshold = get_settings().claim_support_threshold
    return all(max(float(score) for score in scores[start:end]) >= threshold for start, end in claim_ranges)


class GroundedAnswerService:
    async def generate(self, question: str, retrieved: list[RetrievedProvision], as_of: date, conversation_context: str = "") -> AnswerGeneration:
        if not retrieved:
            logger.info("Abstaining because retrieval returned no eligible provision")
            return AnswerGeneration(abstain(as_of), 0, 0)
        logger.info("Generating grounded answer with Ollama chat JSON schema v2 (%d sources)", len(retrieved))
        sources = [{
            "source_id": f"S{index}",
            "citation": f"{item.document.title}, Điều {item.provision.article_no}, Khoản {item.provision.clause_no or '-'}, Điểm {item.provision.point_label or '-'}",
            "text": item.provision.content,
        } for index, item in enumerate(retrieved, start=1)]
        system_prompt = (
            "Bạn là trợ lý tra cứu pháp luật Việt Nam. Chỉ dùng các nguồn được cung cấp; "
            "không suy diễn và không viện dẫn nguồn ngoài. Phải tuân thủ JSON Schema đã được cung cấp. "
            "citation_ids phải là source_id ngắn từ nguồn (ví dụ S1), không được tự tạo mã khác. "
            "S1, S2 và mọi source_id chỉ được đặt trong trường citation_ids; tuyệt đối không viết chúng "
            "trong answer, claim text hoặc bất kỳ nội dung hiển thị cho người dùng. "
            "Lịch sử hội thoại chỉ dùng để hiểu các từ tham chiếu như 'điều đó' hoặc 'trường hợp trên'; "
            "không phải căn cứ pháp lý và không được dùng để tạo kết luận nếu nguồn không hỗ trợ. "
            "Nếu nguồn được cung cấp không trực tiếp trả lời câu hỏi, hãy trả về JSON hợp lệ với "
            "abstain=true, answer='Không đủ căn cứ pháp lý từ các nguồn được cung cấp.', claims=[] và warnings=[]. "
            "Không viết Markdown, giải thích ngoài JSON hoặc phần suy luận."
        )
        user_prompt = (
            f"Ngày áp dụng: {as_of.isoformat()}. "
            f"Lịch sử hội thoại (không phải căn cứ pháp lý): {conversation_context or '(không có)'}\n"
            f"Câu hỏi hiện tại: {question}\n"
            f"Nguồn: {json.dumps(sources, ensure_ascii=False)}"
        )
        settings = get_settings()
        llm_started = perf_counter()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=120) as client:
                # Qwen exposes its final answer in message.content through Ollama's chat API.
                # Disabling thinking prevents a reasoning-only response with blank content.
                response = await client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": settings.ollama_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "format": ModelAnswer.model_json_schema(),
                        "think": False,
                        "keep_alive": "30m",
                        "options": {"temperature": 0},
                    },
                )
                response.raise_for_status()
                body = response.json()
                raw = str(body.get("message", {}).get("content", "")).strip()
                if not raw:
                    logger.warning(
                        "Ollama returned empty chat content (done_reason=%s, eval_count=%s)",
                        body.get("done_reason"),
                        body.get("eval_count"),
                    )
                    return AnswerGeneration(abstain(as_of), (perf_counter() - llm_started) * 1000, 0)
        except Exception as exc:
            logger.warning("Ollama generation failed; returning safe abstention: %s", exc)
            return AnswerGeneration(abstain(as_of), (perf_counter() - llm_started) * 1000, 0)
        llm_ms = (perf_counter() - llm_started) * 1000
        validation_started = perf_counter()
        answer = validate_model_answer(raw, retrieved, as_of, question)
        if answer.status == "grounded" and not claims_are_supported(answer, retrieved):
            logger.warning("Rejected LLM output: a claim is not supported by its cited excerpt")
            answer = abstain(as_of)
        return AnswerGeneration(answer, llm_ms, (perf_counter() - validation_started) * 1000)
