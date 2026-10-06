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
    """Source IDs are for backend citation validation, never for end users.
    Also strips [HeadingText] prefixes that source_text() injects and LLMs may echo back."""
    # Strip (S1), [S1], (S1, S2) etc.
    text = re.sub(r"\s*[\(\[]\s*S\d+(?:\s*[,;]\s*S\d+)*\s*[\)\]]", "", text)
    # Strip leading [Heading text] injected from source_text() (3–100 chars, not a source ID)
    text = re.sub(r"^\s*\[(?!S\d)[^\]]{3,100}\]\s*", "", text)
    return text.strip()


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
    token_budget_ms: float = 0.0
    map_reduce_ms: float = 0.0


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


def extractive_fallback(retrieved: list[RetrievedProvision], as_of: date, minimum_score: float | None = None) -> ChatResponse:
    """Return a verified provision verbatim when a small LLM fails despite strong recall."""
    threshold = minimum_score if minimum_score is not None else get_settings().extractive_fallback_threshold
    if not retrieved or retrieved[0].score < threshold:
        return abstain(as_of)
    item = retrieved[0]
    citation = citation_from(item)
    locator = f"Điều {citation.article}"
    if citation.clause:
        locator += f", Khoản {citation.clause}"
    if citation.point:
        locator += f", Điểm {citation.point}"
    logger.info("Using citation-first fallback from %s (%s, score=%.3f)", citation.document_code, locator, item.score)
    return ChatResponse(
        status="grounded",
        answer=f"Theo {citation.document_title}, {locator}: {item.provision.content}",
        claims=[Claim(text=item.provision.content, citation_ids=[item.provision.id])],
        citations=[citation],
        warnings=["Câu trả lời được trích nguyên văn từ căn cứ pháp lý đã kiểm chứng.", "Thông tin có tính tham khảo, không thay thế tư vấn pháp lý cho tình huống cụ thể."],
        applied_as_of_date=as_of,
    )


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


_FOREIGN_LAW_PATTERNS = (
    # Tên quốc gia/vùng lãnh thổ + "luật/pháp luật/quy định" theo nhiều thứ tự
    r"(?:luật|pháp\s*luật|quy\s*định)\s+(?:của\s+)?(?:mỹ|hoa\s*kỳ|anh|pháp|đức|nhật|trung\s*quốc|hàn\s*quốc|eu|liên\s*minh\s*châu\s*âu|singapore|úc|canada)",
    r"(?:mỹ|hoa\s*kỳ|anh|pháp|đức|nhật|trung\s*quốc|hàn\s*quốc|eu|liên\s*minh\s*châu\s*âu|singapore|úc|canada)\s+(?:luật|pháp\s*luật|quy\s*định)",
    r"luật\s+(?:mỹ|hoa\s*kỳ|anh|pháp|đức|nhật|trung\s*quốc|hàn\s*quốc|eu)",
    r"(?:us|uk|eu)\s+law",
    r"american\s+law",
    r"common\s+law",
)
_FOREIGN_LAW_RE = re.compile("|".join(_FOREIGN_LAW_PATTERNS), re.IGNORECASE)


class GroundedAnswerService:
    async def generate(self, question: str, retrieved: list[RetrievedProvision], as_of: date, conversation_context: str = "", fallback_minimum_score: float | None = None, retrieval_query: str | None = None) -> AnswerGeneration:
        if not retrieved:
            logger.info("Abstaining because retrieval returned no eligible provision")
            return AnswerGeneration(abstain(as_of), 0, 0)

        # ── Jurisdiction guard: câu hỏi về pháp luật nước ngoài → abstain ngay ──
        # LLM nhỏ (≤3B) không thể tự phân biệt jurisdiction qua instruction.
        # Detect ở code level trước khi gọi LLM để tránh trả nguồn VN cho câu hỏi nước ngoài.
        if _FOREIGN_LAW_RE.search(question):
            logger.info("Abstaining: question asks about foreign law — jurisdiction not supported")
            return AnswerGeneration(abstain(as_of), 0, 0)

        settings = get_settings()

        # ── Absolute reranker score gate ──────────────────────────────────
        # Nếu top result score < minimum → không có source nào đủ liên quan,
        # abstain ngay. Tránh LLM thấy source "gần đúng" rồi sinh answer sai.
        top_score = retrieved[0].score
        if top_score < settings.reranker_minimum_absolute_score:
            logger.info(
                "Abstaining: top reranker score %.4f < minimum %.4f — no relevant source found",
                top_score, settings.reranker_minimum_absolute_score,
            )
            return AnswerGeneration(abstain(as_of), 0, 0)

        logger.info("Generating grounded answer with Ollama chat JSON schema v2 (%d sources)", len(retrieved))

        # Build the source list (without score — that key is internal only).
        # Khi provision là khoản/điểm con (clause_no không None), inject heading cấp
        # Điều vào trước nội dung. Giúp LLM biết ngữ cảnh: VD Điều 131 K4
        # "Bên có lỗi gây thiệt hại thì phải bồi thường" trở thành
        # "[Hậu quả pháp lý của giao dịch dân sự vô hiệu] Bên có lỗi..."
        # → LLM nhận ra đây là hậu quả của vô hiệu, không phải vi phạm hợp đồng thông thường.
        def source_text(item: RetrievedProvision) -> str:
            # article_heading = batch-load từ DB (provision cấp Điều, clause_no=None).
            # Nếu parser không tạo provision cấp Điều riêng mà lưu heading vào từng khoản,
            # article_heading sẽ là None → fallback về provision.heading của chính khoản đó.
            # VD: Điều 131 K4: provision.heading = "Hậu quả pháp lý của giao dịch dân sự vô hiệu"
            # → LLM nhận "[Hậu quả pháp lý...] Bên có lỗi gây thiệt hại thì phải bồi thường."
            heading = item.article_heading or item.provision.heading
            if heading and item.provision.clause_no is not None:
                return f"[{heading}] {item.provision.content}"
            return item.provision.content

        sources = [{
            "source_id": f"S{index}",
            "citation": f"{item.document.title}, Điều {item.provision.article_no}, Khoản {item.provision.clause_no or '-'}, Điểm {item.provision.point_label or '-'}",
            "text": source_text(item),
        } for index, item in enumerate(retrieved, start=1)]

        # Reorder sources list trước khi gửi LLM để provisions có heading "Hậu quả pháp lý"
        # xuất hiện ở đầu context cho câu hỏi về quyền lợi/hậu quả.
        # Small LLMs (≤3B) có recency/primacy bias — chúng bỏ qua sources ở giữa/cuối.
        # Source IDs (S1-S7) giữ nguyên → validation trong validate_model_answer không bị ảnh hưởng
        # (validate_model_answer dùng enumerate(retrieved) độc lập với thứ tự sources list).
        _consequence_triggers = ("được gì", "có quyền gì", "hậu quả", "xử lý", "được bồi", "quyền lợi")
        if any(kw in question.lower() for kw in _consequence_triggers):
            _hq_kw = ("hậu quả pháp lý",)
            def _is_consequence_src(item: RetrievedProvision) -> bool:
                h = (item.article_heading or item.provision.heading or "").lower()
                return any(kw in h for kw in _hq_kw)
            paired = list(zip(sources, retrieved))
            paired.sort(key=lambda p: (0 if _is_consequence_src(p[1]) else 1))
            sources = [s for s, _ in paired]
            logger.info(
                "Answering: reordered sources for consequence question — "
                "consequence provisions moved to front: %s",
                [s["source_id"] for s, r in paired if _is_consequence_src(r)],
            )

        system_prompt = (
            "Bạn là trợ lý tra cứu pháp luật Việt Nam. Chỉ dùng các nguồn được cung cấp; "
            "không suy diễn và không viện dẫn nguồn ngoài. Phải tuân thủ JSON Schema đã được cung cấp. "
            "citation_ids phải là source_id ngắn từ nguồn (ví dụ S1), không được tự tạo mã khác. "
            "S1, S2 và mọi source_id chỉ được đặt trong trường citation_ids; tuyệt đối không viết chúng "
            "trong answer, claim text hoặc bất kỳ nội dung hiển thị cho người dùng. "
            "TUYỆT ĐỐI không viết '[...]' hay '(...)' hay bất kỳ tiêu đề/nhãn nội bộ nào vào trường answer — "
            "answer chỉ được chứa văn bản thuần trả lời câu hỏi cho người dùng đọc. "
            "Lịch sử hội thoại chỉ dùng để hiểu các từ tham chiếu như 'điều đó' hoặc 'trường hợp trên'; "
            "không phải căn cứ pháp lý và không được dùng để tạo kết luận nếu nguồn không hỗ trợ. "
            "QUAN TRỌNG — Quy tắc trả lời vs abstain:\n"
            "- NÊN trả lời khi: nguồn chứa đúng điều/khoản được hỏi, hoặc nội dung nguồn trực tiếp mô tả "
            "khái niệm/quy định mà câu hỏi đề cập. Trả lời bằng cách tóm tắt hoặc trích nội dung nguồn.\n"
            "- PHẢI abstain=true khi: (1) không có nguồn nào đề cập đến chủ thể câu hỏi; "
            "(2) câu hỏi hỏi về HẬU QUẢ/XỬ LÝ mà không có nguồn nào quy định hậu quả đó; "
            "(3) câu hỏi hỏi về pháp luật nước ngoài (Mỹ, EU, Nhật, Trung Quốc, v.v.) "
            "trong khi tất cả các nguồn được cung cấp đều là pháp luật Việt Nam — "
            "KHÔNG dùng nguồn Việt Nam để trả lời câu hỏi về pháp luật nước khác. "
            "Ví dụ cụ thể cần abstain: 'hậu quả khi vượt quá X' mà không có nguồn nào nói hậu quả; "
            "'điều gì xảy ra khi vi phạm Y' mà chỉ có nguồn định nghĩa Y; "
            "'Luật Mỹ/EU quy định gì' khi tất cả nguồn là BLDS/luật Việt Nam. "
            "Không suy luận hậu quả từ định nghĩa. "
            "Nếu nguồn có quy định trực tiếp, phải trả lời ngắn gọn từ nguồn đó. Tạo từ một đến ba claims "
            "trực tiếp trả lời câu hỏi; không thêm định nghĩa hoặc ngoại lệ không liên quan đến câu hỏi. "
            "ĐẶC BIỆT — Câu hỏi về quyền lợi/hậu quả ('được gì', 'có quyền gì', 'hậu quả là gì', 'xử lý thế nào'): "
            "Provision có dạng '[Hậu quả pháp lý...] bên A phải bồi thường' trả lời trực tiếp cho 'bên B được gì' "
            "(nghĩa vụ bồi thường của A = quyền được bồi thường của B). Không bỏ qua provision nào có heading "
            "'Hậu quả pháp lý' và nội dung liên quan trực tiếp đến tình huống được hỏi. "
            "Chỉ cite những provisions có nội dung LIÊN QUAN ĐẾN ĐÚNG CHỦ THỂ câu hỏi; "
            "không cite provisions về trường hợp khác chỉ vì chúng có từ khóa tương tự. "
            "TUYỆT ĐỐI không lặp lại câu hỏi trong trường answer — answer phải bắt đầu ngay bằng nội dung trả lời. "
            "Khi nguồn có nhiều nhánh điều kiện phức tạp (VD: 'nếu A thì X; nếu B thì Y'), "
            "phải trích dẫn CHÍNH XÁC từng điều kiện và hậu quả tương ứng, không được hoán đổi hoặc nhầm lẫn giữa các nhánh. "
            "Nếu nguồn được cung cấp không trực tiếp trả lời câu hỏi, hãy trả về JSON hợp lệ với "
            "abstain=true, answer='Không đủ căn cứ pháp lý từ các nguồn được cung cấp.', claims=[] và warnings=[]. "
            "Không viết Markdown, giải thích ngoài JSON hoặc phần suy luận."
        )

        # ------------------------------------------------------------------ #
        # Strategy 1: Token budget — measure before injecting                 #
        # ------------------------------------------------------------------ #
        from .token_budget import TokenBudget
        from .map_reduce import MapReduceSummarizer

        # LLM luôn nhận câu hỏi gốc của user — không dùng retrieval_query vì:
        # 1) retrieval_query được tối ưu cho vector search, không phải cho LLM
        # 2) Nếu dùng retrieval_query, LLM có xu hướng echo lại câu dài đó
        # 3) retrieval_query "bị xử lý như thế nào" có thể kéo sources sai (Điều 321)
        # LLM đã có conversation_context để hiểu ngữ cảnh đầy đủ.
        effective_question = question

        budget_started = perf_counter()
        tb = TokenBudget(
            settings.context_model_limit,
            settings.context_reserved_output,
            settings.context_overhead,
        )
        # fixed_text = everything in the prompt except the source list
        fixed_text = (
            system_prompt
            + f"\nNgày áp dụng: {as_of.isoformat()}."
            + f"\nLịch sử hội thoại: {conversation_context or '(không có)'}"
            + f"\nCâu hỏi hiện tại: {effective_question}"
        )
        sources_fitted, was_truncated = tb.fit_sources_to_budget(sources, fixed_text)
        token_budget_ms = (perf_counter() - budget_started) * 1000

        # ------------------------------------------------------------------ #
        # Strategy 3: Map-Reduce fallback when truncated and enabled          #
        # ------------------------------------------------------------------ #
        map_reduce_ms = 0.0
        if was_truncated:
            logger.warning(
                "TokenBudget: %d/%d sources fit in prompt budget (map_reduce_enabled=%s)",
                len(sources_fitted), len(sources), settings.map_reduce_enabled,
            )
            if settings.map_reduce_enabled:
                mr_started = perf_counter()
                mr_result = await MapReduceSummarizer().summarize(
                    question=question,
                    sources=sources,          # full list, not just fitted
                    target_token_budget=tb.available(fixed_text),
                )
                map_reduce_ms = (perf_counter() - mr_started) * 1000
                if mr_result.merged_text:
                    logger.info(
                        "MapReduce: replaced %d sources with %d-batch summary (%d tokens)",
                        len(sources), mr_result.batch_count, mr_result.total_input_tokens,
                    )
                    # Map-Reduce tạo ra synthetic source — citation validation sẽ
                    # không thể verify S1 synthetic này với retrieved provisions gốc.
                    # Dùng extractive_fallback từ top-1 retrieved provision (đã verified).
                    return AnswerGeneration(
                        extractive_fallback(retrieved, as_of, fallback_minimum_score),
                        0, 0, token_budget_ms, map_reduce_ms,
                    )
                else:
                    logger.warning("MapReduce produced empty result; using fitted subset")

        user_prompt = (
            f"Ngày áp dụng: {as_of.isoformat()}. "
            f"Lịch sử hội thoại (không phải căn cứ pháp lý): {conversation_context or '(không có)'}\n"
            f"Câu hỏi hiện tại: {effective_question}\n"
            f"Nguồn: {json.dumps(sources_fitted, ensure_ascii=False)}"
        )

        if not sources_fitted:
            logger.warning("No sources fit the token budget; returning extractive fallback")
            return AnswerGeneration(
                extractive_fallback(retrieved, as_of, fallback_minimum_score),
                0, 0, token_budget_ms, map_reduce_ms,
            )

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
                    return AnswerGeneration(
                        extractive_fallback(retrieved, as_of, fallback_minimum_score),
                        (perf_counter() - llm_started) * 1000, 0, token_budget_ms, map_reduce_ms,
                    )
        except Exception as exc:
            logger.warning("Ollama generation failed; returning safe abstention: %s", exc)
            return AnswerGeneration(
                extractive_fallback(retrieved, as_of, fallback_minimum_score),
                (perf_counter() - llm_started) * 1000, 0, token_budget_ms, map_reduce_ms,
            )
        llm_ms = (perf_counter() - llm_started) * 1000
        # DEBUG: log raw LLM JSON để trace xem model chọn source nào
        logger.info("LLM raw output (first 800 chars): %s", raw[:800])
        validation_started = perf_counter()
        answer = validate_model_answer(raw, retrieved, as_of, question)
        if answer.status == "abstained":
            # LLM đã đánh giá nguồn không đủ để trả lời → trả abstain thật,
            # không dùng extractive_fallback vì source top-1 có thể không
            # liên quan đến câu hỏi hiện tại (chỉ liên quan đến context).
            logger.info("LLM abstained — returning hard abstain (no extractive fallback)")
            answer = abstain(as_of)
        elif not claims_are_supported(answer, retrieved):
            logger.warning("Rejected LLM output: a claim is not supported by its cited excerpt")
            answer = extractive_fallback(retrieved, as_of, fallback_minimum_score)
        else:
            # ── Post-injection: "Hậu quả pháp lý" provisions bị LLM bỏ qua ──
            # Small LLMs (≤3B) thường bỏ qua provisions ở giữa/cuối context kể
            # cả khi đã reorder. Với câu hỏi về quyền lợi/hậu quả, tự động inject
            # provisions có heading "Hậu quả pháp lý" chưa được cite vào claims.
            _consequence_q = ("được gì", "có quyền gì", "hậu quả", "xử lý", "được bồi", "quyền lợi")
            if answer.status == "grounded" and any(kw in question.lower() for kw in _consequence_q):
                cited_prov_ids = {str(cid) for claim in answer.claims for cid in claim.citation_ids}
                for item in retrieved:
                    h = (item.article_heading or item.provision.heading or "").lower()
                    if "hậu quả pháp lý" not in h:
                        continue
                    if str(item.provision.id) in cited_prov_ids:
                        continue
                    # Inject provision này vào đầu claims list
                    new_citation = citation_from(item)
                    new_claim = Claim(
                        text=source_text(item),
                        citation_ids=[item.provision.id],
                    )
                    answer = ChatResponse(
                        status=answer.status,
                        answer=answer.answer,
                        claims=[new_claim, *answer.claims],
                        citations=[new_citation, *answer.citations],
                        warnings=answer.warnings,
                        applied_as_of_date=answer.applied_as_of_date,
                    )
                    logger.info(
                        "Post-inject: added uncited 'Hậu quả pháp lý' provision %s K%s to claims",
                        item.provision.article_no, item.provision.clause_no,
                    )
        return AnswerGeneration(answer, llm_ms, (perf_counter() - validation_started) * 1000, token_budget_ms, map_reduce_ms)
