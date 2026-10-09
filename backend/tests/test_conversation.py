import asyncio
from types import SimpleNamespace

from app.services.conversation import build_conversation_context, contextual_retrieval_query
from app.services.contextualization import (
    contextualize_for_retrieval,
    parse_standalone_question,
    resolve_nearest_legal_reference,
)


def test_conversation_context_is_bounded_and_keeps_recent_turns():
    messages = [
        SimpleNamespace(role="user", content="Mức phạt vi phạm là bao nhiêu?"),
        SimpleNamespace(role="assistant", content="Mức phạt tối đa là 8%."),
        SimpleNamespace(role="user", content="Vậy có ngoại lệ không?"),
    ]
    context = build_conversation_context(messages, 500)
    assert "Mức phạt vi phạm" in context
    assert context.endswith("Vậy có ngoại lệ không?")
    assert "Người dùng:" in context


def test_contextual_query_keeps_current_question_and_labels_history_as_non_evidence():
    query = contextual_retrieval_query("Vậy có ngoại lệ không?", "Người dùng: Mức phạt vi phạm là bao nhiêu?")
    assert query.startswith("Câu hỏi hiện tại: Vậy có ngoại lệ không?")
    assert "không phải căn cứ pháp lý" in query


def test_contextualizer_accepts_only_a_bounded_standalone_question_json():
    rewritten = parse_standalone_question(
        '{"standalone_question":"Chế tài khi vi phạm nghĩa vụ trong hợp đồng mua bán hàng hóa là gì?"}'
    )
    assert rewritten == "Chế tài khi vi phạm nghĩa vụ trong hợp đồng mua bán hàng hóa là gì?"
    assert parse_standalone_question("Câu trả lời không phải JSON") is None


def test_article_anaphora_uses_nearest_user_locator_not_older_similar_topic():
    context = "\n".join([
        "Người dùng: Điều 301 LTM quy định gì?",
        "Trợ lý: Điều 301 quy định giới hạn mức phạt vi phạm.",
        "Người dùng: Điều 177 LTM quy định gì?",
        "Trợ lý: Điều 177 quy định việc chấm dứt hợp đồng đại lý.",
    ])

    rewritten = resolve_nearest_legal_reference(
        "Điều này có áp dụng khi hợp đồng không có thỏa thuận phạt không?",
        context,
    )

    assert rewritten == "Điều 177 LTM có áp dụng khi hợp đồng không có thỏa thuận phạt không?"


def test_contextualizer_does_not_call_llm_when_nearest_article_is_exact():
    context = "\n".join([
        "Người dùng: Điều 301 LTM quy định gì?",
        "Trợ lý: Nội dung Điều 301.",
        "Người dùng: Điều 177 LTM quy định gì?",
        "Trợ lý: Nội dung Điều 177.",
    ])

    result = asyncio.run(
        contextualize_for_retrieval(
            "Điều này có áp dụng khi hợp đồng không có thỏa thuận phạt không?",
            context,
        )
    )

    assert result.retrieval_query.startswith("Điều 177 LTM")
    assert result.used_llm is False
