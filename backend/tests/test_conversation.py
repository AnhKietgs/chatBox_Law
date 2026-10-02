from types import SimpleNamespace
from app.services.conversation import build_conversation_context, contextual_retrieval_query
from app.services.contextualization import parse_standalone_question


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
