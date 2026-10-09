from app.services.query_augmentation import (
    detect_mixed_legal_domains,
    detect_intent_shift,
    retrieval_domain_filters,
    needs_clarification,
    parse_augmentation,
)


def test_unresolved_reference_without_history_requires_clarification():
    assert needs_clarification("Anh ấy được bao nhiêu?", False) is True
    assert needs_clarification("Anh ấy được bao nhiêu?", True) is False
    assert needs_clarification("Mức phạt vi phạm hợp đồng là bao nhiêu?", False) is False


def test_augmentation_requires_three_distinct_valid_multiqueries_and_hyde():
    parsed = parse_augmentation(
        '''{
          "query_variants": [
            "Mức phạt vi phạm trong hợp đồng thương mại là bao nhiêu?",
            "Giới hạn tiền phạt khi vi phạm hợp đồng mua bán hàng hóa",
            "Quy định về mức phạt vi phạm nghĩa vụ hợp đồng thương mại"
          ],
          "hypothetical_document": "Văn bản pháp luật có thể quy định điều kiện áp dụng và giới hạn mức phạt khi một bên vi phạm nghĩa vụ trong hợp đồng thương mại."
        }''',
        "Mức phạt vi phạm là bao nhiêu?",
    )
    assert parsed is not None
    assert len(parsed.query_variants) == 3
    assert parsed.hypothetical_document is not None

    invalid = parse_augmentation(
        '{"query_variants":["Mức phạt vi phạm là bao nhiêu?"],"hypothetical_document":"Đoạn giả định đủ dài nhưng không có đủ các truy vấn thay thế cần thiết."}',
        "Mức phạt vi phạm là bao nhiêu?",
    )
    assert invalid is None


def test_detects_explicitly_mixed_labor_and_enterprise_governance_query():
    result = detect_mixed_legal_domains(
        "Người lao động được trả lương thế nào và vốn điều lệ khi thành lập doanh nghiệp là bao nhiêu?"
    )
    assert result.is_mixed is True
    assert result.domains == ("lao động", "doanh nghiệp")


def test_employer_mention_does_not_turn_labor_question_into_mixed_domain_query():
    result = detect_mixed_legal_domains("Doanh nghiệp phải trả lương cho người lao động khi nào?")
    assert result.is_mixed is False
    assert result.domains == ("lao động",)


def test_detects_enterprise_and_commercial_question_but_not_normal_commercial_question():
    mixed = detect_mixed_legal_domains(
        "Thành lập doanh nghiệp và mức phạt vi phạm hợp đồng mua bán hàng hóa được quy định thế nào?"
    )
    assert mixed.is_mixed is True
    assert mixed.domains == ("doanh nghiệp", "thương mại")

    commercial_only = detect_mixed_legal_domains("Mức phạt vi phạm hợp đồng mua bán hàng hóa tối đa là bao nhiêu?")
    assert commercial_only.is_mixed is False
    assert commercial_only.domains == ("thương mại",)


def test_converts_explicit_query_domains_to_qdrant_filter_values():
    assert retrieval_domain_filters("Mức phạt hợp đồng mua bán hàng hóa là bao nhiêu?") == ("commercial",)
    assert retrieval_domain_filters("Thành lập doanh nghiệp cần vốn điều lệ bao nhiêu?") == ("enterprise",)
    assert retrieval_domain_filters("Hợp đồng là gì?") == ()
    assert retrieval_domain_filters("Điều 301 LTM quy định gì?") == ("commercial",)
    assert retrieval_domain_filters("Điều 301 BLDS quy định gì?") == ("civil",)


def test_detects_clear_cross_domain_topic_switch_and_keeps_same_domain_context():
    shifted = detect_intent_shift(
        "Người lao động được nghỉ phép như thế nào?",
        ["Mức phạt vi phạm hợp đồng mua bán hàng hóa tối đa là bao nhiêu?"],
    )
    assert shifted.shifted is True
    assert shifted.previous_domains == ("thương mại",)
    assert shifted.current_domains == ("lao động",)

    continuous = detect_intent_shift(
        "Hàng hóa cấm kinh doanh được quy định thế nào?",
        ["Mức phạt vi phạm hợp đồng mua bán hàng hóa tối đa là bao nhiêu?"],
    )
    assert continuous.shifted is False


def test_generic_or_mixed_current_question_does_not_discard_conversation_context():
    assert detect_intent_shift("Nếu vi phạm thì sao?", ["Hợp đồng mua bán hàng hóa là gì?"]).shifted is False
    assert detect_intent_shift(
        "Người lao động được trả lương thế nào và vốn điều lệ là bao nhiêu?",
        ["Hợp đồng mua bán hàng hóa là gì?"],
    ).shifted is False
