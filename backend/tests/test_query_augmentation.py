from app.services.query_augmentation import (
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
