from app.services.legal_parser import parse_legal_text


def test_parser_preserves_article_clause_and_point():
    text = """Điều 300. Phạt vi phạm
1. Phạt vi phạm là việc bên bị vi phạm yêu cầu bên vi phạm trả tiền phạt.
2. Mức phạt do các bên thỏa thuận:
a) Có thể áp dụng theo hợp đồng;
b) Phải phù hợp quy định pháp luật.
Điều 301. Giới hạn
Mức phạt không vượt quá mức luật định."""
    provisions = parse_legal_text(text)
    assert [(p.article_no, p.clause_no, p.point_label) for p in provisions] == [
        ("300", "1", None), ("300", "2", None), ("300", "2", "a"), ("300", "2", "b"), ("301", None, None),
    ]


def test_parser_rejects_unstructured_text():
    try:
        parse_legal_text("đây không phải văn bản đã được nhận diện")
    except ValueError as exc:
        assert "Điều" in str(exc)
    else:
        raise AssertionError("Expected parser to reject text without article headers")


def test_parser_preserves_docling_markdown_article_boundaries_and_lists():
    text = """## Điều 300. Phạt vi phạm
- 1. Bên vi phạm phải chịu phạt khi có thỏa thuận.
- 2. Các bên có thể thỏa thuận:
  - a) Mức phạt theo hợp đồng;
  - b) Cách áp dụng phạt.
## Điều 301. Mức phạt vi phạm
Mức phạt không quá 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm.
"""

    provisions = parse_legal_text(text)

    assert [(p.article_no, p.clause_no, p.point_label) for p in provisions] == [
        ("300", "1", None), ("300", "2", None), ("300", "2", "a"),
        ("300", "2", "b"), ("301", None, None),
    ]
    article_300 = "\n".join(p.content for p in provisions if p.article_no == "300")
    assert "Điều 301" not in article_300
