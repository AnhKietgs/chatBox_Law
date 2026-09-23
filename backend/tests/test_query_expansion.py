from app.services.query_expansion import expand_commercial_query


def test_short_in_scope_query_is_expanded_without_replacing_original():
    result = expand_commercial_query("phạt vi phạm")
    assert result.applied is True
    assert result.original == "phạt vi phạm"
    assert "chế tài vi phạm nghĩa vụ hợp đồng" in result.retrieval_query


def test_short_contract_breach_phrase_expands_to_commercial_penalty_context():
    result = expand_commercial_query("mức vi phạm hợp đồng")
    assert result.applied is True
    assert "chế tài vi phạm nghĩa vụ hợp đồng" in result.retrieval_query
    assert "hợp đồng thương mại mua bán hàng hóa" in result.retrieval_query


def test_detailed_in_scope_query_has_focused_legal_anchors():
    long_result = expand_commercial_query("Mức phạt vi phạm hợp đồng mua bán hàng hóa tối đa là bao nhiêu?")
    assert long_result.applied is True
    assert long_result.retrieval_query != long_result.original
    assert "giới hạn mức phạt theo tỷ lệ phần trăm" in long_result.retrieval_query


def test_out_of_scope_short_query_is_not_expanded():
    outside_scope = expand_commercial_query("Luật hôn nhân và gia đình")
    assert outside_scope.applied is False
    assert outside_scope.retrieval_query == outside_scope.original
