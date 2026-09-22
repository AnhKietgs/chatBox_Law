from app.services.query_expansion import expand_short_commercial_query


def test_short_in_scope_query_is_expanded_without_replacing_original():
    result = expand_short_commercial_query("phạt vi phạm")
    assert result.applied is True
    assert result.retrieval_query.startswith("phạt vi phạm\n")
    assert "chế tài vi phạm nghĩa vụ hợp đồng" in result.retrieval_query


def test_long_query_and_out_of_scope_short_query_are_not_expanded():
    long_result = expand_short_commercial_query("Mức phạt vi phạm trong hợp đồng mua bán hàng hóa được quy định thế nào")
    outside_scope = expand_short_commercial_query("Luật hôn nhân và gia đình")
    assert long_result.applied is False
    assert long_result.retrieval_query == long_result.original
    assert outside_scope.applied is False
    assert outside_scope.retrieval_query == outside_scope.original
