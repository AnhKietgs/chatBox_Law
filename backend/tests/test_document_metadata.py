import pytest

from app.services.document_metadata import infer_legacy_domain, normalize_category, normalize_domain


def test_normalizes_admin_domain_aliases_and_rejects_unknown_domain():
    assert normalize_domain("thương mại") == "commercial"
    assert normalize_domain("labor") == "labor"
    with pytest.raises(ValueError):
        normalize_domain("bất kỳ")


def test_normalizes_category_and_inferrs_initial_corpus_domains():
    assert normalize_category("Hợp đồng mua bán") == "hợp_đồng_mua_bán"
    assert infer_legacy_domain("LTM-2005", "Luật Thương mại 2005") == "commercial"
    assert infer_legacy_domain("BLDS-2015", "Bộ luật Dân sự 2015") == "civil"
