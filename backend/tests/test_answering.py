import json
import sys
from datetime import date
from types import ModuleType, SimpleNamespace
from uuid import uuid4
from app.models import LegalDocument, LegalProvision, LegalVersion, VersionStatus
from app.services import answering
from app.services.answering import claims_are_supported, extractive_fallback, validate_model_answer


def candidate():
    doc = LegalDocument(id=uuid4(), code="LTM-2005", title="Luật Thương mại")
    version = LegalVersion(id=uuid4(), document_id=doc.id, version_label="2005", official_url="https://vbpl.vn/example", effective_from=date(2006, 1, 1), status=VersionStatus.published, source_object_key="source.pdf")
    provision = LegalProvision(id=uuid4(), version_id=version.id, article_no="301", clause_no=None, point_label=None, content="Mức phạt không quá tám phần trăm giá trị phần nghĩa vụ hợp đồng bị vi phạm.", ordinal=1)
    return SimpleNamespace(provision=provision, version=version, document=doc, score=0.9)


def test_grounded_response_only_accepts_retrieved_ids():
    item = candidate()
    raw = json.dumps({"answer": "Có giới hạn.", "claims": [{"text": "Có giới hạn 8%.", "citation_ids": ["S1"]}]})
    result = validate_model_answer(raw, [item], date.today())
    assert result.status == "grounded"
    assert result.citations[0].article == "301"


def test_unknown_citation_forces_abstention():
    raw = json.dumps({"answer": "Sai nguồn", "claims": [{"text": "Sai", "citation_ids": ["S99"]}]})
    assert validate_model_answer(raw, [candidate()], date.today()).status == "abstained"


def test_question_echo_forces_abstention():
    question = "Luật hôn nhân và gia đình"
    raw = json.dumps({"answer": question, "claims": [{"text": "Có giới hạn 8%.", "citation_ids": ["S1"]}]})
    assert validate_model_answer(raw, [candidate()], date.today(), question).status == "abstained"


def test_model_can_abstain_with_a_schema_valid_response():
    raw = json.dumps({"answer": "Không đủ căn cứ pháp lý từ các nguồn được cung cấp.", "claims": [], "warnings": [], "abstain": True})
    assert validate_model_answer(raw, [candidate()], date.today(), "Câu hỏi ngoài phạm vi").status == "abstained"


def test_high_confidence_source_uses_extractive_fallback(monkeypatch):
    item = candidate()
    monkeypatch.setattr(answering, "get_settings", lambda: SimpleNamespace(extractive_fallback_threshold=0.85))
    result = extractive_fallback([item], date.today())
    assert result.status == "grounded"
    assert result.citations[0].article == "301"
    assert result.claims[0].citation_ids == [item.provision.id]


def test_assistant_message_wrapper_is_unwrapped_before_validation():
    answer = {"answer": "Có giới hạn.", "claims": [{"text": "Có giới hạn 8%.", "citation_ids": ["S1"]}]}
    raw = json.dumps({"role": "assistant", "content": json.dumps(answer)})
    assert validate_model_answer(raw, [candidate()], date.today()).status == "grounded"


def test_claim_must_be_supported_by_its_cited_excerpt(monkeypatch):
    class FakeReranker:
        def compute_score(self, pairs, normalize):
            assert normalize is True
            return [0.95]

    retrieval_stub = ModuleType("app.services.retrieval")
    retrieval_stub.get_reranker_model = lambda: FakeReranker()
    monkeypatch.setitem(sys.modules, "app.services.retrieval", retrieval_stub)
    raw = json.dumps({"answer": "Có giới hạn.", "claims": [{"text": "Có giới hạn 8%.", "citation_ids": ["S1"]}]})
    answer = validate_model_answer(raw, [candidate()], date.today())
    assert claims_are_supported(answer, [candidate()]) is False

    item = candidate()
    answer = validate_model_answer(raw, [item], date.today())
    assert claims_are_supported(answer, [item]) is True
