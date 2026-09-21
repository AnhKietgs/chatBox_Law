import sys
from datetime import date
from types import ModuleType, SimpleNamespace
from uuid import uuid4

# The lightweight local test environment intentionally does not install the
# Qdrant client.  Retrieval scoring is unit-tested without opening Qdrant.
qdrant_stub = ModuleType("qdrant_client")
qdrant_stub.QdrantClient = object
qdrant_stub.models = SimpleNamespace()
sys.modules.setdefault("qdrant_client", qdrant_stub)

from app.models import LegalDocument, LegalProvision, LegalVersion, VersionStatus
from app.services import retrieval


def candidate(article: str, content: str):
    document = LegalDocument(id=uuid4(), code="LTM-2005", title="Luật Thương mại 2005")
    version = LegalVersion(
        id=uuid4(), document_id=document.id, version_label="Gốc",
        official_url="https://vbpl.vn/example", effective_from=date(2006, 1, 1),
        status=VersionStatus.published, source_object_key="source.pdf",
    )
    provision = LegalProvision(
        id=uuid4(), version_id=version.id, article_no=article, heading="Mức phạt vi phạm",
        content=content, ordinal=1,
    )
    return retrieval.RetrievedProvision(provision, version, document, 0.1)


def test_reranker_drops_results_below_normalized_confidence_threshold(monkeypatch):
    class FakeReranker:
        def compute_score(self, pairs, normalize):
            assert normalize is True
            assert "Điều 301" in pairs[0][1]
            return [0.92, 0.12]

    monkeypatch.setattr(retrieval, "get_reranker_model", lambda: FakeReranker())
    monkeypatch.setattr(retrieval, "get_settings", lambda: SimpleNamespace(confidence_threshold=0.55))

    result = retrieval.Reranker().rerank(
        "Mức phạt vi phạm là bao nhiêu?",
        [candidate("301", "Không quá 8%."), candidate("66", "Chuyển khẩu hàng hóa.")],
    )

    assert len(result) == 1
    assert result[0].provision.article_no == "301"
    assert result[0].score == 0.92
