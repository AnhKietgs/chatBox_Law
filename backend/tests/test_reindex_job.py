from types import SimpleNamespace
from uuid import uuid4

from app.models import IngestionJob, JobStatus, LegalVersion, VersionStatus
from app.services import ingestion


class FakeSession:
    def __init__(self, job, version):
        self.job = job
        self.version = version
        self.commits = 0

    def get(self, model, _identifier):
        return self.job if model is IngestionJob else self.version if model is LegalVersion else None

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def test_reindex_job_reports_progress_and_succeeds(monkeypatch):
    version_id = uuid4()
    job = SimpleNamespace(version_id=version_id, status=JobStatus.queued, progress=0, message=None)
    version = SimpleNamespace(id=version_id, status=VersionStatus.published)
    db = FakeSession(job, version)
    records = [(uuid4(), f"Điều {index}", {}) for index in range(35)]
    upserted = []

    monkeypatch.setattr(ingestion, "_build_embedding_records", lambda _: records)
    monkeypatch.setattr(
        ingestion,
        "HybridVectorStore",
        lambda: SimpleNamespace(upsert_many=lambda batch: upserted.extend(batch)),
    )

    ingestion.reindex_version_job(db, uuid4())

    assert len(upserted) == len(records)
    assert job.status == JobStatus.succeeded
    assert job.progress == 100
    assert job.message == "Đã lập chỉ mục lại 35 đơn vị pháp lý"
    assert db.commits >= 3
