from .celery_app import celery
from .database import SessionLocal
from .services.ingestion import process_version


@celery.task(bind=True, name="lawrag.process_version")
def process_version_task(self, job_id: str, filename: str):
    db = SessionLocal()
    try:
        process_version(db, __import__("uuid").UUID(job_id), filename)
    finally:
        db.close()
