from celery import Celery
from .config import get_settings

celery = Celery("lawrag", broker=get_settings().redis_url, backend=get_settings().redis_url, include=["app.tasks"])
celery.conf.update(task_track_started=True, task_serializer="json", result_serializer="json", accept_content=["json"])
