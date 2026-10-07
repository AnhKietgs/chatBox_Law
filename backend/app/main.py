from contextlib import asynccontextmanager
import asyncio
import logging
from fastapi import FastAPI
from sqlalchemy import inspect, text
from fastapi.middleware.cors import CORSMiddleware
from .database import Base, engine
from .api import admin, chat
from .config import get_settings
from .services.document_metadata import infer_legacy_domain

app_logger = logging.getLogger("app")
app_logger.setLevel(logging.INFO)
if not app_logger.handlers:
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    app_logger.addHandler(stdout_handler)
app_logger.propagate = False
logger = logging.getLogger(__name__)


def migrate_existing_schema() -> None:
    """Small idempotent migrations for local Docker volumes created before MVP changes."""
    inspector = inspect(engine)
    with engine.begin() as connection:
        if "legal_provisions" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("legal_provisions")}
            if "chunk_index" not in columns:
                connection.execute(text("ALTER TABLE legal_provisions ADD COLUMN chunk_index INTEGER NOT NULL DEFAULT 1"))
                logger.info("Schema migration: added legal_provisions.chunk_index")
        if "legal_documents" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("legal_documents")}
            if "domain" not in columns:
                connection.execute(text("ALTER TABLE legal_documents ADD COLUMN domain VARCHAR(50) NOT NULL DEFAULT 'general'"))
                logger.info("Schema migration: added legal_documents.domain")
            if "category" not in columns:
                connection.execute(text("ALTER TABLE legal_documents ADD COLUMN category VARCHAR(100) NOT NULL DEFAULT 'general'"))
                logger.info("Schema migration: added legal_documents.category")
            # Only initial known corpus gets an automatic domain. Other legacy
            # documents stay 'general' until an admin classifies them.
            rows = connection.execute(text("SELECT id, code, title, domain FROM legal_documents")).mappings().all()
            for row in rows:
                if row["domain"] == "general":
                    inferred = infer_legacy_domain(row["code"], row["title"])
                    if inferred != "general":
                        connection.execute(text("UPDATE legal_documents SET domain = :domain WHERE id = :id"), {"domain": inferred, "id": row["id"]})
        if "conversations" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("conversations")}
            if "memory_summary" not in columns:
                connection.execute(text("ALTER TABLE conversations ADD COLUMN memory_summary TEXT"))
                logger.info("Schema migration: added conversations.memory_summary")


async def warm_models() -> None:
    """Move cold-start work before the chat endpoint begins accepting traffic."""
    try:
        from .services.vector_store import get_embedder
        from .services.retrieval import get_reranker_model
        logger.info("Warming BGE-M3 embedding model")
        await asyncio.to_thread(get_embedder)
        logger.info("Warming bge-reranker-v2-m3 model")
        await asyncio.to_thread(get_reranker_model)
    except Exception as exc:
        # Do not prevent the admin UI from recovering a development deployment.
        # The first chat request still falls back safely if a model is unavailable.
        logger.warning("Local retrieval model warm-up failed: %s", exc)
    try:
        import httpx
        settings = get_settings()
        logger.info("Warming Ollama model %s", settings.ollama_model)
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "stream": False,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {"temperature": settings.llm_temperature, "num_predict": 1},
                },
            )
            response.raise_for_status()
    except Exception as exc:
        logger.warning("Ollama model warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    migrate_existing_schema()
    if get_settings().warm_models_on_startup:
        await warm_models()
    yield


app = FastAPI(title="Law RAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080", "http://localhost:5173"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/healthz")
def healthcheck():
    return {"status": "ok"}
