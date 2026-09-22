from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./lawrag.db"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "change-me-minio"
    minio_bucket: str = "legal-sources"
    jwt_secret: str = "development-only-secret"
    admin_email: str = "admin@example.com"
    admin_password: str = "change-me-now"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "Qwen3.5:2b"
    public_rate_limit_per_minute: int = 20
    confidence_threshold: float = 0.55
    claim_support_threshold: float = 0.65
    index_batch_size: int = 32
    retrieval_debug_logs: bool = False
    retrieval_debug_top_k: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
