from functools import lru_cache
from pydantic import Field
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
    # Kept only as the legacy fallback when Reranker.rerank() is called directly.
    # The production retrieval flow below uses a per-query percentile instead.
    confidence_threshold: float = 0.50
    conversation_confidence_threshold: float = 0.30
    # Dynamic reranker selection: retain candidates at or above this percentile
    # within the current recall batch. 70 = the top roughly 30% of that batch.
    rerank_score_percentile: float = Field(default=75, ge=0, le=100)
    conversation_rerank_score_percentile: float = Field(default=75, ge=0, le=100)
    # Absolute floor: nếu top reranker score < giá trị này → abstain ngay,
    # không để LLM thấy source không liên quan rồi sinh câu trả lời sai.
    reranker_minimum_absolute_score: float = 0.05
    claim_support_threshold: float = 0.65
    extractive_fallback_threshold: float = 0.50
    index_batch_size: int = 32
    retrieval_debug_logs: bool = False
    retrieval_debug_top_k: int = 8
    conversation_context_messages: int = 6
    conversation_context_max_chars: int = 3000
    contextualize_retrieval_with_llm: bool = True
    contextualize_timeout_seconds: float = 30
    retrieval_augmentation_enabled: bool = True
    retrieval_augmentation_model: str = ""
    retrieval_augmentation_timeout_seconds: float = 30
    warm_models_on_startup: bool = True
    # ------------------------------------------------------------------ #
    # Context window overflow prevention                                   #
    # ------------------------------------------------------------------ #
    # Token budget — estimate = ceil(len / 3.5), conservative for Vietnamese
    context_model_limit: int = 8192       # total token limit of the model
    context_reserved_output: int = 1024   # tokens set aside for generated answer
    context_overhead: int = 256           # JSON wrappers, system markers
    # Retrieval pool vs. injection limit
    retrieval_recall_pool: int = 30       # candidates retrieved for reranking
    retrieval_inject_limit: int = 5       # top provisions injected into prompt
    # Map-Reduce (default OFF; enable via MAP_REDUCE_ENABLED=true in .env)
    map_reduce_enabled: bool = False
    map_batch_token_limit: int = 2000     # max tokens per map batch
    map_reduce_timeout_seconds: float = 60.0
    # Sliding window history + memory summary
    history_window_turns: int = 5         # recent turns kept verbatim
    memory_summary_enabled: bool = True   # summarise older turns via LLM
    memory_summary_max_tokens: int = 150  # target length of the summary paragraph


@lru_cache
def get_settings() -> Settings:
    return Settings()
