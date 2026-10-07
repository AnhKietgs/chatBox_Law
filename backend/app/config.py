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
    # Low non-zero sampling avoids brittle repetitions while keeping legal
    # answers, JSON schemas, and retrieval rewrites highly deterministic.
    llm_temperature: float = Field(default=0.1, ge=0, le=1)
    public_rate_limit_per_minute: int = 20
    admin_login_rate_limit_per_minute: int = 8
    rate_limit_redis_enabled: bool = True
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
    # 0.20: loại bỏ các case không có văn bản đúng chủ đề (VD: hỏi Luật LĐ
    # nhưng chỉ tìm được BLDS định nghĩa thời hạn, score ~0.13).
    # Các case đúng thường ≥ 0.40; ngưỡng 0.20 là buffer an toàn.
    reranker_minimum_absolute_score: float = 0.20
    # Web search is a strictly secondary path. Hybrid RAG always runs first;
    # only an empty/low-score rerank result can activate this adapter.
    web_search_fallback_enabled: bool = True
    web_search_rag_threshold: float = Field(default=0.20, ge=0, le=1)
    # Provider-agnostic HTTP gateway. Configure these in .env after choosing a
    # search provider/tool; see services/web_search.py for the request contract.
    web_search_api_url: str = ""
    web_search_api_key: str = ""
    web_search_provider: str = "generic"
    web_search_timeout_seconds: float = Field(default=15, gt=0, le=60)
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    claim_support_threshold: float = 0.65
    extractive_fallback_threshold: float = 0.50
    index_batch_size: int = 32
    # Legal structure-aware chunks are normally one Điều/Khoản/Điểm. Only an
    # unusually long unit is split into numbered sub-chunks.
    chunk_max_tokens: int = Field(default=450, ge=100, le=2000)
    chunk_split_overlap_tokens: int = Field(default=40, ge=0, le=200)
    chunk_audit_sample_size: int = Field(default=5, ge=0, le=20)
    # Development enables readable top-k traces by default. Production stays
    # quiet unless explicitly overridden with RETRIEVAL_DEBUG_LOGS=true.
    app_env: str = "development"
    retrieval_debug_logs: bool | None = None
    retrieval_debug_top_k: int = 8
    conversation_context_messages: int = 6
    conversation_context_max_chars: int = 3000
    contextualize_retrieval_with_llm: bool = True
    contextualize_timeout_seconds: float = 30
    # Ignore old conversation context for a clearly different legal domain.
    # This prevents a contextualizer from attaching a new topic to prior turns.
    intent_shift_detection_enabled: bool = True
    retrieval_augmentation_enabled: bool = True
    retrieval_augmentation_model: str = ""
    retrieval_augmentation_timeout_seconds: float = 30
    # A question that explicitly combines separate legal branches should be
    # clarified before HyDE/MultiQuery amplifies only one of them.
    mixed_domain_clarification_enabled: bool = True
    warm_models_on_startup: bool = True
    # ------------------------------------------------------------------ #
    # Context window overflow prevention                                   #
    # ------------------------------------------------------------------ #
    # Token budget uses max(2 tokens/lexical item, len/2.5 chars): a
    # conservative Vietnamese BPE bound which prefers early MapReduce.
    context_model_limit: int = 8192       # total token limit of the model
    context_budget_safety_margin: float = Field(default=0.85, gt=0, le=1)
    context_reserved_output: int = 1024   # tokens set aside for generated answer
    context_overhead: int = 256           # JSON wrappers, system markers
    # Retrieval pool vs. injection limit
    retrieval_recall_pool: int = 30       # candidates retrieved for reranking
    retrieval_inject_limit: int = 7       # top provisions injected into prompt
    # Map-Reduce is the normal overflow path. It preserves source IDs in a
    # compact evidence digest so the final answer can still cite many sources.
    # Set MAP_REDUCE_ENABLED=false only for constrained/offline environments.
    map_reduce_enabled: bool = True
    map_batch_token_limit: int = 2000     # max tokens per map batch
    map_reduce_timeout_seconds: float = 60.0
    # Sliding window history + memory summary
    history_window_turns: int = 5         # recent turns kept verbatim
    memory_summary_enabled: bool = True   # summarise older turns via LLM
    memory_summary_max_tokens: int = 150  # target length of the summary paragraph

    @property
    def retrieval_debug_enabled(self) -> bool:
        if self.retrieval_debug_logs is not None:
            return self.retrieval_debug_logs
        return self.app_env.strip().lower() in {"development", "dev", "local", "test"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
