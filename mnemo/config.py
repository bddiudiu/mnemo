"""Global configuration for memori.

Environment variables override defaults:
  MNEMO_STORAGE_BACKEND=chroma|lance  (vector store)
  MNEMO_DB_URL=sqlite:///memori.db    (relational store)
  MNEMO_EMBEDDING_PROVIDER=openai|local
  MNEMO_OPENAI_API_KEY=sk-...
  MNEMO_PORT=8080
  MNEMO_MAX_CONTEXT_TOKENS=65536
  MNEMO_LOG_LEVEL=INFO
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Storage
    vector_backend: str = os.getenv("MNEMO_STORAGE_BACKEND", "chroma")
    db_url: str = os.getenv("MNEMO_DB_URL", "sqlite+aiosqlite:///memori.db")
    chroma_persist_dir: str = os.getenv("MNEMO_CHROMA_DIR", "./data/chroma")

    # Embedding
    embedding_provider: str = os.getenv("MNEMO_EMBEDDING_PROVIDER", "openai")
    openai_api_key: str = os.getenv("MNEMO_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = os.getenv("MNEMO_EMBEDDING_MODEL", "text-embedding-3-small")

    # Server
    port: int = int(os.getenv("MNEMO_PORT", "8080"))
    log_level: str = os.getenv("MNEMO_LOG_LEVEL", "INFO")

    # Memory limits
    max_context_tokens: int = int(os.getenv("MNEMO_MAX_CONTEXT_TOKENS", "65536"))
    compress_threshold: float = 0.8  # trigger auto-compress at 80% capacity
    default_top_k: int = 10          # default recall count
    max_episodic_per_session: int = 1000

    # TTL defaults (seconds)
    working_memory_ttl: int = 3600          # 1 hour
    episodic_memory_ttl: int = 86400 * 30   # 30 days
    semantic_memory_ttl: int = 86400 * 365  # 1 year


config = Config()
