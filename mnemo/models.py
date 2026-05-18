"""Core data models for memori.

All models are Pydantic v2 for FastAPI integration + SQLAlchemy
for persistence. The in-memory types use dataclasses for lightweight
access in the core engine.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    WORKING = "working"       # Current session context
    EPISODIC = "episodic"     # Historical session events
    SEMANTIC = "semantic"     # Entities and relationships


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


# ── Request / Response Models ──────────────────────────────────────

class MemoryCreate(BaseModel):
    """Request: store a new memory."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = "default"
    content: str = Field(..., min_length=1, max_length=32768)
    memory_type: MemoryType = MemoryType.EPISODIC
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    ttl: Optional[int] = None  # seconds


class MemoryRecallRequest(BaseModel):
    """Request: recall memories by query."""
    agent_id: str = "default"
    session_id: Optional[str] = None
    query: str = Field(..., min_length=1)
    memory_type: Optional[MemoryType] = None
    top_k: int = Field(default=10, ge=1, le=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MemorySearchRequest(BaseModel):
    """Request: full-text / keyword search."""
    agent_id: str = "default"
    query: str = Field(..., min_length=1)
    memory_type: Optional[MemoryType] = None
    limit: int = Field(default=20, ge=1, le=100)


class MemoryResponse(BaseModel):
    """Response: a single memory record."""
    id: str
    session_id: str
    agent_id: str
    content: str
    memory_type: MemoryType
    confidence: float
    metadata: dict[str, Any]
    created_at: datetime
    last_accessed_at: datetime
    ttl: Optional[int] = None

    model_config = {"from_attributes": True}


class MemoryListResponse(BaseModel):
    memories: list[MemoryResponse]
    total: int


class SessionCreate(BaseModel):
    """Request: create a new session."""
    agent_id: str = "default"
    max_context_tokens: int = 65536


class SessionResponse(BaseModel):
    id: str
    agent_id: str
    status: SessionStatus
    summary: Optional[str] = None
    current_tokens: int = 0
    max_tokens: int = 65536
    created_at: datetime

    model_config = {"from_attributes": True}


class RecallResult(BaseModel):
    """A single recall hit with relevance score."""
    memory: MemoryResponse
    score: float                    # 0-1 relevance
    recall_layer: str               # "working" | "episodic" | "semantic"


class RecallResponse(BaseModel):
    results: list[RecallResult]
    query: str
    total_found: int


class EntityResponse(BaseModel):
    id: str
    name: str
    entity_type: str
    confidence: float
    relationships: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    storage_backend: str
    embedding_provider: str


# ── SQLAlchemy ORM Models ──────────────────────────────────────────

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, JSON, Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class MemoryRecord(Base):
    __tablename__ = "memories"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True, nullable=False)
    agent_id = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    memory_type = Column(String(20), nullable=False, default="episodic")
    confidence = Column(Float, default=1.0)
    embedding_id = Column(String, nullable=True)  # FK to Chroma/LanceDB
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_accessed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ttl = Column(Integer, nullable=True)


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, index=True, nullable=False)
    status = Column(String(20), default="active")
    summary = Column(Text, nullable=True)
    current_tokens = Column(Integer, default=0)
    max_tokens = Column(Integer, default=65536)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
