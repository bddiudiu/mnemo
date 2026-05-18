"""SDK type definitions."""

from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Memory:
    id: str
    session_id: str
    agent_id: str
    content: str
    memory_type: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    ttl: Optional[int] = None


@dataclass
class Session:
    id: str
    agent_id: str
    status: str = "active"
    summary: Optional[str] = None
    current_tokens: int = 0
    max_tokens: int = 65536
    created_at: Optional[datetime] = None
