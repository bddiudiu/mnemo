"""Abstract storage backends for memori.

To add a new storage backend, subclass the appropriate base and
register it in storage/__init__.py.
"""

from abc import ABC, abstractmethod
from typing import Optional

from mnemo.models import MemoryRecord, SessionRecord


class RelationalStore(ABC):
    """Abstract interface for structured memory/session storage."""

    @abstractmethod
    async def init(self) -> None:
        """Initialize storage (create tables, indices, etc.)."""
        ...

    @abstractmethod
    async def store_memory(self, record: MemoryRecord) -> str:
        """Persist a memory record. Returns the memory ID."""
        ...

    @abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve a single memory by ID."""
        ...

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if deleted."""
        ...

    @abstractmethod
    async def search_memories(
        self,
        agent_id: str,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Full-text search across memories for an agent."""
        ...

    @abstractmethod
    async def list_memories(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MemoryRecord], int]:
        """List memories with pagination. Returns (items, total_count)."""
        ...

    @abstractmethod
    async def create_session(self, record: SessionRecord) -> str:
        """Create a new session. Returns session ID."""
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get session by ID."""
        ...

    @abstractmethod
    async def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session fields. Returns True if updated."""
        ...

    @abstractmethod
    async def touch_memory(self, memory_id: str) -> None:
        """Update last_accessed_at timestamp."""
        ...


class VectorStore(ABC):
    """Abstract interface for vector similarity storage (Chroma, LanceDB, etc.)."""

    @abstractmethod
    async def init(self) -> None:
        """Initialize the vector store."""
        ...

    @abstractmethod
    async def insert(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> str:
        """Insert a vector with metadata. Returns the vector store ID."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_dict: Optional[dict] = None,
    ) -> list[tuple[str, float, dict]]:
        """Search for similar vectors. Returns [(memory_id, score, metadata), ...]."""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a vector by memory ID."""
        ...


class GraphStore(ABC):
    """Abstract interface for semantic knowledge graphs."""

    @abstractmethod
    async def init(self) -> None:
        """Initialize the graph store."""
        ...

    @abstractmethod
    async def upsert_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict,
    ) -> None:
        """Create or update an entity node."""
        ...

    @abstractmethod
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[dict] = None,
    ) -> None:
        """Add a directed edge between two entities."""
        ...

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[dict]:
        """Get an entity node by ID."""
        ...

    @abstractmethod
    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search entities by name or type."""
        ...
