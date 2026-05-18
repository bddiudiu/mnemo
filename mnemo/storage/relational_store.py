"""SQLite relational store implementation.

Uses SQLAlchemy 2.0 async with aiosqlite. Development default.
For production, swap to PostgreSQL by changing MEMORI_DB_URL.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func, delete, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from mnemo.config import config
from mnemo.models import Base, MemoryRecord, SessionRecord
from mnemo.storage.base import RelationalStore

logger = logging.getLogger(__name__)


class SQLiteStore(RelationalStore):
    """Async SQLite-backed relational store."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or config.db_url
        self.engine = create_async_engine(self.db_url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init(self) -> None:
        """Create tables and indices."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("SQLite store initialized: %s", self.db_url)

    # ── Memory CRUD ────────────────────────────────────────────

    async def store_memory(self, record: MemoryRecord) -> str:
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
            return record.id

    async def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(MemoryRecord).where(MemoryRecord.id == memory_id)
            )
            return result.scalar_one_or_none()

    async def delete_memory(self, memory_id: str) -> bool:
        async with self.session_factory() as session:
            result = await session.execute(
                delete(MemoryRecord).where(MemoryRecord.id == memory_id)
            )
            await session.commit()
            return result.rowcount > 0

    async def search_memories(
        self,
        agent_id: str,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        async with self.session_factory() as session:
            stmt = select(MemoryRecord).where(
                MemoryRecord.agent_id == agent_id,
                MemoryRecord.content.ilike(f"%{query}%"),
            )
            if memory_type:
                stmt = stmt.where(MemoryRecord.memory_type == memory_type)
            stmt = stmt.order_by(MemoryRecord.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def list_memories(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MemoryRecord], int]:
        async with self.session_factory() as session:
            conditions = [MemoryRecord.agent_id == agent_id]
            if session_id:
                conditions.append(MemoryRecord.session_id == session_id)
            if memory_type:
                conditions.append(MemoryRecord.memory_type == memory_type)

            # Count
            count_stmt = select(func.count()).where(*conditions)
            total = (await session.execute(count_stmt)).scalar() or 0

            # Fetch
            stmt = (
                select(MemoryRecord)
                .where(*conditions)
                .order_by(MemoryRecord.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all()), total

    # ── Session CRUD ───────────────────────────────────────────

    async def create_session(self, record: SessionRecord) -> str:
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
            return record.id

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(SessionRecord).where(SessionRecord.id == session_id)
            )
            return result.scalar_one_or_none()

    async def update_session(self, session_id: str, **kwargs) -> bool:
        async with self.session_factory() as session:
            result = await session.execute(
                update(SessionRecord)
                .where(SessionRecord.id == session_id)
                .values(**kwargs)
            )
            await session.commit()
            return result.rowcount > 0

    # ── Utilities ──────────────────────────────────────────────

    async def touch_memory(self, memory_id: str) -> None:
        async with self.session_factory() as session:
            await session.execute(
                update(MemoryRecord)
                .where(MemoryRecord.id == memory_id)
                .values(last_accessed_at=datetime.now(timezone.utc))
            )
            await session.commit()
