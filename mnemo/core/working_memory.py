"""Working Memory: current context window with auto-compression.

Manages the active context window for an agent session. When the
window exceeds the compress threshold (80% of max tokens), it
triggers automatic summarization of oldest messages.
"""

from __future__ import annotations

import logging
from typing import Optional

from mnemo.config import config
from mnemo.models import MemoryRecord
from mnemo.storage.base import RelationalStore

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Manages per-session context windows.

    Responsibilities:
    - Track message count and token usage for a session
    - Trigger auto-compress when threshold is exceeded
    - Fast keyword/recency-based recall of recent messages
    """

    def __init__(self, store: RelationalStore):
        self.store = store
        self.max_tokens = config.max_context_tokens
        self.compress_threshold = config.compress_threshold

    async def init(self) -> None:
        """No-op: uses relational_store which is initialized separately."""
        pass

    async def add_message(
        self,
        session_id: str,
        content: str,
        token_count: Optional[int] = None,
    ) -> None:
        """Add a message to the working memory context.

        Estimates tokens if not provided (chars/4 heuristic).
        Triggers auto-compress if threshold exceeded.
        """
        if token_count is None:
            token_count = len(content) // 4  # rough heuristic

        # Update session token count
        session = await self.store.get_session(session_id)
        if session:
            new_tokens = session.current_tokens + token_count
            await self.store.update_session(session_id, current_tokens=new_tokens)

            # Check compress threshold
            if new_tokens > self.max_tokens * self.compress_threshold:
                logger.info(
                    "Session %s at %d/%d tokens — triggering auto-compress",
                    session_id,
                    new_tokens,
                    self.max_tokens,
                )
                await self._auto_compress(session_id)

    async def recall(
        self,
        agent_id: str,
        session_id: Optional[str],
        query: str,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        """Recall from working memory — fast keyword + recency search."""
        records, _ = await self.store.list_memories(
            agent_id=agent_id,
            session_id=session_id,
            memory_type="working",
            limit=top_k,
        )
        # Simple keyword relevance filter
        query_lower = query.lower()
        scored = []
        for r in records:
            score = self._keyword_score(r.content, query_lower)
            if score > 0:
                scored.append(r)
        return scored[:top_k]

    async def _auto_compress(self, session_id: str) -> None:
        """Compress oldest working memory messages into a summary.

        1. Fetch oldest working memories for the session.
        2. Summarize them via Summarizer (LLM if available, else extractive).
        3. Store summary as an EPISODIC memory.
        4. Reduce working memory token count.
        """
        session = await self.store.get_session(session_id)
        if not session:
            return

        from mnemo.compression.summarizer import Summarizer

        # Fetch oldest working memories (simplified: fetch all working, take oldest half)
        records, _ = await self.store.list_memories(
            agent_id=session.agent_id,
            session_id=session_id,
            memory_type="working",
            limit=50,
        )
        if not records:
            return

        # Take oldest half for compression
        to_compress = records[: len(records) // 2 + 1]
        messages = [r.content for r in to_compress]

        summarizer = Summarizer()
        summary = await summarizer.compress(messages)

        # Store summary as episodic memory
        from mnemo.models import MemoryRecord

        summary_record = MemoryRecord(
            session_id=session_id,
            agent_id=session.agent_id,
            content=summary,
            memory_type="episodic",
            confidence=0.9,
            metadata_={"source": "auto_compress", "compressed_count": len(to_compress)},
        )
        await self.store.store_memory(summary_record)

        # Reduce token count
        new_tokens = max(0, session.current_tokens - sum(len(m) // 4 for m in messages))
        await self.store.update_session(
            session_id,
            current_tokens=new_tokens,
            summary=summary,
        )
        logger.info(
            "Session %s auto-compressed: %d messages → summary (tokens: %d → %d)",
            session_id,
            len(to_compress),
            session.current_tokens,
            new_tokens,
        )

    @staticmethod
    def _keyword_score(content: str, query: str) -> float:
        """Simple keyword overlap scoring."""
        content_lower = content.lower()
        if query in content_lower:
            return 1.0
        # Check individual words
        query_words = set(query.split())
        content_words = set(content_lower.split())
        overlap = query_words & content_words
        if overlap:
            return len(overlap) / len(query_words)
        return 0.0
