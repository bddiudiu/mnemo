"""Episodic Memory: vectorized historical session events.

Stores and retrieves past conversation turns as vector embeddings.
Uses ChromaDB (default) or LanceDB for similarity search.
"""

from __future__ import annotations

import logging
from typing import Optional

from mnemo.config import config
from mnemo.models import MemoryRecord
from mnemo.storage.base import RelationalStore, VectorStore

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """Vector-searchable episodic memory store.

    Responsibilities:
    - Index memory content as embeddings in the vector store
    - Recall semantically similar past memories
    - Sync deletions between relational and vector stores
    """

    def __init__(self, store: RelationalStore, vector_store: VectorStore):
        self.store = store
        self.vector_store = vector_store

    async def init(self) -> None:
        """Initialize the vector store."""
        await self.vector_store.init()

    async def index_memory(self, memory_id: str, content: str) -> None:
        """Generate embedding for content and store in vector DB."""
        embedding = await self._embed(content)
        if embedding:
            await self.vector_store.insert(
                memory_id=memory_id,
                embedding=embedding,
                metadata={"content": content[:512]},
            )

    async def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        min_confidence: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Recall episodic memories by semantic similarity."""
        query_embedding = await self._embed(query)
        if not query_embedding:
            return []

        # Search vector store
        hits = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Fetch full records from relational store
        results = []
        for mem_id, score, _meta in hits:
            if score < min_confidence:
                continue
            record = await self.store.get_memory(mem_id)
            if record and record.agent_id == agent_id:
                results.append((record, score))

        return results

    async def _embed(self, text: str) -> Optional[list[float]]:
        """Generate embedding for text using configured provider.

        Returns a list of floats or None if embedding fails.
        MVP: uses a simple hash-based placeholder. Replace with
        real OpenAI/local embedding in production.
        """
        provider = config.embedding_provider
        if provider == "openai" and config.openai_api_key:
            from mnemo.embedding.cloud import OpenAIEmbedder

            embedder = OpenAIEmbedder()
            return await embedder.embed(text)
        elif provider == "local":
            from mnemo.embedding.local import LocalEmbedder

            embedder = LocalEmbedder()
            return await embedder.embed(text)
        else:
            # Fallback: hash-based placeholder for testing
            # NOT for production — returns deterministic but useless vectors
            import hashlib

            h = hashlib.sha256(text.encode()).digest()
            # Normalize to 384-dim (text-embedding-3-small size)
            vec = [float(b) / 255.0 for b in h] * (384 // 32 + 1)
            return vec[:384]
