"""Vector store implementations: Chroma (default) and LanceDB."""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from mnemo.config import config
from mnemo.storage.base import VectorStore

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    """Chroma-based vector store — embedded, no external server needed."""

    def __init__(self):
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None

    async def init(self) -> None:
        self.client = chromadb.PersistentClient(
            path=config.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="memori_memories",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaStore initialized at %s", config.chroma_persist_dir)

    async def insert(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> str:
        if not self.collection:
            raise RuntimeError("ChromaStore not initialized. Call init() first.")
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        return memory_id

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_dict: Optional[dict] = None,
    ) -> list[tuple[str, float, dict]]:
        if not self.collection:
            raise RuntimeError("ChromaStore not initialized. Call init() first.")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
        )
        hits = []
        for i, mem_id in enumerate(results["ids"][0]):
            score = 1.0 - results["distances"][0][i]  # cosine distance → similarity
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            hits.append((mem_id, max(0.0, score), meta))
        return hits

    async def delete(self, memory_id: str) -> bool:
        if not self.collection:
            return False
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False


class LanceDBStore(VectorStore):
    """LanceDB-based vector store — better for production (incremental writes).

    Placeholder — implement when lance-db dependency is added.
    """

    async def init(self) -> None:
        raise NotImplementedError("LanceDB store not yet implemented")

    async def insert(
        self, memory_id: str, embedding: list[float], metadata: dict
    ) -> str:
        raise NotImplementedError

    async def search(
        self, query_embedding: list[float], top_k: int = 10, filter_dict=None
    ):
        raise NotImplementedError

    async def delete(self, memory_id: str) -> bool:
        raise NotImplementedError
