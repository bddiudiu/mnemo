"""Graph store implementation using NetworkX + SQLite.

Stores knowledge graph nodes and edges for semantic memory.
"""

import logging
from typing import Optional

from mnemo.storage.base import GraphStore

logger = logging.getLogger(__name__)


class NetworkXGraphStore(GraphStore):
    """In-memory graph store backed by NetworkX.

    Simple and dependency-free. For production, swap to Neo4j
    by implementing the GraphStore interface.
    """

    def __init__(self):
        self._graph = None
        self._persist_path: Optional[str] = None

    async def init(self) -> None:
        import networkx as nx
        self._graph = nx.DiGraph()
        logger.info("NetworkX graph store initialized")

    async def upsert_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict,
    ) -> None:
        if self._graph is None:
            await self.init()
        self._graph.add_node(entity_id, name=name, type=entity_type, **properties)

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[dict] = None,
    ) -> None:
        if self._graph is None:
            await self.init()
        self._graph.add_edge(source_id, target_id, type=relation_type, **(properties or {}))

    async def get_entity(self, entity_id: str) -> Optional[dict]:
        if self._graph is None:
            return None
        if entity_id in self._graph:
            data = dict(self._graph.nodes[entity_id])
            data["id"] = entity_id
            return data
        return None

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        if self._graph is None:
            return []
        results = []
        query_lower = query.lower()
        for node_id, attrs in self._graph.nodes(data=True):
            if query_lower in attrs.get("name", "").lower():
                if entity_type and attrs.get("type") != entity_type:
                    continue
                results.append({"id": node_id, **attrs})
                if len(results) >= limit:
                    break
        return results
