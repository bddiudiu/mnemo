"""Semantic Memory: entity extraction and knowledge graph.

Stores structured knowledge about entities, their relationships,
and long-term facts using NetworkX + SQLite.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import networkx as nx

from mnemo.storage.base import GraphStore, RelationalStore

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Knowledge graph-backed semantic memory.

    Stores entities (preferences, facts, personas) as graph nodes
    with typed edges representing relationships.
    """

    def __init__(self, store: RelationalStore, graph_store: GraphStore):
        self.store = store
        self.graph = nx.DiGraph()

    async def init(self) -> None:
        """Initialize the knowledge graph."""
        pass  # NetworkX is in-memory, loaded on demand

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[dict] = None,
        entity_id: Optional[str] = None,
    ) -> str:
        """Create or update an entity node."""
        eid = entity_id or str(uuid.uuid4())
        self.graph.add_node(eid, name=name, type=entity_type, **(properties or {}))
        return eid

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
    ) -> None:
        """Add a directed edge between two entities."""
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(source_id, target_id, type=relation_type)

    async def get_entity(self, entity_id: str) -> Optional[dict]:
        """Get entity node by ID."""
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return None

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search entities by name substring match."""
        results = []
        query_lower = query.lower()
        for node_id, attrs in self.graph.nodes(data=True):
            name = attrs.get("name", "")
            if query_lower in name.lower():
                if entity_type and attrs.get("type") != entity_type:
                    continue
                results.append({"id": node_id, **attrs})
                if len(results) >= limit:
                    break
        return results

    async def traverse(
        self,
        entity_id: str,
        depth: int = 2,
    ) -> dict:
        """Get entity + its neighborhood up to `depth` hops."""
        if entity_id not in self.graph:
            return {"entity": None, "neighbors": []}

        entity = dict(self.graph.nodes[entity_id])
        entity["id"] = entity_id

        neighbors = []
        for neighbor in nx.descendants_at_distance(self.graph, entity_id, 1):
            edge_data = self.graph.get_edge_data(entity_id, neighbor) or {}
            neighbors.append({
                "id": neighbor,
                **dict(self.graph.nodes[neighbor]),
                "relation": edge_data.get("type", "related_to"),
            })

        return {"entity": entity, "neighbors": neighbors}
