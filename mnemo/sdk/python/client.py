"""Python SDK for memori.

Provides the MemoriClient — the main entry point for agents to
interact with the memory middleware. 5 lines of code to add
persistent memory to any agent.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from mnemo.models import MemoryType

logger = logging.getLogger(__name__)


class MnemoClient:
    """Client for the memori memory middleware.

    Usage:
        client = MemoriClient(agent_id="my-agent")
        client.store("User prefers Python 3.12", memory_type="semantic")
        results = client.recall("Python version")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        agent_id: str = "default",
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.api_key = api_key
        self._client: httpx.Optional[Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            headers["X-Agent-ID"] = self.agent_id
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    def store(
        self,
        content: str,
        memory_type: str = "episodic",
        session_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Store a new memory. Returns the memory ID."""
        payload = {
            "agent_id": self.agent_id,
            "content": content,
            "memory_type": memory_type,
            "confidence": confidence,
            "metadata": metadata or {},
        }
        if session_id:
            payload["session_id"] = session_id
        if ttl:
            payload["ttl"] = ttl

        resp = self.client.post("/api/v1/memories", json=payload)
        resp.raise_for_status()
        return resp.json()["id"]

    def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 10,
        min_confidence: float = 0.0,
        session_id: Optional[str] = None,
    ) -> list[dict]:
        """Recall relevant memories by semantic similarity."""
        payload = {
            "agent_id": self.agent_id,
            "query": query,
            "top_k": top_k,
            "min_confidence": min_confidence,
        }
        if memory_type:
            payload["memory_type"] = memory_type
        if session_id:
            payload["session_id"] = session_id

        resp = self.client.post("/api/v1/memories/recall", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    def get(self, memory_id: str) -> dict:
        """Get a single memory by ID."""
        resp = self.client.get(f"/api/v1/memories/{memory_id}")
        resp.raise_for_status()
        return resp.json()

    def forget(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if successful."""
        resp = self.client.delete(f"/api/v1/memories/{memory_id}")
        return resp.status_code == 204

    def search(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Full-text search across memories."""
        payload = {
            "agent_id": self.agent_id,
            "query": query,
            "limit": limit,
        }
        if memory_type:
            payload["memory_type"] = memory_type

        resp = self.client.post("/api/v1/memories/search", json=payload)
        resp.raise_for_status()
        return resp.json().get("memories", [])

    def list_memories(
        self,
        session_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List memories with pagination."""
        params = {
            "agent_id": self.agent_id,
            "limit": limit,
            "offset": offset,
        }
        if session_id:
            params["session_id"] = session_id
        if memory_type:
            params["memory_type"] = memory_type

        resp = self.client.get("/api/v1/memories", params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("memories", []), data.get("total", 0)
