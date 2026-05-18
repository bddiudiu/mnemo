"""End-to-end integration tests for memori.

These tests run against a real in-memory SQLite + ephemeral Chroma instance.
"""

import os
import pytest
from httpx import AsyncClient, ASGITransport

from mnemo.api import app
from mnemo.storage import relational_store, get_vector_store


# Ensure ephemeral storage for tests
os.environ["MEMORI_DB_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["MEMORI_CHROMA_DIR"] = "/tmp/memori_test_chroma"


@pytest.fixture
async def client():
    """Yield an async HTTP client against the FastAPI app."""
    await relational_store.init()
    vs = get_vector_store()
    await vs.init()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_store_and_get_memory(client):
    # Store
    payload = {
        "agent_id": "test-agent",
        "content": "User prefers dark mode UI",
        "memory_type": "semantic",
        "confidence": 0.95,
    }
    resp = await client.post("/api/v1/memories", json=payload)
    assert resp.status_code == 201
    mem = resp.json()
    assert mem["content"] == "User prefers dark mode UI"
    assert mem["memory_type"] == "semantic"
    mem_id = mem["id"]

    # Get
    resp = await client.get(f"/api/v1/memories/{mem_id}")
    assert resp.status_code == 200
    assert resp.json()["content"] == payload["content"]


@pytest.mark.asyncio
async def test_search_memories(client):
    # Seed
    for content in [
        "Production database is PostgreSQL 16",
        "Staging uses SQLite for testing",
        "User email is user@example.com",
    ]:
        await client.post(
            "/api/v1/memories",
            json={
                "agent_id": "test-agent",
                "content": content,
                "memory_type": "episodic",
            },
        )

    # Search
    resp = await client.post(
        "/api/v1/memories/search",
        json={
            "agent_id": "test-agent",
            "query": "PostgreSQL",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert any("PostgreSQL" in m["content"] for m in data["memories"])


@pytest.mark.asyncio
async def test_list_memories(client):
    # Seed
    for i in range(3):
        await client.post(
            "/api/v1/memories",
            json={
                "agent_id": "list-agent",
                "content": f"Memory {i+1}",
                "memory_type": "working",
            },
        )

    resp = await client.get("/api/v1/memories?agent_id=list-agent&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 3
    assert all(m["agent_id"] == "list-agent" for m in data["memories"])
    assert len(data["memories"]) == 3


@pytest.mark.asyncio
async def test_delete_memory(client):
    resp = await client.post(
        "/api/v1/memories",
        json={
            "agent_id": "test-agent",
            "content": "To be deleted",
            "memory_type": "episodic",
        },
    )
    mem_id = resp.json()["id"]

    resp = await client.delete(f"/api/v1/memories/{mem_id}")
    assert resp.status_code == 204

    resp = await client.get(f"/api/v1/memories/{mem_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_session_create_and_get(client):
    resp = await client.post(
        "/api/v1/sessions",
        json={
            "agent_id": "test-agent",
            "max_context_tokens": 4096,
        },
    )
    assert resp.status_code == 201
    session = resp.json()
    assert session["agent_id"] == "test-agent"
    assert session["max_tokens"] == 4096
    session_id = session["id"]

    resp = await client.get(f"/api/v1/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == session_id
