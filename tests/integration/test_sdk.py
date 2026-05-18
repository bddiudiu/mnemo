"""Tests for the MemoriClient SDK.

Uses TestClient (synchronous) to test against the ASGI app.
"""

import pytest
from starlette.testclient import TestClient

from mnemo.api import app
from mnemo.sdk.python.client import MemoriClient


@pytest.fixture
def client():
    """Yield a MemoriClient wired to a TestClient."""
    http_client = TestClient(app)
    sdk = MemoriClient(base_url="http://test", agent_id="test-agent")
    sdk._client = http_client
    yield sdk


def test_store_and_get(client):
    """Test storing and retrieving a memory."""
    # Store
    mem_id = client.store("Test memory content", memory_type="episodic")
    assert mem_id is not None

    # Get
    mem = client.get(mem_id)
    assert mem["content"] == "Test memory content"
    assert mem["memory_type"] == "episodic"
    assert mem["agent_id"] == "test-agent"

    # Delete
    success = client.forget(mem_id)
    assert success is True

    # Verify deleted
    resp = client._client.get(f"/api/v1/memories/{mem_id}")
    assert resp.status_code == 404


def test_recall(client):
    """Test semantic recall (placeholder embedding, so results are basic)."""
    # Seed
    for content in [
        "User prefers Python 3.12",
        "Production database is PostgreSQL 16",
        "Staging uses SQLite",
    ]:
        client.store(content, memory_type="episodic")

    # Recall
    results = client.recall("database", top_k=5)
    assert isinstance(results, list)


def test_search(client):
    """Test full-text search."""
    client.store("Important: API key must not be logged", memory_type="semantic")
    client.store("Reminder: deploy on Fridays is risky", memory_type="semantic")

    results = client.search("API key", limit=10)
    assert any("API key" in r["content"] for r in results)


def test_list(client):
    """Test listing memories with pagination."""
    # Use a unique agent to avoid interference from other tests
    list_client = MemoriClient(base_url="http://test", agent_id="list-only-agent")
    list_client._client = client._client

    for i in range(5):
        list_client.store(f"Memory item {i+1}", memory_type="working")

    memories, total = list_client.list_memories(limit=10)
    assert total == 5
    assert len(memories) == 5
