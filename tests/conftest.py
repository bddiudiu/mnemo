"""Test configuration and shared fixtures."""

import pytest

from mnemo.storage import relational_store


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
async def cleanup_db():
    """Clear all tables before each integration test to ensure isolation."""
    from sqlalchemy import text

    try:
        async with relational_store.session_factory() as session:
            await session.execute(text("DELETE FROM memories"))
            await session.execute(text("DELETE FROM sessions"))
            await session.commit()
    except Exception:
        pass  # Tables may not exist yet
    yield
