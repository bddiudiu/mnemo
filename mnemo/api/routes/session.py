"""Session management endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from mnemo.models import SessionCreate, SessionResponse, SessionRecord
from mnemo.storage import relational_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["sessions"])


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(req: SessionCreate):
    """Create a new agent session."""
    record = SessionRecord(
        agent_id=req.agent_id,
        max_tokens=req.max_context_tokens,
    )
    session_id = await relational_store.create_session(record)
    stored = await relational_store.get_session(session_id)
    if not stored:
        raise HTTPException(status_code=500, detail="Session creation failed")
    return _to_response(stored)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session status."""
    record = await relational_store.get_session(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    return _to_response(record)


def _to_response(record) -> SessionResponse:
    return SessionResponse(
        id=record.id,
        agent_id=record.agent_id,
        status=record.status,
        summary=record.summary,
        current_tokens=record.current_tokens,
        max_tokens=record.max_tokens,
        created_at=record.created_at,
    )
