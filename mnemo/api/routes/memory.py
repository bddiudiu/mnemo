"""Memory CRUD and recall endpoints."""

import logging
from typing import Optional

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from mnemo.models import (
    MemoryCreate,
    MemoryRecallRequest,
    MemorySearchRequest,
    MemoryResponse,
    MemoryListResponse,
    RecallResponse,
    RecallResult,
)
from mnemo.storage import relational_store, get_vector_store
from mnemo.core.episodic_memory import EpisodicMemory
from mnemo.core.working_memory import WorkingMemory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["memories"])

# Lazy-init engines
_episodic: Optional[EpisodicMemory] = None
_working: Optional[WorkingMemory] = None


async def _get_episodic() -> EpisodicMemory:
    global _episodic
    if _episodic is None:
        _episodic = EpisodicMemory(relational_store, get_vector_store())
        await _episodic.init()
    return _episodic


async def _get_working() -> WorkingMemory:
    global _working
    if _working is None:
        _working = WorkingMemory(relational_store)
        await _working.init()
    return _working


@router.post("/memories", response_model=MemoryResponse, status_code=201)
async def store_memory(req: MemoryCreate):
    """Store a new memory."""
    from mnemo.models import MemoryRecord

    # Route by memory type
    record = MemoryRecord(
        session_id=req.session_id,
        agent_id=req.agent_id,
        content=req.content,
        memory_type=req.memory_type.value,
        confidence=req.confidence,
        metadata_=req.metadata,
        ttl=req.ttl,
    )

    memory_id = await relational_store.store_memory(record)

    # If episodic, also index in vector store
    if req.memory_type.value == "episodic":
        episodic = await _get_episodic()
        await episodic.index_memory(memory_id, req.content)

    # If working, add to context window
    if req.memory_type.value == "working":
        working = await _get_working()
        await working.add_message(req.session_id, req.content)

    stored = await relational_store.get_memory(memory_id)
    if not stored:
        raise HTTPException(status_code=500, detail="Memory storage failed")

    return MemoryResponse(
        id=stored.id,
        session_id=stored.session_id,
        agent_id=stored.agent_id,
        content=stored.content,
        memory_type=stored.memory_type,
        confidence=stored.confidence,
        metadata=stored.metadata_,
        created_at=stored.created_at,
        last_accessed_at=stored.last_accessed_at,
        ttl=stored.ttl,
    )


@router.post("/memories/recall", response_model=RecallResponse)
async def recall_memories(req: MemoryRecallRequest):
    """Recall memories across all layers."""
    results: list[RecallResult] = []

    # Layer 1: Working memory (fast, recency-based)
    if req.memory_type is None or req.memory_type.value == "working":
        working = await _get_working()
        wm_results = await working.recall(
            req.agent_id,
            req.session_id,
            req.query,
            top_k=req.top_k,
        )
        for mem in wm_results:
            results.append(RecallResult(
                memory=_to_response(mem),
                score=0.8,  # working memory always scores high for recency
                recall_layer="working",
            ))

    # Layer 2: Episodic memory (vector similarity)
    if req.memory_type is None or req.memory_type.value == "episodic":
        episodic = await _get_episodic()
        em_results = await episodic.recall(
            req.agent_id,
            req.query,
            top_k=req.top_k,
            min_confidence=req.min_confidence,
        )
        for mem, score in em_results:
            results.append(RecallResult(
                memory=_to_response(mem),
                score=round(score, 4),
                recall_layer="episodic",
            ))

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)
    results = results[:req.top_k]

    return RecallResponse(
        results=results,
        query=req.query,
        total_found=len(results),
    )


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str):
    """Get a single memory by ID."""
    record = await relational_store.get_memory(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    await relational_store.touch_memory(memory_id)
    return _to_response(record)


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(memory_id: str):
    """Forget a memory."""
    deleted = await relational_store.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    # Also remove from vector store if present
    try:
        vs = get_vector_store()
        await vs.delete(memory_id)
    except Exception:
        pass  # Not in vector store, that's fine


@router.post("/memories/search", response_model=MemoryListResponse)
async def search_memories(req: MemorySearchRequest):
    """Full-text search across memories."""
    records = await relational_store.search_memories(
        agent_id=req.agent_id,
        query=req.query,
        memory_type=req.memory_type.value if req.memory_type else None,
        limit=req.limit,
    )
    return MemoryListResponse(
        memories=[_to_response(r) for r in records],
        total=len(records),
    )


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    agent_id: str = Query(default="default"),
    session_id: Optional[str] = Query(default=None),
    memory_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List memories for an agent with pagination."""
    records, total = await relational_store.list_memories(
        agent_id=agent_id,
        session_id=session_id,
        memory_type=memory_type,
        limit=limit,
        offset=offset,
    )
    return MemoryListResponse(
        memories=[_to_response(r) for r in records],
        total=total,
    )


def _to_response(record) -> MemoryResponse:
    """Convert a MemoryRecord ORM object to a MemoryResponse."""
    return MemoryResponse(
        id=record.id,
        session_id=record.session_id,
        agent_id=record.agent_id,
        content=record.content,
        memory_type=record.memory_type,
        confidence=record.confidence,
        metadata=getattr(record, 'metadata_', getattr(record, 'metadata', {})),
        created_at=record.created_at,
        last_accessed_at=record.last_accessed_at,
        ttl=record.ttl,
    )
