"""Health check endpoint."""

from fastapi import APIRouter

from memori import __version__
from mnemo.config import config
from mnemo.models import HealthResponse

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=__version__,
        storage_backend=config.vector_backend,
        embedding_provider=config.embedding_provider,
    )
