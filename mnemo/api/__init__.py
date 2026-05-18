"""FastAPI application factory for memori."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from memori import __version__
from mnemo.api.middleware import RateLimitMiddleware, AuthMiddleware
from mnemo.api.routes import health, memory, session
from mnemo.storage import relational_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler."""
    logging.basicConfig(
        level=getattr(logging, os.getenv("MEMORI_LOG_LEVEL", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    await relational_store.init()
    logger.info("memori v%s started", __version__)
    yield
    logger.info("memori shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="memori",
        description="Agent Memory Middleware — Persistent memory for AI agents",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware (order matters: auth → rate-limit)
    api_key = os.getenv("MEMORI_API_KEY")  # None = no auth (local mode)
    app.add_middleware(AuthMiddleware, api_key=api_key)
    app.add_middleware(RateLimitMiddleware, rate=100, window=60)

    # Routes
    app.include_router(health.router)
    app.include_router(memory.router)
    app.include_router(session.router)

    return app


app = create_app()
