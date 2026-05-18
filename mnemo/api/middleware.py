"""API middleware: auth, rate limiting, request logging."""

import time
import logging
from collections import defaultdict
from typing import Optional

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("mnemo.api")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-agent token-bucket rate limiter.

    Default: 100 requests per 60 seconds per agent_id.
    In local/dev mode (no MEMORI_API_KEY set), rate limiting is relaxed.
    """

    def __init__(self, app, rate: int = 100, window: int = 60):
        super().__init__(app)
        self.rate = rate
        self.window = window
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health endpoint
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        agent_id = request.headers.get("X-Agent-ID", "anonymous")
        now = time.time()

        # Clean old entries
        bucket = self._buckets[agent_id]
        bucket[:] = [t for t in bucket if now - t < self.window]

        if len(bucket) >= self.rate:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        bucket.append(now)
        return await call_next(request)


class AuthMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication.

    If MEMORI_API_KEY is not set, auth is disabled (local/dev mode).
    If set, requests must include X-API-Key header.
    """

    def __init__(self, app, api_key: Optional[str] = None):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if self.api_key is None:
            return await call_next(request)

        # Skip auth for health endpoint
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        provided = request.headers.get("X-API-Key")
        if not provided or provided != self.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
