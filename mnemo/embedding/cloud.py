"""OpenAI embedding provider."""

import logging
from typing import Optional

from mnemo.config import config

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """Generate embeddings via OpenAI API."""

    def __init__(self):
        self.api_key = config.openai_api_key
        self.model = config.embedding_model

    async def embed(self, text: str) -> Optional[list[float]]:
        """Generate embedding for a single text."""
        import httpx

        if not self.api_key:
            logger.warning("No OpenAI API key set")
            return None

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": text,
                        "model": self.model,
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["data"][0]["embedding"]
            except Exception as e:
                logger.error("OpenAI embedding failed: %s", e)
                return None
