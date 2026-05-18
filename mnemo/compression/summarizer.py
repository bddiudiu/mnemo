"""Context compression: LLM-based summarizer and eviction policies."""

import logging
import os
from typing import Optional

import httpx

from mnemo.config import config

logger = logging.getLogger(__name__)


class Summarizer:
    """Compress conversation history into concise summaries.

    Uses OpenAI API (or any OpenAI-compatible endpoint) for
    abstractive summarization. Falls back to extractive summarization
    if no API key is available.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or config.openai_api_key
        self.model = model
        self.base_url = base_url

    async def compress(self, messages: list[str]) -> str:
        """Compress a list of messages into a single summary.

        If OpenAI API key is available, uses abstractive summarization.
        Otherwise falls back to extractive.
        """
        if not messages:
            return ""

        if self.api_key:
            try:
                return await self._llm_compress(messages)
            except Exception as e:
                logger.warning("LLM compression failed (%s), falling back to extractive", e)

        return self._extractive_summary(messages)

    async def _llm_compress(self, messages: list[str]) -> str:
        """Call OpenAI API for abstractive summarization."""
        text = "\n".join(f"- {m[:500]}" for m in messages)
        prompt = (
            "Summarize the following conversation history into a concise paragraph. "
            "Preserve key facts, user preferences, and decisions. Omit filler.\n\n"
            f"{text}\n\nSummary:"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _extractive_summary(messages: list[str]) -> str:
        """Simple extractive summary: keep first, last, and key sentences."""
        parts = []
        if messages:
            parts.append(messages[0][:200])
        if len(messages) > 2:
            parts.append(f"... ({len(messages) - 2} messages omitted) ...")
        if len(messages) > 1:
            parts.append(messages[-1][:200])
        return " | ".join(parts)
