"""Entity extraction for semantic memory.

Two strategies:
1. Rule-based: keyword/regex patterns for common agent memory content.
2. LLM-based (optional): OpenAI API for rich entity extraction.

Rule-based is the default (zero deps, zero latency).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import httpx

from mnemo.config import config

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract structured entities from raw text for semantic memory.

    Default strategy: rule-based regex extraction.
    If an OpenAI API key is available, `extract_with_llm()` can be used
    for richer, context-aware extraction.
    """

    # Patterns for common agent memory statements
    _PATTERNS = [
        # Preference: User prefers/likes/wants...
        (
            r"(?:user|customer|client)\s+(?:prefers?|likes?|wants?|hates?|dislikes?)\s+(.+?)(?:\.|;|$)",
            "preference",
        ),
        # Important fact
        (r"(?:important|note|remember)\s*[:\-]?\s*(.+?)(?:\.|;|$)", "fact"),
        # Environment / tech stack
        (
            r"(?:production|staging|dev|local)\s+(?:database|db|server|api|env|environment)\s+(?:is|uses?)\s+(.+?)(?:\.|;|$)",
            "environment",
        ),
        # API key / credentials (masked)
        (
            r"(?:api[_\-]?key|token|credential|password)\s*[:=]\s*(.+?)(?:\.|;|$)",
            "credential",
        ),
        # Version numbers
        (r"(?:version|v)\s*[:=]?\s*(\d+(?:\.\d+)*)", "version"),
        # Persona / role
        (
            r"(?:user|they|he|she)\s+(?:is|works? as|role is)\s+(?:an?\s+)?(.+?)(?:\.|;|$)",
            "persona",
        ),
        # Constraint / requirement
        (r"(?:must|should|need|require)\s+(?:to\s+)?(.+?)(?:\.|;|$)", "constraint"),
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or config.openai_api_key

    def extract(self, text: str) -> list[dict]:
        """Rule-based entity extraction.

        Returns a list of entity dicts with keys: name, type, confidence.
        """
        entities = []
        text_lower = text.lower()

        for pattern, entity_type in self._PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                raw = match.group(1).strip()
                # Clean up
                raw = raw.rstrip(".,;:!?")
                if len(raw) < 2:
                    continue
                entities.append(
                    {
                        "name": raw,
                        "type": entity_type,
                        "confidence": 0.7,
                    }
                )

        # Deduplicate by name
        seen = set()
        unique = []
        for e in entities:
            key = (e["name"].lower(), e["type"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    async def extract_with_llm(self, text: str) -> list[dict]:
        """LLM-based entity extraction via OpenAI API.

        Returns richer entities with relationships. Falls back to rule-based.
        """
        if not self.api_key:
            return self.extract(text)

        prompt = (
            "Extract key entities and facts from the following text. "
            "Output a JSON list of objects with fields: name, type (preference/fact/"
            "environment/persona/constraint/other), and a short description.\n\n"
            f"Text: {text[:2000]}\n\n"
            "JSON:"
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                import json

                parsed = json.loads(content)
                return (
                    parsed.get("entities", parsed)
                    if isinstance(parsed, dict)
                    else parsed
                )
        except Exception as e:
            logger.warning(
                "LLM entity extraction failed: %s. Falling back to rule-based.", e
            )
            return self.extract(text)
