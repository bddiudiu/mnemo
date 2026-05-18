"""Local embedding provider using sentence-transformers.

Requires: pip install memori[local] or pip install sentence-transformers
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Generate embeddings locally with sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded local embedding model: %s", self.model_name)
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Run: pip install memori[local]"
                )
                return None
        return self._model

    async def embed(self, text: str) -> Optional[list[float]]:
        """Generate embedding locally."""
        if self.model is None:
            return None
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
