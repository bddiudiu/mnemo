"""memori — Agent Memory Middleware.

Three-layer memory architecture for AI agents:
  - Working Memory: current context window with auto-compression
  - Episodic Memory: vectorized historical session events
  - Semantic Memory: entity extraction and knowledge graph
"""

__version__ = "0.1.0"
__all__ = ["MnemoClient"]

from mnemo.sdk.python.client import MnemoClient
