"""Storage package.

Exports the active storage backends based on config.
"""

from mnemo.config import config
from mnemo.storage.relational_store import SQLiteStore

# Default: always use SQLite for relational
relational_store = SQLiteStore()

# Vector store: lazy-init based on config
_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        backend = config.vector_backend
        if backend == "lance":
            from mnemo.storage.vector_store import LanceDBStore
            _vector_store = LanceDBStore()
        else:
            from mnemo.storage.vector_store import ChromaStore
            _vector_store = ChromaStore()
    return _vector_store
