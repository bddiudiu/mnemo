"""API route modules.

Routes are registered directly in mnemo.api.create_app().
"""

from mnemo.api.routes import health, memory, session

__all__ = ["health", "memory", "session"]
