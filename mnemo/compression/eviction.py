"""Memory eviction policies: LRU, LFU, TTL-based."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional


class EvictionPolicy(ABC):
    """Abstract eviction policy for memory management."""

    @abstractmethod
    def should_evict(
        self, access_count: int, last_access: float, ttl: Optional[int] = None
    ) -> bool:
        """Return True if the memory item should be evicted."""
        ...


class LRUPolicy(EvictionPolicy):
    """Least Recently Used: evict if not accessed within `max_age` seconds."""

    def __init__(self, max_age: int = 86400 * 7):  # 7 days default
        self.max_age = max_age

    def should_evict(
        self, access_count: int, last_access: float, ttl: Optional[int] = None
    ) -> bool:
        return (time.time() - last_access) > self.max_age


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used: evict if accessed fewer than `min_access` times."""

    def __init__(self, min_access: int = 2):
        self.min_access = min_access

    def should_evict(
        self, access_count: int, last_access: float, ttl: Optional[int] = None
    ) -> bool:
        return access_count < self.min_access


class TTLPolicy(EvictionPolicy):
    """Time-To-Live: evict if TTL has expired."""

    def should_evict(
        self, access_count: int, last_access: float, ttl: Optional[int] = None
    ) -> bool:
        if ttl is None:
            return False
        return (time.time() - last_access) > ttl
