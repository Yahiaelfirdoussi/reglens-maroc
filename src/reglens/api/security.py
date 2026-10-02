"""API keys (user and admin roles) and a per-key token-bucket rate limit."""

import hashlib
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "admin"]


def key_id(key: str) -> str:
    """Short, non-reversible identifier for logs and rate-limit buckets (never log keys)."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


class KeyRing:
    """Admin keys can do everything; user keys can query. No keys configured = open (dev)."""

    def __init__(self, user_keys: Iterable[str], admin_keys: Iterable[str]) -> None:
        self._admin = {k for k in admin_keys if k}
        self._user = {k for k in user_keys if k} | self._admin

    @property
    def enabled(self) -> bool:
        return bool(self._user)

    def role(self, key: str | None) -> Role | None:
        if not self.enabled:
            return "admin"
        if not key:
            return None
        if key in self._admin:
            return "admin"
        return "user" if key in self._user else None


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Token bucket: ``per_minute`` requests, refilled continuously, per caller."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._capacity = float(per_minute)
        self._rate = per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, caller: str) -> tuple[bool, float]:
        """(allowed, seconds to wait before retrying)."""
        if self._capacity <= 0:
            return True, 0.0
        now = self._clock()
        with self._lock:
            bucket = self._buckets.setdefault(caller, _Bucket(self._capacity, now))
            bucket.tokens = min(self._capacity, bucket.tokens + (now - bucket.updated) * self._rate)
            bucket.updated = now
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return True, 0.0
            return False, (1 - bucket.tokens) / self._rate
