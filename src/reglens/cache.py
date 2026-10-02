"""In-memory answer cache with a time-to-live (cleared whenever documents are ingested)."""

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

V = TypeVar("V")


def cache_key(question: str, k: int) -> str:
    """Questions differing only by case or spacing share an entry."""
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    return f"{k}|{normalized}"


class TTLCache(Generic[V]):
    def __init__(
        self, ttl_s: float, max_items: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._ttl_s = ttl_s
        self._max_items = max_items
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> V | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            expires, value = entry
            if expires < self._clock():
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: V) -> None:
        if self._ttl_s <= 0 or self._max_items <= 0:
            return
        with self._lock:
            self._items[key] = (self._clock() + self._ttl_s, value)
            self._items.move_to_end(key)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
