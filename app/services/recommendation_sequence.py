"""Resumable, deterministic recommendation sequences.

A *sequence* is the complete ordered list of recommendations for one frozen
profile. A page is a slice of it. Nothing about a sequence lives only in this
process: the cursor a client holds carries the sequence's fingerprint and an
offset, and the request carries the inputs the sequence was built from, so any
instance can rebuild the same list and serve the same slice. That is what makes
paging survive a Cloud Run restart, a scale-to-zero, or a second instance.

The cache below is therefore an optimisation and nothing more. A hit avoids
re-ranking; a miss re-ranks and gets the identical answer.

Depth comes from *windows*, not from a larger response limit. Window 0 is
exactly what production already ranks: the ALS top-N, ordered by LambdaMART.
Window 1 is the next N titles by ALS score, ordered the same way, and so on.
Windows are concatenated in ALS order, so every title in an earlier window
precedes every title in a later one and nothing is shuffled.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import AppError

CURSOR_VERSION = 1
SEQUENCE_VERSION = "sequence-v1"
MAX_CURSOR_OFFSET = 100_000


def sequence_fingerprint(payload: Mapping[str, Any]) -> str:
    """Identity of a sequence: its inputs and everything that ranks them."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def encode_cursor(fingerprint: str, offset: int) -> str:
    """An opaque position in one sequence. It holds no user data and no secret."""
    raw = json.dumps({"v": CURSOR_VERSION, "f": fingerprint, "o": int(offset)}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("ascii")).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, int]:
    """Fingerprint and offset, or a 422 the client can act on."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        if payload.get("v") != CURSOR_VERSION:
            raise ValueError("unsupported cursor version")
        fingerprint = str(payload["f"])
        offset = int(payload["o"])
    except (ValueError, KeyError, TypeError, binascii.Error, UnicodeError) as exc:
        raise AppError("The recommendation cursor is not valid", code="invalid_cursor", status_code=422) from exc
    if not 0 <= offset <= MAX_CURSOR_OFFSET or len(fingerprint) != 32:
        raise AppError("The recommendation cursor is not valid", code="invalid_cursor", status_code=422)
    return fingerprint, offset


@dataclass
class RankedSequence:
    """The part of a sequence ranked so far, plus what the next window needs."""

    fingerprint: str
    # "fast" (ALS + LambdaMART windows) or "hybrid" (one starter list).
    source: str | None = None
    items: list[int] = field(default_factory=list)
    # Hybrid results carry scoring fields the catalog record does not have.
    payloads: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Every candidate retrieved so far, kept or not, so the next window starts
    # where the previous one ended.
    consumed: set[int] = field(default_factory=set)
    # None until the first window, so "no franchise seen yet" is distinguishable
    # from "the franchise filter is off".
    seen_series: set[str] | None = None
    windows: int = 0
    window_sizes: list[int] = field(default_factory=list)
    exhausted: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    touched: float = field(default_factory=time.monotonic)


class SequenceCache:
    """A small LRU of partially ranked sequences.

    Bounded in size and age. Losing an entry costs one re-rank, never
    correctness, so eviction needs no coordination with clients.
    """

    def __init__(self, size: int = 256, ttl_seconds: float = 1800.0):
        self.size = max(1, int(size))
        self.ttl_seconds = float(ttl_seconds)
        self._entries: OrderedDict[str, RankedSequence] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, fingerprint: str) -> tuple[RankedSequence, bool]:
        """The sequence for `fingerprint`, and whether it was already cached."""
        now = time.monotonic()
        with self._lock:
            existing = self._entries.get(fingerprint)
            if existing is not None and now - existing.touched <= self.ttl_seconds:
                existing.touched = now
                self._entries.move_to_end(fingerprint)
                return existing, True
            created = RankedSequence(fingerprint=fingerprint)
            self._entries[fingerprint] = created
            self._entries.move_to_end(fingerprint)
            while len(self._entries) > self.size:
                self._entries.popitem(last=False)
            return created, False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
