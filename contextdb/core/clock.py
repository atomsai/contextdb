"""Injectable clock — time is a source of truth, not a side effect.

Every write, supersede, as-of query, decay calculation, and TTL check
must read the same clock. ``datetime.now(utc)`` in a multi-process or
replay setting makes ``as_of`` and ``valid_until`` disagree about what
"now" was. Tests freeze time; production uses wall-clock UTC.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """Wall-clock UTC. The production default."""
    return datetime.now(tz=timezone.utc)


class FrozenClock:
    """Deterministic clock for tests and replay.

    Advance with :meth:`advance` or assign :attr:`now` directly. Naive
    datetimes are rejected — timezone-unaware "now" is how temporal
    bugs hide.
    """

    def __init__(self, now: datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware datetime.")
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> datetime:
        """Advance by a :class:`~datetime.timedelta` constructed from ``kwargs``."""
        from datetime import timedelta

        self.now = self.now + timedelta(**kwargs)
        return self.now
