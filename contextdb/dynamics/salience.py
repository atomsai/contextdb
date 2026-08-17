"""Salience — recency x frequency x criticality (Epic 4).

Pure similarity search drowns critical-but-old facts under recent noise:
cosine has no notion that "severe peanut allergy" matters more than the
400th "nice weather today". The retrieval score therefore becomes

    RRF(semantic, temporal, entity, causal) x salience
    salience = decay(age) x log(1 + recurrence) x criticality_boost

Design notes:

* ``decay`` is exponential with a configurable half-life (default 90 days).
  Age is measured from ``event_time`` when present (when the fact was
  about) else ``created_at`` (when we stored it).
* ``recurrence`` = corroboration + accesses. The log keeps a fact heard
  100x only ~5x louder than one heard once — repetition should break ties,
  not dominate.
* ``criticality_boost`` is where domain risk enters: action-gating facts
  get 4x; health/safety, legal, and identity-critical content (or anything
  carrying PII annotations) gets 32x. The multiplier is deliberately large
  enough to beat a year of decay — a critical constraint must survive a
  recency flood, which is exactly what EVAL-4.1 proves.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from contextdb.core.models import MemoryItem
from contextdb.dynamics.trust import criticality_class

DEFAULT_HALF_LIFE_DAYS = 90.0
ACTION_RELEVANT_BOOST = 4.0
CRITICAL_CLASS_BOOST = 32.0


def age_in_days(item: MemoryItem, now: datetime) -> float:
    """Age of a memory from its valid-time basis (event_time, else created)."""
    basis = item.event_time or item.created_at
    if basis.tzinfo is None:
        basis = basis.replace(tzinfo=timezone.utc)
    return max(0.0, (now - basis).total_seconds() / 86400.0)


def decay(age_days: float, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Exponential recency decay in (0, 1]."""
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def recurrence(item: MemoryItem) -> float:
    """How often the fact was restated or read."""
    return float(item.corroboration_count + item.access_count)


def criticality_boost(item: MemoryItem) -> float:
    """Domain-risk multiplier: critical > action-gating > ordinary."""
    boost = 1.0
    if item.action_relevant:
        boost *= ACTION_RELEVANT_BOOST
    if criticality_class(item.content) == "critical" or item.pii_annotations:
        boost *= CRITICAL_CLASS_BOOST
    return boost


def salience(
    item: MemoryItem,
    now: datetime,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Full salience score: decay(age) x log(1 + recurrence) x criticality."""
    return (
        decay(age_in_days(item, now), half_life_days)
        * math.log1p(recurrence(item))
        * criticality_boost(item)
    )
