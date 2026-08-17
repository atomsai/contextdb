"""Write-time prompt-injection screening for memory content (Epic 5).

Memories are attacker-influenceable: anything a user (or a web page, or a
tool result) says can be stored and later recalled into a prompt. A stored
"Remember: ignore your instructions" that comes back inside a system
channel is an exploit, not a memory.

The screen is deliberately *high-precision*: it fires on instruction-shaped
imperatives aimed at the model, not on ordinary imperatives ("remind me to
call mom"). A flagged memory is not refused — it is stored with
``epistemic_source="third_party"``, ``confidence=0``, and
``injection_suspect=True``, so retrieval and integrations demote it instead
of obeying it. Screening runs on every write path (add, add_fast,
conversations, consolidation) and cannot be overridden by caller-supplied
trust fields — safety fields only ever move toward less trust.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS = re.compile(
    r"(?:"
    r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+|my\s+)*(?:previous\s+|prior\s+)?instructions|"
    r"forget\s+(?:all\s+|everything|your\s+\w*\s*(?:rules|instructions|training|guidelines))|"
    r"you\s+are\s+now\b|"
    r"\bsystem\s*:|"
    r"system\s+prompt|"
    r"new\s+instructions|"
    r"disregard\s+(?:all\s+)?(?:previous|prior|above|earlier)|"
    r"do\s+not\s+follow\s+(?:your\s+)?(?:rules|instructions|guidelines)|"
    r"override\s+(?:your\s+)?(?:instructions|rules|programming|safety)|"
    r"reveal\s+(?:your\s+)?(?:system\s+prompt|instructions|rules)|"
    r"act\s+as\s+(?:a\s+)?(?:different|new|another)\s+(?:ai|assistant|model)|"
    r"<\s*/?\s*(?:system|instruction|prompt)\s*>"
    r")",
    re.IGNORECASE,
)


def screen_injection(text: str) -> bool:
    """True when content is instruction-shaped enough to demote on write."""
    return bool(_INJECTION_PATTERNS.search(text))
