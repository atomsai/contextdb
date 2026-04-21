"""RL-guided memory manager — decide ADD / UPDATE / DELETE / NOOP per write.

In v0.1 we ship the inference side of the RL manager: a policy prompt that
asks the LLM to pick one of four actions for a new piece of content given
the current top-20 candidate memories. This mirrors Memory-R1's action
space (Yu et al., 2024) and gives us a drop-in surface for swapping in a
trained model later without reshuffling the client.

Training (PPO over trajectories of ADD/UPDATE/DELETE/NOOP outcomes) lives
in a separate research harness — not shipped with the library.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextdb.core.models import MemoryItem
    from contextdb.utils.llm import LLMProvider


_POLICY_PROMPT = """You are a memory manager. Given new content and the top
candidate existing memories, choose exactly one action and return strict JSON.

Actions:
- "ADD"    — store the content as a new memory.
- "UPDATE" — merge into an existing memory; must include target_memory_id and merged content.
- "DELETE" — discard an existing memory superseded by this content; must include target_memory_id.
- "NOOP"   — the content is redundant; do nothing.

Schema:
{"action": "ADD|UPDATE|DELETE|NOOP",
 "target_memory_id": "string|null",
 "content": "string|null",
 "reasoning": "string"}

New content: "{content}"

Candidate memories (id :: content):
{candidates}
"""


def _safe_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                loaded = json.loads(text[start : end + 1])
                return loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


class RLMemoryManager:
    """LLM-driven inference-time policy over ADD/UPDATE/DELETE/NOOP."""

    def __init__(self, llm: LLMProvider, max_candidates: int = 10) -> None:
        self.llm = llm
        self.max_candidates = max_candidates

    async def decide(
        self,
        content: str,
        candidates: list[MemoryItem],
    ) -> dict[str, Any]:
        cand_snippets = "\n".join(
            f"- {m.id} :: {m.content[:200]}" for m in candidates[: self.max_candidates]
        ) or "(none)"
        prompt = _POLICY_PROMPT.replace("{content}", content).replace(
            "{candidates}", cand_snippets
        )
        response = await self.llm.generate(prompt, temperature=0.0, max_tokens=400)
        decision = _safe_json(response)
        action = str(decision.get("action", "ADD")).upper()
        if action not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
            action = "ADD"
        decision["action"] = action
        return decision
