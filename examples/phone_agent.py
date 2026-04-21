"""Voice-style conversational agent using working memory for the turn buffer.

The :class:`WorkingMemory` surface evicts old turns when the session exceeds
the configured token budget, so the agent's context window never grows
unboundedly.
"""

from __future__ import annotations

import asyncio

from contextdb import ContextDBConfig, init


async def main() -> None:
    config = ContextDBConfig(
        storage_url="sqlite:///phone.db",
        embedding_model="mock",
        embedding_dim=32,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=False,
    )
    db = init(config=config)
    async with db:
        session = db.working(session_id="call-3821", max_tokens=120)
        await session.push("User: Hi, calling about order #7788.")
        await session.push("Agent: Let me look that up — I see it shipped yesterday.")
        await session.push("User: Great, when will it arrive?")
        await session.push("Agent: Tracking shows Friday delivery.")

        print("Current window:")
        print(await session.context_window())

        # Push more turns than the budget allows — oldest turns get evicted.
        for i in range(5):
            await session.push(
                f"User: follow-up {i} with some extra filler text to exceed the cap."
            )
        print("\nAfter eviction:")
        print(await session.context_window())


if __name__ == "__main__":
    asyncio.run(main())
