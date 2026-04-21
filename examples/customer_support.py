"""Customer support agent that remembers prior tickets and preferences.

Run with::

    OPENAI_API_KEY=... python examples/customer_support.py

For a zero-network demo, the script defaults to the mock embedding + LLM
providers so it completes end-to-end without credentials.
"""

from __future__ import annotations

import asyncio

from contextdb import ContextDBConfig, init


async def main() -> None:
    config = ContextDBConfig(
        storage_url="sqlite:///support.db",
        embedding_model="mock",
        embedding_dim=64,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=False,
    )
    db = init(user_id="acct-42", config=config)
    async with db:
        await db.add(
            "Customer prefers email over phone, and only Tuesdays 9am-11am PT.",
            source="onboarding",
        )
        await db.add(
            "Ticket #1024: shipping delay — resolved by overnighting replacement unit.",
            source="ticket-history",
        )
        await db.add(
            "Ticket #1037: API integration failed because of expired API key.",
            source="ticket-history",
        )

        hits = await db.search("how do I contact this customer?")
        print("Relevant prior notes:")
        for m in hits:
            print(f"  - {m.content}")

        hits = await db.search("has this account had integration issues?")
        print("\nIntegration history:")
        for m in hits:
            print(f"  - {m.content}")


if __name__ == "__main__":
    asyncio.run(main())
