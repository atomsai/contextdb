"""Research assistant that accumulates notes and summarizes clusters.

Demonstrates multi-graph mode: enable_multi_graph=True wires up the
temporal + causal graphs on top of semantic, so queries with temporal or
causal cues rank neighboring memories higher.
"""

from __future__ import annotations

import asyncio

from contextdb import ContextDBConfig, init


async def main() -> None:
    config = ContextDBConfig(
        storage_url="sqlite:///research.db",
        embedding_model="mock",
        embedding_dim=64,
        llm_model="mock",
        llm_api_key="mock",
        enable_multi_graph=True,
    )
    db = init(user_id="researcher-1", config=config)
    async with db:
        await db.add("MAGMA reports a 45.5% boost on multi-hop reasoning.")
        await db.add("Memory-R1 hits 48% F1 with only 152 examples.")
        await db.add(
            "HippoRAG's personalized PageRank gives a ~20% recall improvement."
        )

        hits = await db.search("what boosts agent reasoning accuracy?")
        print("Findings:")
        for m in hits:
            print(f"  - {m.content}")


if __name__ == "__main__":
    asyncio.run(main())
