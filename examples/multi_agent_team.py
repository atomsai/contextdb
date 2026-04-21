"""Two agents sharing memories over a :class:`MemoryBus`.

This example shows a planner agent publishing a task to the bus, which a
researcher agent subscribes to and reacts by recording a reflection. The
bus is in-process; swap for Redis/NATS in production.
"""

from __future__ import annotations

import asyncio

from contextdb import ContextDBConfig, init


async def main() -> None:
    config = ContextDBConfig(
        storage_url="sqlite:///team.db",
        embedding_model="mock",
        embedding_dim=32,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=False,
    )
    db = init(config=config)
    async with db:
        bus = db.bus()
        notifications: list[str] = []

        async def researcher(payload: dict) -> None:
            notifications.append(payload["task"])
            await db.experiential.record_trajectory(
                action=f"research:{payload['task']}",
                outcome="started",
                success=True,
            )

        await bus.subscribe("tasks.new", researcher)
        await bus.publish("tasks.new", {"task": "Summarize 2026 RL-memory papers"})
        await bus.publish("tasks.new", {"task": "Benchmark retrieval latency"})

        print(f"Researcher picked up {len(notifications)} tasks:")
        for t in notifications:
            print(f"  - {t}")

        trajectories = await db.experiential.list_trajectories()
        print(f"\nPersisted {len(trajectories)} trajectory memories.")


if __name__ == "__main__":
    asyncio.run(main())
