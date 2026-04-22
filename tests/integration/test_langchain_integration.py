"""Tests for the LangChain adapter — sync wrappers in particular."""

from __future__ import annotations

import pytest

from contextdb import ContextDB
from contextdb.integrations.langchain import ContextDBMemory


@pytest.mark.asyncio
async def test_langchain_async_roundtrip(client: ContextDB) -> None:
    async with client:
        memory = ContextDBMemory(client, session_id="chat-1", max_tokens=500)
        await memory.asave_context({"input": "my AC is leaking"}, {"output": "noted"})
        vars_ = await memory.aload_memory_variables({"input": "AC leak"})
        assert "AC" in vars_["history"] or "leak" in vars_["history"]


@pytest.mark.asyncio
async def test_langchain_sync_from_worker_thread(client: ContextDB) -> None:
    """sync methods must work when dispatched from a non-event-loop thread."""
    import asyncio

    async with client:
        memory = ContextDBMemory(client, session_id="chat-sync", max_tokens=500)

        def _sync_write_then_read() -> dict[str, str]:
            memory.save_context({"input": "customer Alex"}, {"output": "hi Alex"})
            return memory.load_memory_variables({"input": "who just messaged?"})

        vars_ = await asyncio.to_thread(_sync_write_then_read)
        assert "Alex" in vars_["history"]

        await asyncio.to_thread(memory.clear)
        cleared = await asyncio.to_thread(
            memory.load_memory_variables, {"input": "who?"}
        )
        assert "Alex" not in cleared["history"]
