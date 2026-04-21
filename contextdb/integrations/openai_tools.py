"""OpenAI function-calling tool definitions for ContextDB.

Returns a list of JSON Schema tool specs that can be passed directly into
``openai.chat.completions.create(..., tools=[...])``. Each tool name maps to
a bound async callable returned by :func:`make_tool_handlers`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextdb.client import ContextDB


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI tools JSON schema for ContextDB memory operations."""
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_add",
                "description": "Store a new fact, observation, or experience in long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["FACTUAL", "EXPERIENTIAL", "WORKING"],
                        },
                        "entity_mentions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search long-term memory semantically and temporally.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_get_entity",
                "description": "Retrieve the profile and associated memories for a named entity.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_forget",
                "description": "Bulk-delete memories by entity or age.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                        "older_than_days": {"type": "integer"},
                    },
                },
            },
        },
    ]


ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


def make_tool_handlers(client: ContextDB) -> dict[str, ToolHandler]:
    """Bind tool names to async callables executing against ``client``."""
    from datetime import timedelta

    from contextdb.core.models import MemoryType

    async def _add(args: dict[str, Any]) -> dict[str, Any]:
        mt_value = args.get("memory_type", "FACTUAL")
        mt = MemoryType(mt_value if isinstance(mt_value, str) else "FACTUAL")
        item = await client.add(
            content=args["content"],
            memory_type=mt,
            entity_mentions=args.get("entity_mentions"),
        )
        return {"id": item.id}

    async def _search(args: dict[str, Any]) -> list[dict[str, Any]]:
        hits = await client.search(args["query"], top_k=int(args.get("top_k", 5)))
        return [{"id": m.id, "content": m.content} for m in hits]

    async def _get_entity(args: dict[str, Any]) -> dict[str, Any]:
        return await client.get_entity(args["name"])

    async def _forget(args: dict[str, Any]) -> dict[str, int]:
        older_than: timedelta | None = None
        if "older_than_days" in args:
            older_than = timedelta(days=int(args["older_than_days"]))
        deleted = await client.forget(entity=args.get("entity"), older_than=older_than)
        return {"deleted": deleted}

    return {
        "memory_add": _add,
        "memory_search": _search,
        "memory_get_entity": _get_entity,
        "memory_forget": _forget,
    }
