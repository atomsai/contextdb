"""LangChain memory adapter.

Exposes :class:`ContextDBMemory`, a small class that conforms to LangChain's
conversational memory interface (``load_memory_variables`` /
``save_context``) backed by :class:`~contextdb.client.ContextDB`. We do
**not** import ``langchain`` here — keeping the dependency optional — and
instead structure the adapter so that LangChain's duck-typed interface is
met by method signatures alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextdb.client import ContextDB


class ContextDBMemory:
    """LangChain-compatible memory backed by ContextDB.

    Usage:
        >>> from contextdb import init
        >>> db = init()
        >>> memory = ContextDBMemory(db, session_id="chat-1")
        >>> chain = ConversationChain(llm=llm, memory=memory)
    """

    memory_key: str = "history"

    def __init__(
        self,
        client: ContextDB,
        session_id: str,
        max_tokens: int = 2000,
        top_k: int = 5,
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.max_tokens = max_tokens
        self.top_k = top_k

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = str(inputs.get("input", "")) or str(next(iter(inputs.values()), ""))
        hits = await self.client.search(query, top_k=self.top_k) if query else []
        working = self.client.working(self.session_id, max_tokens=self.max_tokens)
        window = await working.context_window()
        history = "\n".join(m.content for m in hits)
        return {self.memory_key: f"{history}\n{window}".strip()}

    async def asave_context(
        self, inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> None:
        working = self.client.working(self.session_id, max_tokens=self.max_tokens)
        user_text = str(inputs.get("input", "")).strip()
        bot_text = str(outputs.get("output", "")).strip()
        if user_text:
            await working.push(f"User: {user_text}")
        if bot_text:
            await working.push(f"Assistant: {bot_text}")

    async def aclear(self) -> None:
        working = self.client.working(self.session_id, max_tokens=self.max_tokens)
        await working.clear()
