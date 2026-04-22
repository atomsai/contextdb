"""LangChain memory adapter.

Exposes :class:`ContextDBMemory`, a small class that conforms to LangChain's
conversational memory interface (``load_memory_variables`` /
``save_context``) backed by :class:`~contextdb.client.ContextDB`. We do
**not** import ``langchain`` here — keeping the dependency optional — and
instead structure the adapter so that LangChain's duck-typed interface is
met by method signatures alone.

Both sync and async entry points are provided. LangChain code paths still
exist that call memories synchronously, so the sync wrappers bridge to the
async client via :func:`asyncio.run` / ``run_until_complete``. The async
methods (``aload_memory_variables`` / ``asave_context`` / ``aclear``) are
preferred whenever the caller is already inside an event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from contextdb.client import ContextDB

_T = TypeVar("_T")


def _run_sync(coro: Awaitable[_T]) -> _T:
    """Execute an awaitable from synchronous code.

    Uses ``asyncio.run`` when no loop is running. When called from inside an
    already-running loop (e.g. a Jupyter cell) we fall back to creating a
    dedicated loop on a worker thread — ``asyncio.run`` refuses to nest, and
    ``run_until_complete`` on a running loop deadlocks. Callers inside a
    loop should prefer the ``a*`` methods.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is None:
        return asyncio.run(coro)  # type: ignore[arg-type]

    import threading

    result: list[_T] = []
    error: list[BaseException] = []

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            result.append(loop.run_until_complete(coro))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


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

    # -- Async (preferred) ------------------------------------------------- #

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

    # -- Sync (LangChain legacy paths) ------------------------------------- #

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper over :meth:`aload_memory_variables`."""
        return _run_sync(self.aload_memory_variables(inputs))

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Synchronous wrapper over :meth:`asave_context`."""
        _run_sync(self.asave_context(inputs, outputs))

    def clear(self) -> None:
        """Synchronous wrapper over :meth:`aclear`."""
        _run_sync(self.aclear())
