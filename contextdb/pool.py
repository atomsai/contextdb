"""Bounded pool of per-user clients.

Prefer a single unscoped ``init()`` and ``user_id=`` on each call. Use this
pool only when a host cannot change those call sites and still needs one
``ContextDB`` per user. Each client holds a store connection and an
in-process vector index — that is the cost the serviceagent team asked about.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from contextdb.client import ContextDB
from contextdb.core.config import ContextDBConfig
from contextdb.core.policy import TrustPolicy


class ContextDBPool:
    """LRU of scoped clients sharing one :class:`ContextDBConfig`."""

    def __init__(
        self,
        config: ContextDBConfig,
        *,
        max_clients: int = 256,
        trust_policy: TrustPolicy | None = None,
    ) -> None:
        if max_clients < 1:
            raise ValueError("max_clients must be >= 1")
        self.config = config
        self.max_clients = max_clients
        self.trust_policy = trust_policy
        self._clients: OrderedDict[str, ContextDB] = OrderedDict()

    def client(self, user_id: str, **kwargs: Any) -> ContextDB:
        """Return a scoped client, evicting the least-recently used if full.

        Evicted clients are not closed here — call :meth:`aclose` on shutdown,
        or ``await pool.drop(user_id)`` to close one. An in-use evicted client
        remains usable until the host drops its reference.
        """
        existing = self._clients.get(user_id)
        if existing is not None:
            self._clients.move_to_end(user_id)
            return existing
        created = ContextDB(
            self.config,
            user_id=user_id,
            trust_policy=self.trust_policy,
            **kwargs,
        )
        self._clients[user_id] = created
        if len(self._clients) > self.max_clients:
            self._clients.popitem(last=False)
        return created

    async def drop(self, user_id: str) -> None:
        client = self._clients.pop(user_id, None)
        if client is not None:
            await client.close()

    async def aclose(self) -> None:
        while self._clients:
            _uid, client = self._clients.popitem(last=True)
            await client.close()

    def __len__(self) -> int:
        return len(self._clients)
