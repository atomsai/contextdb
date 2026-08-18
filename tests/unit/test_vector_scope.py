"""Vector candidate scoping and process-local index coherence.

Two invariants, verified on SQLite always and on Postgres when
``CONTEXTDB_TEST_POSTGRES_URL`` points at a test database:

* Candidates are scoped *before* ranking — a large foreign scope cannot
  starve an in-scope hit out of the candidate budget.
* The process-local index tracks the store revision — a write committed
  by another instance (or process) is reflected in the next search.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest

from contextdb.core.models import MemoryItem
from contextdb.store.base import BaseStore
from contextdb.store.factory import open_store
from contextdb.store.vector_index import NumpyIndex
from tests.pg_util import fresh_pg_database

_BACKENDS = ["sqlite", "postgres"]
_DIM = 4


@asynccontextmanager
async def _url(backend: str, tmp_path: Path) -> AsyncIterator[str]:
    if backend == "postgres":
        async with fresh_pg_database() as url:
            yield url
    else:
        yield f"sqlite:///{tmp_path}/{uuid.uuid4().hex}.db"


def _item(content: str, user_id: str, embedding: list[float]) -> MemoryItem:
    return MemoryItem(content=content, user_id=user_id, embedding=embedding)


def test_numpy_index_allowlist_scopes_before_ranking() -> None:
    idx = NumpyIndex(dimension=_DIM)
    foreign = [f"foreign-{i}" for i in range(20)]
    idx.add(
        foreign + ["mine"],
        np.asarray([[1.0, 0.0, 0.0, 0.0]] * 20 + [[0.9, 0.1, 0.0, 0.0]], dtype=np.float32),
    )
    # Without an allowlist the 20 perfectly-aligned foreign vectors would
    # take every candidate slot above top_k=5's over-fetch budget.
    hits = idx.search(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        top_k=5,
        include_ids={"mine"},
    )
    assert [h[0] for h in hits] == ["mine"]
    # An empty allowlist ranks nothing.
    assert (
        idx.search(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            top_k=5,
            include_ids=set(),
        )
        == []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", _BACKENDS)
async def test_store_scopes_vector_candidates_before_ranking(
    backend: str, tmp_path: Path
) -> None:
    """20 foreign vectors perfectly aligned with the query must not starve
    the one in-scope vector out of the candidate set."""
    async with _url(backend, tmp_path) as url:
        store = open_store(url, embedding_dim=_DIM)
        await store.initialize()
        try:
            foreign_user = f"foreign-{uuid.uuid4().hex}"
            my_user = f"mine-{uuid.uuid4().hex}"
            for i in range(20):
                await store.add(
                    _item(f"foreign note {i}", foreign_user, [1.0, 0.0, 0.0, 0.0])
                )
            mine = _item("my note", my_user, [0.9, 0.1, 0.0, 0.0])
            await store.add(mine)

            hits = await store.search_by_embedding(
                [1.0, 0.0, 0.0, 0.0], top_k=5, user_id=my_user
            )
            assert [h.id for h in hits] == [mine.id]

            foreign_hits = await store.search_by_embedding(
                [1.0, 0.0, 0.0, 0.0], top_k=5, user_id=foreign_user
            )
            assert len(foreign_hits) == 5
            assert all(h.user_id == foreign_user for h in foreign_hits)
        finally:
            await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", _BACKENDS)
async def test_index_tracks_foreign_instance_writes(
    backend: str, tmp_path: Path
) -> None:
    """Two store instances over one database: a write through instance A
    must be visible to instance B's next search even though B's index was
    already loaded (the revision check forces a rebuild)."""
    async with _url(backend, tmp_path) as url:
        user = f"u-{uuid.uuid4().hex}"
        a: BaseStore = open_store(url, embedding_dim=_DIM)
        await a.initialize()
        b: BaseStore = open_store(url, embedding_dim=_DIM)
        await b.initialize()
        try:
            # Warm B's index at the current (empty) revision.
            assert (
                await b.search_by_embedding([1.0, 0.0, 0.0, 0.0], top_k=1, user_id=user)
                == []
            )

            item = _item("written by A after B warmed up", user, [1.0, 0.0, 0.0, 0.0])
            await a.add(item)

            hits = await b.search_by_embedding([1.0, 0.0, 0.0, 0.0], top_k=5, user_id=user)
            assert any(h.id == item.id for h in hits)

            # Deletes propagate the same way.
            assert await a.delete(item.id, hard=True) is True
            hits_after = await b.search_by_embedding(
                [1.0, 0.0, 0.0, 0.0], top_k=5, user_id=user
            )
            assert all(h.id != item.id for h in hits_after)
        finally:
            await a.close()
            await b.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", _BACKENDS)
async def test_warm_index_stays_incremental_for_local_writes(
    backend: str, tmp_path: Path
) -> None:
    """The fast path: one instance's own sequential writes must not force
    a rebuild — the index takes the incremental delta."""
    async with _url(backend, tmp_path) as url:
        store = open_store(url, embedding_dim=_DIM)
        await store.initialize()
        try:
            user = f"u-{uuid.uuid4().hex}"
            first = _item("first", user, [1.0, 0.0, 0.0, 0.0])
            await store.add(first)
            # Warm the index via a search.
            await store.search_by_embedding([1.0, 0.0, 0.0, 0.0], top_k=1, user_id=user)
            assert store._index_loaded is True
            revision = store._loaded_revision
            assert revision is not None

            second = _item("second", user, [0.0, 1.0, 0.0, 0.0])
            await store.add(second)
            # Incremental path: still loaded, revision advanced by exactly one.
            assert store._index_loaded is True
            assert store._loaded_revision == revision + 1

            hits = await store.search_by_embedding(
                [0.0, 1.0, 0.0, 0.0], top_k=5, user_id=user
            )
            assert any(h.id == second.id for h in hits)
        finally:
            await store.close()
