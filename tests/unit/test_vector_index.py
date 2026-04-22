"""Tests for :class:`NumpyIndex` and :class:`FAISSIndex`."""

from __future__ import annotations

import numpy as np
import pytest

from contextdb.store.vector_index import NumpyIndex

try:
    import faiss  # noqa: F401

    from contextdb.store.vector_index import FAISSIndex

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


def test_numpy_index_add_and_search() -> None:
    idx = NumpyIndex(dimension=4)
    idx.add(
        ["a", "b"],
        np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )
    hits = idx.search(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert hits[0][0] == "a"


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")
def test_faiss_remove_lazy_excludes_from_search() -> None:
    """remove() should tombstone without rebuilding; search must exclude the id."""
    # Use 20 vectors so removing one stays under the 10% rebuild threshold.
    idx = FAISSIndex(dimension=4)
    n = 20
    ids = [f"id{i}" for i in range(n)]
    vectors = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        vectors[i, i % 4] = 1.0
    # Ensure the first vector sits alone on the query direction so search
    # would place it at rank 0 without the tombstone filter.
    vectors[0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    idx.add(ids, vectors)

    idx.remove(["id0"])
    # Single removal of 1/20 (5%) stays under the 10% threshold — tombstone
    # persists and the underlying FAISS index is NOT rebuilt.
    assert "id0" in idx._removed_ids
    assert idx._index.ntotal == n
    assert len(idx) == n - 1

    hits = idx.search(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=5)
    returned_ids = [h[0] for h in hits]
    assert "id0" not in returned_ids


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")
def test_faiss_auto_rebuild_above_threshold() -> None:
    """Passing the 10% threshold must trigger a rebuild that clears tombstones."""
    idx = FAISSIndex(dimension=4)
    n = 20
    ids = [f"id{i}" for i in range(n)]
    vectors = np.eye(n, 4, dtype=np.float32)[:n]
    # Pad to dimension 4 if needed.
    if vectors.shape[1] < 4:
        vectors = np.pad(vectors, ((0, 0), (0, 4 - vectors.shape[1])))
    idx.add(ids, vectors.astype(np.float32))
    # Removing 3 of 20 (15%) crosses the 10% threshold and rebuilds.
    idx.remove(["id0", "id1", "id2"])
    assert idx._removed_ids == set()
    assert idx._index.ntotal == n - 3
    assert len(idx) == n - 3


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")
def test_faiss_readd_clears_tombstone() -> None:
    """Re-adding a tombstoned id must make it visible again."""
    idx = FAISSIndex(dimension=4)
    vec = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    idx.add(["a"], vec)
    idx.remove(["a"])
    assert len(idx) == 0
    idx.add(["a"], vec)
    assert "a" not in idx._removed_ids
    hits = idx.search(vec[0], top_k=1)
    assert hits and hits[0][0] == "a"
