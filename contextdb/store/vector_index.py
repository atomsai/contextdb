"""Vector indices for fast nearest-neighbor search.

Two implementations ship with ContextDB:

* :class:`FAISSIndex` — uses ``faiss-cpu`` when installed; suitable for
  collections up to tens of millions of vectors.
* :class:`NumpyIndex` — pure-numpy brute force; used as a fallback and for
  test determinism. Fine for <10K vectors.

Callers should prefer :func:`get_vector_index`, which picks the best available
implementation at runtime.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalize rows. Zero vectors pass through unchanged."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return cast("NDArray[np.float32]", (vectors / norms).astype(np.float32))


class VectorIndex(ABC):
    """Minimal vector index surface used by :class:`SQLiteStore`."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: NDArray[np.float32]) -> None: ...

    @abstractmethod
    def search(self, query: NDArray[np.float32], top_k: int = 10) -> list[tuple[str, float]]: ...

    @abstractmethod
    def remove(self, ids: list[str]) -> None: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...


class NumpyIndex(VectorIndex):
    """Brute-force cosine similarity over stacked float32 vectors."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._ids: list[str] = []
        self._vectors: NDArray[np.float32] = np.zeros((0, dimension), dtype=np.float32)

    def add(self, ids: list[str], embeddings: NDArray[np.float32]) -> None:
        if not ids:
            return
        reshaped = np.asarray(embeddings, dtype=np.float32).reshape(len(ids), self.dimension)
        normalized = _normalize(reshaped)
        self._ids.extend(ids)
        self._vectors = (
            np.vstack([self._vectors, normalized]) if len(self._vectors) else normalized
        )

    def search(self, query: NDArray[np.float32], top_k: int = 10) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        q = _normalize(np.asarray(query, dtype=np.float32).reshape(1, self.dimension))[0]
        scores = self._vectors @ q
        k = min(top_k, len(self._ids))
        top_idx = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in top_idx]

    def remove(self, ids: list[str]) -> None:
        drop = set(ids)
        keep = [i for i, mid in enumerate(self._ids) if mid not in drop]
        self._ids = [self._ids[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else np.zeros(
            (0, self.dimension), dtype=np.float32
        )

    def save(self, path: str) -> None:
        payload = {
            "dimension": self.dimension,
            "ids": self._ids,
            "vectors": self._vectors,
        }
        Path(path).write_bytes(pickle.dumps(payload))

    def load(self, path: str) -> None:
        payload = pickle.loads(Path(path).read_bytes())
        self.dimension = int(payload["dimension"])
        self._ids = list(payload["ids"])
        self._vectors = np.asarray(payload["vectors"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)


class FAISSIndex(VectorIndex):
    """FAISS-backed index; requires the ``faiss-cpu`` optional dependency.

    Removal is O(1): an id goes into :attr:`_removed_ids` and is filtered out
    of every :meth:`search` call. The underlying FAISS index is rebuilt when
    the tombstone set exceeds :attr:`_rebuild_threshold` (10% by default) so
    search over-fetch stays bounded and index memory does not grow without
    limit.
    """

    _rebuild_threshold: float = 0.1

    def __init__(self, dimension: int, index_type: str = "flat") -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - exercised only without faiss
            raise RuntimeError(
                "FAISS is not installed. Install with `pip install contextdb[faiss]` "
                "or use NumpyIndex."
            ) from exc
        self._faiss = faiss
        self.dimension = dimension
        self.index_type = index_type
        self._index: Any = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []
        self._removed_ids: set[str] = set()

    def add(self, ids: list[str], embeddings: NDArray[np.float32]) -> None:
        if not ids:
            return
        reshaped = np.asarray(embeddings, dtype=np.float32).reshape(len(ids), self.dimension)
        normalized = _normalize(reshaped)
        self._index.add(normalized)
        self._ids.extend(ids)
        # Re-adding a previously tombstoned id clears the tombstone.
        if self._removed_ids:
            self._removed_ids.difference_update(ids)

    def search(self, query: NDArray[np.float32], top_k: int = 10) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        # Over-fetch when tombstones are present so filtered-out hits do not
        # starve the final top-k.
        live = len(self._ids) - len(self._removed_ids)
        if live <= 0:
            return []
        fetch = min(top_k + len(self._removed_ids), len(self._ids))
        q = _normalize(np.asarray(query, dtype=np.float32).reshape(1, self.dimension))
        scores, indices = self._index.search(q, fetch)
        out: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:
                continue
            mid = self._ids[int(idx)]
            if mid in self._removed_ids:
                continue
            out.append((mid, float(score)))
            if len(out) >= top_k:
                break
        return out

    def remove(self, ids: list[str]) -> None:
        """Tombstone the given ids. Amortized O(1) per id.

        When tombstones exceed :attr:`_rebuild_threshold` of the live set,
        :meth:`rebuild` is triggered to reclaim memory.
        """
        if not ids:
            return
        present = {mid for mid in ids if mid in self._ids}
        if not present:
            return
        self._removed_ids.update(present)
        total = len(self._ids)
        if total and len(self._removed_ids) / total > self._rebuild_threshold:
            self.rebuild()

    def rebuild(self) -> None:
        """Reconstruct the underlying FAISS index without the tombstoned ids.

        Expensive (O(n) over live vectors) but amortized across removals.
        """
        if not self._removed_ids:
            return
        keep_idx = [i for i, mid in enumerate(self._ids) if mid not in self._removed_ids]
        new_ids = [self._ids[i] for i in keep_idx]
        if keep_idx:
            snapshot = self._vectors_snapshot()
            vectors = np.stack([snapshot[i] for i in keep_idx], axis=0).astype(np.float32)
        else:
            vectors = np.zeros((0, self.dimension), dtype=np.float32)
        self._index = self._faiss.IndexFlatIP(self.dimension)
        self._ids = []
        self._removed_ids.clear()
        if len(new_ids):
            self._index.add(vectors)
            self._ids = new_ids

    def _vectors_snapshot(self) -> NDArray[np.float32]:
        # Reconstruct all current vectors from the FAISS index.
        return np.stack(
            [self._index.reconstruct(i) for i in range(self._index.ntotal)], axis=0
        ).astype(np.float32)

    def save(self, path: str) -> None:
        # Persisted state always reflects the post-rebuild, tombstone-free view
        # so stale ids never leak into a reloaded index.
        if self._removed_ids:
            self.rebuild()
        self._faiss.write_index(self._index, f"{path}.faiss")
        Path(f"{path}.ids").write_bytes(pickle.dumps(self._ids))

    def load(self, path: str) -> None:
        self._index = self._faiss.read_index(f"{path}.faiss")
        self._ids = pickle.loads(Path(f"{path}.ids").read_bytes())
        self._removed_ids = set()

    def __len__(self) -> int:
        return len(self._ids) - len(self._removed_ids)


def get_vector_index(dimension: int, prefer: str = "auto") -> VectorIndex:
    """Return the best available index for ``dimension``.

    ``prefer`` may be ``"faiss"``, ``"numpy"``, or ``"auto"`` (the default,
    which uses FAISS if installed and falls back to NumPy otherwise).
    """
    if prefer == "numpy":
        return NumpyIndex(dimension)
    if prefer == "faiss":
        return FAISSIndex(dimension)
    try:
        return FAISSIndex(dimension)
    except RuntimeError:
        return NumpyIndex(dimension)
