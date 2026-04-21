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
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalize rows. Zero vectors pass through unchanged."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


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
        arr = np.asarray(embeddings, dtype=np.float32).reshape(len(ids), self.dimension)
        arr = _normalize(arr)
        self._ids.extend(ids)
        self._vectors = np.vstack([self._vectors, arr]) if len(self._vectors) else arr

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
    """FAISS-backed index; requires the ``faiss-cpu`` optional dependency."""

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

    def add(self, ids: list[str], embeddings: NDArray[np.float32]) -> None:
        if not ids:
            return
        arr = np.asarray(embeddings, dtype=np.float32).reshape(len(ids), self.dimension)
        arr = _normalize(arr)
        self._index.add(arr)
        self._ids.extend(ids)

    def search(self, query: NDArray[np.float32], top_k: int = 10) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        q = _normalize(np.asarray(query, dtype=np.float32).reshape(1, self.dimension))
        scores, indices = self._index.search(q, min(top_k, len(self._ids)))
        out: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:
                continue
            out.append((self._ids[int(idx)], float(score)))
        return out

    def remove(self, ids: list[str]) -> None:
        drop = set(ids)
        keep_idx = [i for i, mid in enumerate(self._ids) if mid not in drop]
        if len(keep_idx) == len(self._ids):
            return
        # FAISS flat index: rebuild from scratch.
        self._index = self._faiss.IndexFlatIP(self.dimension)
        new_ids = [self._ids[i] for i in keep_idx]
        self._ids = []
        if new_ids:
            vectors = np.stack(
                [self._vectors_snapshot()[i] for i in keep_idx], axis=0
            ).astype(np.float32)
            self._index.add(vectors)
            self._ids = new_ids

    def _vectors_snapshot(self) -> NDArray[np.float32]:
        # Reconstruct all current vectors from the FAISS index.
        return np.stack(
            [self._index.reconstruct(i) for i in range(self._index.ntotal)], axis=0
        ).astype(np.float32)

    def save(self, path: str) -> None:
        self._faiss.write_index(self._index, f"{path}.faiss")
        Path(f"{path}.ids").write_bytes(pickle.dumps(self._ids))

    def load(self, path: str) -> None:
        self._index = self._faiss.read_index(f"{path}.faiss")
        self._ids = pickle.loads(Path(f"{path}.ids").read_bytes())

    def __len__(self) -> int:
        return len(self._ids)


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
