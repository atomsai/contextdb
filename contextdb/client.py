"""The :class:`ContextDB` client — the public entry point.

Wires together storage, embeddings, LLM, PII, graphs, formation, evolution,
retrieval, and audit behind a small surface:

* ``add`` / ``search`` / ``get`` / ``update`` / ``delete`` — CRUD + recall.
* ``add_conversation`` — run raw text through the formation pipeline.
* ``forget`` / ``stats`` / ``consolidate`` / ``prune`` — lifecycle operations.
* ``factual`` / ``experiential`` / ``working`` — typed memory sub-APIs.

The client is lazy: resources are created on the first await, which keeps
``contextdb.init()`` cheap and side-effect-free.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from contextdb.core.config import ContextDBConfig
from contextdb.core.exceptions import ContextDBError
from contextdb.core.models import EpistemicSource, MemoryItem, MemoryStatus, MemoryType
from contextdb.dynamics.trust import TrustEngine, infer_action_relevant
from contextdb.privacy.pii_detector import PIIDetector
from contextdb.store.sqlite_store import SQLiteStore
from contextdb.utils.embeddings import EmbeddingProvider, get_embedding_provider
from contextdb.utils.llm import LazyLLM, LLMProvider

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from contextdb.agents.memory_bus import MemoryBus
    from contextdb.agents.rl_manager import RLMemoryManager
    from contextdb.dynamics.evolution import AutoLinker, Consolidator, Pruner
    from contextdb.dynamics.formation import FormationPipeline
    from contextdb.dynamics.retrieval import RetrievalEngine
    from contextdb.graphs.base import BaseGraph
    from contextdb.memory.experiential import ExperientialMemory
    from contextdb.memory.factual import FactualMemory
    from contextdb.memory.working import WorkingMemory
    from contextdb.privacy.audit import AuditLogger
    from contextdb.privacy.retention import RetentionManager


class ContextDB:
    """Memory operating system for AI agents — the user-facing interface."""

    def __init__(self, config: ContextDBConfig, user_id: str | None = None) -> None:
        self.config = config
        self.user_id = user_id
        self._store: SQLiteStore | None = None
        self._embedder: EmbeddingProvider | None = None
        self._llm: LLMProvider | None = None
        self._pii: PIIDetector | None = None
        self._graphs: dict[str, BaseGraph] = {}
        self._retrieval: RetrievalEngine | None = None
        self._formation: FormationPipeline | None = None
        self._auto_linker: AutoLinker | None = None
        self._consolidator: Consolidator | None = None
        self._pruner: Pruner | None = None
        self._audit: AuditLogger | None = None
        self._retention: RetentionManager | None = None
        self._memory_bus: MemoryBus | None = None
        self._rl_manager: RLMemoryManager | None = None
        self._trust: TrustEngine | None = None
        self._consolidation_task: asyncio.Task[None] | None = None
        self._initialized = False
        self._warn_if_llm_unusable()

    def _warn_if_llm_unusable(self) -> None:
        """Warn loudly at init when extraction cannot possibly work.

        The historical failure mode was silent: no key meant every
        extraction call failed downstream and the pipeline degraded to
        storing raw text. We cannot fix remote misconfiguration, but we can
        make sure the operator hears about it before the first write.
        """
        model = self.config.llm_model
        if model in {"mock", "test"}:
            return
        if self.config.llm_base_url is not None:
            return
        if self.config.llm_api_key:
            return
        message = (
            f"ContextDB: no API key configured for LLM model '{model}'. "
            "Extraction, compression, and consolidation will raise ConfigError "
            "on first use instead of silently degrading. Set "
            "CONTEXTDB_LLM_API_KEY / OPENAI_API_KEY, or pass llm_base_url to "
            "use an OpenAI-compatible endpoint (Groq, Ollama, vLLM, Together)."
        )
        _logger.warning(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        # Core building blocks
        self._embedder = get_embedding_provider(
            self.config.embedding_model,
            self.config.llm_api_key,
            dimension=self.config.embedding_dim,
            base_url=self.config.embedding_base_url,
        )
        dim = self._embedder.dimension()
        self._store = SQLiteStore(
            storage_url=self.config.storage_url,
            user_id=self.user_id,
            embedding_dim=dim,
        )
        await self._store.initialize()
        if self._llm is None:
            # Lazy: realtime writes (add_fast) must not require — or touch —
            # an LLM. The provider is constructed on first actual use.
            self._llm = LazyLLM(
                self.config.llm_model,
                self.config.llm_api_key,
                base_url=self.config.llm_base_url,
            )
        self._pii = PIIDetector(
            action=self.config.pii_action,
            encryption_key=self.config.pii_encryption_key,
        )

        # Graphs (local imports to avoid circular references at module load)
        from contextdb.graphs.semantic import SemanticGraph

        semantic = SemanticGraph(self._store)
        await semantic.initialize()
        self._graphs["semantic"] = semantic

        if self.config.enable_entity_graph:
            from contextdb.graphs.entity import EntityGraph

            entity_graph = EntityGraph(self._store, self._llm)
            await entity_graph.initialize()
            self._graphs["entity"] = entity_graph

        if self.config.enable_multi_graph:
            from contextdb.graphs.causal import CausalGraph
            from contextdb.graphs.temporal import TemporalGraph

            temporal = TemporalGraph(self._store)
            await temporal.initialize()
            self._graphs["temporal"] = temporal
            causal = CausalGraph(self._store, self._llm)
            await causal.initialize()
            self._graphs["causal"] = causal

        # Dynamics
        from contextdb.dynamics.evolution import AutoLinker, Consolidator, Pruner
        from contextdb.dynamics.formation import (
            FormationPipeline,
            MemoryCompressor,
            MemoryExtractor,
            Segmenter,
        )
        from contextdb.dynamics.retrieval import (
            QueryClassifier,
            RetrievalEngine,
            RetrievalFuser,
        )

        self._auto_linker = AutoLinker(self._graphs)
        self._retrieval = RetrievalEngine(
            self._store, self._graphs, QueryClassifier(), RetrievalFuser()
        )
        self._formation = FormationPipeline(
            Segmenter(),
            MemoryExtractor(self._llm),
            MemoryCompressor(self._llm),
            self._pii,
            self._embedder,
        )
        from contextdb.graphs.semantic import SemanticGraph as _SemanticGraphType

        semantic_graph = self._graphs["semantic"]
        assert isinstance(semantic_graph, _SemanticGraphType)
        self._consolidator = Consolidator(self._store, semantic_graph, self._llm)
        self._pruner = Pruner(self._store)

        # Privacy
        if self.config.enable_audit:
            from contextdb.privacy.audit import AuditLogger

            self._audit = AuditLogger(self._store)
            await self._audit.initialize()

        # Trust write path (Epics 1-2): slot dedupe, corroboration, supersede.
        self._trust = TrustEngine(self._store, self._audit)

        from contextdb.core.models import RetentionPolicy
        from contextdb.privacy.retention import RetentionManager

        self._retention = RetentionManager(
            self._store,
            self._audit,
            RetentionPolicy(
                default_ttl=(
                    timedelta(days=self.config.retention_ttl_days)
                    if self.config.retention_ttl_days
                    else None
                )
            ),
        )

        # RL (optional, paid tier)
        if self.config.enable_rl_manager:
            from contextdb.agents.rl_manager import RLMemoryManager

            self._rl_manager = RLMemoryManager(self._llm)

        self._initialized = True

    # ------------------------------------------------------------------ #
    # Accessors guarded against misuse
    # ------------------------------------------------------------------ #

    def _require_store(self) -> SQLiteStore:
        if self._store is None:
            raise ContextDBError("ContextDB not initialized; await _ensure_init() first.")
        return self._store

    # ------------------------------------------------------------------ #
    # Core CRUD / search
    # ------------------------------------------------------------------ #

    async def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACTUAL,
        metadata: dict[str, Any] | None = None,
        event_time: datetime | None = None,
        source: str = "",
        entity_mentions: list[str] | None = None,
        *,
        epistemic_source: EpistemicSource | None = None,
        confidence: float | None = None,
        action_relevant: bool | None = None,
        entity_key: str | None = None,
        attribute_key: str | None = None,
    ) -> MemoryItem:
        """Store a memory.

        The keyword-only trust parameters are additive (Epic 1). When both
        ``entity_key`` and ``attribute_key`` are supplied, the write routes
        through the trust engine: a same-value write into the occupied slot
        corroborates the existing memory instead of duplicating it, and a
        different-value write supersedes it (Epic 2).
        """
        await self._ensure_init()
        assert self._pii is not None
        assert self._embedder is not None
        assert self._trust is not None
        store = self._require_store()

        processed, pii_annotations = self._pii.process(content)

        # Optional RL override: NOOP / UPDATE / DELETE short-circuit ADD.
        if self._rl_manager is not None:
            candidates = await store.list_memories(limit=20)
            decision = await self._rl_manager.decide(processed, candidates)
            action = decision.get("action", "ADD").upper()
            if action == "NOOP":
                raise ContextDBError("RL manager chose NOOP; nothing stored.")
            if action == "UPDATE" and decision.get("target_memory_id"):
                target = decision["target_memory_id"]
                merged = decision.get("content") or processed
                return await self.update(target, content=merged, metadata=metadata)
            if action == "DELETE" and decision.get("target_memory_id"):
                await self.delete(decision["target_memory_id"])

        embedding = (await self._embedder.embed([processed]))[0]
        item = MemoryItem(
            content=processed,
            embedding=embedding,
            memory_type=memory_type,
            source=source,
            metadata=metadata or {},
            event_time=event_time or datetime.now(tz=timezone.utc),
            pii_annotations=pii_annotations,
            entity_mentions=entity_mentions or [],
            epistemic_source=epistemic_source or "user_stated",
            confidence=confidence if confidence is not None else 1.0,
            action_relevant=(
                action_relevant
                if action_relevant is not None
                else infer_action_relevant(processed)
            ),
            entity_key=entity_key,
            attribute_key=attribute_key,
            valid_from=datetime.now(tz=timezone.utc),
        )

        if entity_key and attribute_key:
            stored, outcome = await self._trust.write(item, user_id=self.user_id)
        else:
            stored = await store.add(item)
            outcome = "added"
            if self._audit is not None:
                await self._audit.log(
                    operation="CREATE",
                    memory_id=stored.id,
                    user_id=self.user_id,
                    details={"memory_type": memory_type.value},
                )

        # Only freshly-stored memories get graph links; a corroboration
        # reuses the existing node.
        if (
            outcome in {"added", "superseded"}
            and self.config.enable_auto_link
            and self._auto_linker is not None
        ):
            await self._auto_linker.link(
                stored.id,
                {
                    "content": stored.content,
                    "embedding": stored.embedding,
                    "event_time": stored.event_time,
                },
            )
        return stored

    async def search(
        self,
        query: str,
        top_k: int = 10,
        memory_type: MemoryType | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        as_of: datetime | None = None,
    ) -> list[MemoryItem]:
        """Semantic + graph recall.

        Defaults to *currently valid* memories only (Epic 2): a superseded
        fact is retained for audit but no longer recalled. Pass ``as_of``
        for a time-travel query — "what did we believe at moment T?".
        """
        await self._ensure_init()
        assert self._embedder is not None
        assert self._retrieval is not None
        moment = as_of or datetime.now(tz=timezone.utc)
        query_embedding = (await self._embedder.embed([query]))[0]
        # Over-fetch so validity/type filtering cannot starve top_k.
        items = await self._retrieval.search(query, query_embedding, top_k=top_k * 3)
        items = [m for m in items if m.is_valid_at(moment)]
        if memory_type is not None:
            items = [m for m in items if m.memory_type == memory_type]
        if time_range is not None:
            start, end = time_range
            items = [m for m in items if m.event_time and start <= m.event_time <= end]
        items = items[:top_k]
        if self._audit is not None:
            await self._audit.log(
                operation="SEARCH",
                user_id=self.user_id,
                details={
                    "query": query,
                    "hits": len(items),
                    "returned_ids": [m.id for m in items],
                    "as_of": moment.isoformat(),
                },
            )
        return items

    async def add_fast(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACTUAL,
        metadata: dict[str, Any] | None = None,
        entity_mentions: list[str] | None = None,
    ) -> MemoryItem:
        """Realtime write path: append raw + embed inline, NEVER call an LLM.

        For voice loops, where an extraction call on the write path is a
        missed turn. The memory is recallable immediately as raw content
        (``epistemic_source="user_stated"``, ``confidence=0.5``,
        ``pending_consolidation=True``); extraction, slot dedupe, and
        contradiction handling happen later in :meth:`consolidate_pending`
        or the background consolidation loop.

        PII is still processed before embedding — the privacy rule is not
        a latency trade-off we make.
        """
        await self._ensure_init()
        assert self._pii is not None
        assert self._embedder is not None
        store = self._require_store()

        processed, pii_annotations = self._pii.process(content)
        embedding = (await self._embedder.embed([processed]))[0]
        now = datetime.now(tz=timezone.utc)
        item = MemoryItem(
            content=processed,
            embedding=embedding,
            memory_type=memory_type,
            metadata=metadata or {},
            event_time=now,
            pii_annotations=pii_annotations,
            entity_mentions=entity_mentions or [],
            epistemic_source="user_stated",
            confidence=0.5,
            action_relevant=infer_action_relevant(processed),
            valid_from=now,
            pending_consolidation=True,
        )
        stored = await store.add(item)
        if self._audit is not None:
            await self._audit.log(
                operation="CREATE",
                memory_id=stored.id,
                user_id=self.user_id,
                details={"memory_type": memory_type.value, "fast": True},
            )
        return stored

    async def consolidate_pending(self, batch_size: int = 50) -> int:
        """Drain the fast-write queue: extract, dedupe, supersede behind the write.

        Each pending memory is run through LLM extraction. Extracted facts
        are written through the trust engine (slot dedupe / corroboration /
        supersede) and the raw memory is archived; when extraction yields
        nothing, the raw memory itself is upgraded in place (heuristic
        action-relevance) so nothing is ever lost. Returns the number of
        pending memories processed.
        """
        await self._ensure_init()
        assert self._formation is not None
        assert self._trust is not None
        assert self._pii is not None
        assert self._embedder is not None
        store = self._require_store()

        pending = await store.list_pending_consolidation(limit=batch_size)
        processed_count = 0
        for raw in pending:
            facts = await self._formation.extractor.extract(raw.content)
            if not facts:
                await store.update(
                    raw.id,
                    pending_consolidation=False,
                    action_relevant=infer_action_relevant(raw.content),
                )
                await self._link(raw.id)
                processed_count += 1
                continue
            for fact in facts:
                fact_content, fact_pii = self._pii.process(fact["content"])
                embedding = (await self._embedder.embed([fact_content]))[0]
                new_item = MemoryItem(
                    content=fact_content,
                    embedding=embedding,
                    memory_type=MemoryType(fact["memory_type"]),
                    source=raw.source,
                    pii_annotations=fact_pii,
                    entity_mentions=list(fact.get("entities", [])),
                    epistemic_source=fact.get("epistemic_source", "user_stated"),
                    confidence=float(fact.get("confidence", 0.8)),
                    action_relevant=bool(
                        fact.get("action_relevant", infer_action_relevant(fact_content))
                    ),
                    entity_key=fact.get("entity_key"),
                    attribute_key=fact.get("attribute_key"),
                    metadata={"consolidated_from": [raw.id]},
                )
                saved, outcome = await self._trust.write(new_item, user_id=self.user_id)
                if outcome in {"added", "superseded"}:
                    await self._link(saved.id)
            await store.update(
                raw.id,
                pending_consolidation=False,
                status=MemoryStatus.ARCHIVED,
            )
            processed_count += 1
        return processed_count

    async def start_consolidation_loop(
        self, interval_seconds: float = 5.0
    ) -> asyncio.Task[None]:
        """Background consolidator for realtime hosts (Epic 3).

        Batch hosts can call :meth:`consolidate` / :meth:`consolidate_pending`
        directly instead. The loop keeps running until :meth:`close`.
        """
        await self._ensure_init()
        if self._consolidation_task is not None and not self._consolidation_task.done():
            return self._consolidation_task

        async def _loop() -> None:
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    await self.consolidate_pending()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001
                    _logger.exception("background consolidation pass failed")

        self._consolidation_task = asyncio.create_task(_loop())
        return self._consolidation_task

    async def _link(self, memory_id: str) -> None:
        """Best-effort graph auto-link for a stored memory."""
        if not (self.config.enable_auto_link and self._auto_linker is not None):
            return
        item = await self._require_store().get_raw(memory_id)
        if item is None:
            return
        await self._auto_linker.link(
            item.id,
            {
                "content": item.content,
                "embedding": item.embedding,
                "event_time": item.event_time,
            },
        )

    async def get(self, memory_id: str) -> MemoryItem | None:
        await self._ensure_init()
        item = await self._require_store().get(memory_id)
        if self._audit is not None and item is not None:
            await self._audit.log(
                operation="READ", memory_id=memory_id, user_id=self.user_id
            )
        return item

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        await self._ensure_init()
        assert self._pii is not None
        assert self._embedder is not None
        kwargs: dict[str, Any] = {}
        if content is not None:
            processed, pii = self._pii.process(content)
            embedding = (await self._embedder.embed([processed]))[0]
            kwargs["content"] = processed
            kwargs["embedding"] = embedding
            kwargs["pii_annotations"] = pii
        if metadata is not None:
            kwargs["metadata"] = metadata
        item = await self._require_store().update(memory_id, **kwargs)
        if self._audit is not None:
            await self._audit.log(
                operation="UPDATE", memory_id=memory_id, user_id=self.user_id
            )
        return item

    async def delete(self, memory_id: str, hard: bool = False) -> None:
        await self._ensure_init()
        await self._require_store().delete(memory_id, hard=hard)
        if self._audit is not None:
            await self._audit.log(
                operation="DELETE", memory_id=memory_id, user_id=self.user_id
            )

    async def add_conversation(self, conversation: str, source: str = "") -> list[MemoryItem]:
        """Run a raw conversation through the formation pipeline and store results.

        Extracted facts carrying entity+attribute slot keys go through the
        trust engine, so repeats corroborate and corrections supersede
        instead of piling up as parallel "truths".
        """
        await self._ensure_init()
        assert self._formation is not None
        assert self._trust is not None
        items = await self._formation.process(conversation, source=source)
        stored: list[MemoryItem] = []
        store = self._require_store()
        for item in items:
            if item.entity_key and item.attribute_key:
                saved, outcome = await self._trust.write(item, user_id=self.user_id)
            else:
                saved = await store.add(item)
                outcome = "added"
            if (
                outcome in {"added", "superseded"}
                and self.config.enable_auto_link
                and self._auto_linker is not None
            ):
                await self._auto_linker.link(
                    saved.id,
                    {
                        "content": saved.content,
                        "embedding": saved.embedding,
                        "event_time": saved.event_time,
                    },
                )
            stored.append(saved)
        if self._audit is not None:
            await self._audit.log(
                operation="CREATE",
                user_id=self.user_id,
                details={"count": len(stored), "source": source},
            )
        return stored

    async def forget(
        self,
        user_id: str | None = None,
        entity: str | None = None,
        older_than: timedelta | None = None,
    ) -> int:
        """Bulk-delete memories.

        Age-only forgets use a single SQL ``DELETE`` to stay O(1) in Python
        memory. Entity-scoped forgets must inspect JSON-serialised
        ``entity_mentions`` and free-text content, so they stream memories in
        500-row pages rather than loading the full table.
        """
        await self._ensure_init()
        store = self._require_store()

        # Fast path: age-only deletes lower to a single SQL statement.
        if entity is None and older_than is not None:
            cutoff = datetime.now(tz=timezone.utc) - older_than
            deleted = await store.delete_older_than(
                cutoff.isoformat(), user_id=user_id, hard=True
            )
            if self._audit is not None:
                await self._audit.log(
                    operation="ERASE",
                    user_id=user_id,
                    details={"bulk": True, "count": deleted, "older_than": older_than.days},
                )
            return deleted

        now = datetime.now(tz=timezone.utc)
        needle = entity.lower() if entity is not None else None
        deleted = 0
        async for m in store.iter_memories(user_id=user_id, batch_size=500):
            matches = True
            if needle is not None:
                ents = [e.lower() for e in m.entity_mentions]
                if needle not in ents and needle not in m.content.lower():
                    matches = False
            if older_than is not None and now - m.created_at < older_than:
                matches = False
            if matches:
                await store.delete(m.id, hard=True)
                deleted += 1
                if self._audit is not None:
                    await self._audit.log(
                        operation="ERASE",
                        memory_id=m.id,
                        user_id=user_id,
                        details={"bulk": True},
                    )
        return deleted

    async def stats(self) -> dict[str, Any]:
        await self._ensure_init()
        store = self._require_store()
        total = await store.count(self.user_id)
        by_type = await store.count_by_type(self.user_id)
        return {
            "total_memories": total,
            "user_id": self.user_id,
            "by_type": by_type,
            "graphs": list(self._graphs.keys()),
        }

    async def consolidate(self, min_cluster_size: int = 5) -> list[MemoryItem]:
        """Batch consolidation: drain the fast-write queue, then cluster-merge."""
        await self._ensure_init()
        assert self._consolidator is not None
        await self.consolidate_pending()
        return await self._consolidator.consolidate(min_cluster_size=min_cluster_size)

    async def prune(self, strategy: str = "decay", **kwargs: Any) -> int:
        await self._ensure_init()
        assert self._pruner is not None
        return await self._pruner.prune(strategy=strategy, **kwargs)

    async def get_timeline(
        self,
        entity: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[MemoryItem]:
        await self._ensure_init()
        if "temporal" in self._graphs:
            from contextdb.graphs.temporal import TemporalGraph

            temporal = self._graphs["temporal"]
            assert isinstance(temporal, TemporalGraph)
            return await temporal.get_timeline(entity=entity, start=start, end=end)
        store = self._require_store()
        memories = await store.list_memories(limit=10000)
        filtered = [m for m in memories if m.event_time is not None]
        filtered.sort(key=lambda m: m.event_time or datetime.min.replace(tzinfo=timezone.utc))
        if start is not None:
            filtered = [m for m in filtered if m.event_time and m.event_time >= start]
        if end is not None:
            filtered = [m for m in filtered if m.event_time and m.event_time <= end]
        if entity is not None:
            filtered = [
                m
                for m in filtered
                if entity.lower() in " ".join(m.entity_mentions).lower()
                or entity.lower() in m.content.lower()
            ]
        return filtered

    async def get_entity(self, name: str) -> dict[str, Any]:
        await self._ensure_init()
        if "entity" in self._graphs:
            from contextdb.graphs.entity import EntityGraph

            entity_graph = self._graphs["entity"]
            assert isinstance(entity_graph, EntityGraph)
            return await entity_graph.get_entity_profile(name)
        return {"name": name, "memories": [], "attributes": {}}

    # ------------------------------------------------------------------ #
    # Typed memory surfaces
    # ------------------------------------------------------------------ #

    @property
    def factual(self) -> FactualMemory:
        from contextdb.memory.factual import FactualMemory

        return FactualMemory(self, self.user_id)

    @property
    def experiential(self) -> ExperientialMemory:
        from contextdb.memory.experiential import ExperientialMemory

        return ExperientialMemory(self, self.user_id)

    def working(self, session_id: str, max_tokens: int = 4000) -> WorkingMemory:
        from contextdb.memory.working import WorkingMemory

        return WorkingMemory(self, session_id, max_tokens=max_tokens)

    @property
    def privacy(self) -> RetentionManager:
        if self._retention is None:
            raise ContextDBError(
                "ContextDB not initialized; call await db._ensure_init() first."
            )
        return self._retention

    @property
    def audit(self) -> AuditLogger | None:
        return self._audit

    def bus(self) -> MemoryBus:
        """Return (or create) the in-process multi-agent event bus."""
        from contextdb.agents.memory_bus import MemoryBus

        if self._memory_bus is None:
            self._memory_bus = MemoryBus()
        return self._memory_bus

    # ------------------------------------------------------------------ #
    # Resource management
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        if self._consolidation_task is not None:
            self._consolidation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consolidation_task
            self._consolidation_task = None
        if self._store is not None:
            await self._store.close()
        self._initialized = False

    async def __aenter__(self) -> ContextDB:
        await self._ensure_init()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
