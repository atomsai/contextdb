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
import hashlib
import inspect
import logging
import warnings
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from contextdb.core.clock import Clock, utc_now
from contextdb.core.config import ContextDBConfig
from contextdb.core.exceptions import ConfigError, ContextDBError, SourceRequiredError
from contextdb.core.models import (
    EpistemicSource,
    MemoryExplanation,
    MemoryItem,
    MemoryStatus,
    MemoryType,
)
from contextdb.core.policy import TrustPolicy
from contextdb.core.slots import canonicalize_slot, infer_negation, infer_slot
from contextdb.dynamics.trust import TrustEngine, infer_action_relevant, speaker_id
from contextdb.privacy.injection import screen_injection
from contextdb.privacy.pii_detector import PIIDetector
from contextdb.store.base import BaseStore
from contextdb.store.factory import open_store
from contextdb.utils.embeddings import EmbeddingProvider, get_embedding_provider, wrap_embedder
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

    def __init__(
        self,
        config: ContextDBConfig,
        user_id: str | None = None,
        *,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        clock: Clock | None = None,
        trust_policy: TrustPolicy | None = None,
    ) -> None:
        self.config = config
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.session_id = session_id
        self.clock: Clock = clock or utc_now
        # Honour config.relevance_floor when the caller did not pass a policy.
        self.trust_policy = trust_policy or TrustPolicy(
            relevance_floor=config.relevance_floor
        )
        self._store: BaseStore | None = None
        self._hooks: dict[str, list[Callable[..., Any]]] = {}
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
        self._embedder = wrap_embedder(
            get_embedding_provider(
                self.config.embedding_model,
                self.config.llm_api_key,
                dimension=self.config.embedding_dim,
                base_url=self.config.embedding_base_url,
            ),
            cache_size=self.config.embedding_cache_size,
            timeout_seconds=self.config.embed_timeout_seconds,
        )
        dim = self._embedder.dimension()
        self._store = open_store(
            self.config.storage_url,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
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
            self._store,
            self._graphs,
            QueryClassifier(),
            RetrievalFuser(),
            enable_salience=self.config.enable_salience,
            salience_half_life_days=self.config.salience_half_life_days,
            clock=self.clock,
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
        self._trust = TrustEngine(self._store, self._audit, clock=self.clock)

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

        # RL (optional local pathway)
        if self.config.enable_rl_manager:
            from contextdb.agents.rl_manager import RLMemoryManager

            self._rl_manager = RLMemoryManager(self._llm)

        self._initialized = True

    # ------------------------------------------------------------------ #
    # Accessors guarded against misuse
    # ------------------------------------------------------------------ #

    def _require_store(self) -> BaseStore:
        if self._store is None:
            raise ContextDBError("ContextDB not initialized; await _ensure_init() first.")
        return self._store

    def _resolve_user(self, user_id: str | None) -> str | None:
        if self.user_id is not None:
            if user_id is not None and user_id != self.user_id:
                raise ConfigError(
                    f"This client is scoped to user_id={self.user_id!r} and "
                    f"cannot operate as {user_id!r}. Call init() without "
                    "user_id and pass user_id= on each call."
                )
            return self.user_id
        return user_id

    def _require_epistemic_source(
        self,
        source: EpistemicSource | None,
        *,
        path: str,
        strict: bool = False,
    ) -> None:
        if source is not None:
            return
        message = (
            f"ContextDB.{path}: epistemic source was omitted. "
            "The write is stored as user_stated. Hosts that forget this "
            "field produce first-party-looking memories that never trip an "
            "error. Pass source='user_stated', 'agent_inferred', or "
            "'third_party'."
        )
        if strict or self.config.require_source:
            raise SourceRequiredError(message)
        _logger.warning(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def on(
        self,
        event: str,
        hook: Callable[[str, dict[str, Any]], Awaitable[None] | None],
    ) -> None:
        """Register an ops callback. Events: write, recall, confirm, forget,
        injection_suspect, embed_fallback. Use ``*`` for all events.
        """
        self._hooks.setdefault(event, []).append(hook)

    async def _emit(self, event: str, **payload: Any) -> None:
        hooks = [*self._hooks.get(event, []), *self._hooks.get("*", [])]
        for hook in hooks:
            result = hook(event, payload)
            if inspect.isawaitable(result):
                await result

    # ------------------------------------------------------------------ #
    # Core CRUD / search
    # ------------------------------------------------------------------ #

    @staticmethod
    def _screen_item(item: MemoryItem) -> MemoryItem:
        """Write-time injection screen (Epic 5).

        Runs on every write path. Caller-supplied trust fields can never
        clear the flag — screening only moves a memory toward less trust.
        """
        if not item.injection_suspect and screen_injection(item.content):
            item.injection_suspect = True
            item.epistemic_source = "third_party"
            item.confidence = 0.0
        return item

    def _apply_slot(
        self,
        item: MemoryItem,
        raw_text: str | None = None,
        *,
        extractor_bound: bool = False,
    ) -> MemoryItem:
        """Canonicalize an explicit slot or infer one from raw text.

        When ``extractor_bound`` is set (LLM extraction / consolidation),
        the extractor is a security boundary: a high-stakes slot the raw
        text does not independently support is demoted, and a value the
        raw text contradicts is rewritten to the raw-derived value.
        Direct ``factual.add`` is developer-authored and is not bound.
        """
        from contextdb.core.slots import canonical_slot_value

        source_text = raw_text or item.content
        explicit = canonicalize_slot(item.entity_key, item.attribute_key)
        inferred = infer_slot(source_text)
        chosen = explicit or inferred
        if explicit is not None and inferred is not None:
            if (explicit.entity, explicit.attribute) != (inferred.entity, inferred.attribute):
                chosen = inferred
                item.epistemic_source = "agent_inferred"
                item.confidence = min(item.confidence, 0.4)
                warnings.warn(
                    "entity/attribute did not match the text; stored as "
                    "agent_inferred at confidence <= 0.4. That fact will not "
                    "pass recall_for_action until it is corroborated or confirmed.",
                    UserWarning,
                    stacklevel=3,
                )
            elif extractor_bound:
                raw_val = canonical_slot_value(source_text, inferred)
                extracted_val = canonical_slot_value(item.content, explicit)
                if raw_val != extracted_val:
                    item.epistemic_source = "agent_inferred"
                    item.confidence = min(item.confidence, 0.4)
                    item.slot_value = raw_val
        if (
            extractor_bound
            and explicit is not None
            and inferred is None
            and explicit.slot_class in {"health", "legal", "identity", "money"}
        ):
            item.epistemic_source = "agent_inferred"
            item.confidence = min(item.confidence, 0.4)
        if chosen is not None:
            item.entity_key = chosen.entity
            item.attribute_key = chosen.attribute
            item.slot_class = chosen.slot_class
            if item.slot_value is None:
                item.slot_value = canonical_slot_value(source_text, chosen)
        item.negated = infer_negation(source_text)
        if item.tenant_id is None:
            item.tenant_id = self.tenant_id
        if item.agent_id is None:
            item.agent_id = self.agent_id
        if item.session_id is None:
            item.session_id = self.session_id
        return item

    def _pii_shadow(self, annotations: list[Any], processed: str) -> str | None:
        """Typed stand-in so a query with a real email can still retrieve [EMAIL]."""
        if not annotations:
            return None
        kinds = sorted({a.pii_type.value.lower() for a in annotations})
        return f"user {' '.join(kinds)} on file. {processed}"

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
        user_id: str | None = None,
    ) -> MemoryItem:
        """Store a memory.

        The keyword-only trust parameters are additive. When both
        ``entity_key`` and ``attribute_key`` are supplied, the write routes
        through the trust engine: a same-value write into the occupied slot
        corroborates the existing memory instead of duplicating it, and a
        different-value write supersedes it.

        ``user_id`` overrides the client default for this call. A scoped
        client cannot be widened to another user.
        """
        uid = self._resolve_user(user_id)
        if memory_type == MemoryType.FACTUAL:
            self._require_epistemic_source(epistemic_source, path="add")
        await self._ensure_init()
        assert self._pii is not None
        assert self._embedder is not None
        assert self._trust is not None
        store = self._require_store()

        processed, pii_annotations = self._pii.process(content)
        shadow = self._pii_shadow(pii_annotations, processed)
        embed_text = shadow or processed

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

        embedding = (await self._embedder.embed([embed_text]))[0]
        now = self.clock()
        item = MemoryItem(
            content=processed,
            embedding=embedding,
            memory_type=memory_type,
            source=source,
            metadata=metadata or {},
            event_time=event_time or now,
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
            valid_from=now,
            pii_shadow=shadow,
            user_id=uid,
            corroborated_by=[speaker_id(uid, self.session_id, self.agent_id)],
        )
        self._apply_slot(item, raw_text=content)
        self._screen_item(item)
        if item.injection_suspect:
            await self._emit(
                "injection_suspect",
                memory_id=item.id,
                user_id=uid,
                content=item.content,
            )

        if item.entity_key and item.attribute_key:
            stored, outcome = await self._trust.write(item, user_id=uid)
        else:
            stored = await store.add(item)
            outcome = "added"
            if self._audit is not None:
                await self._audit.log(
                    operation="CREATE",
                    memory_id=stored.id,
                    user_id=uid,
                    details={"memory_type": memory_type.value},
                )

        # Only freshly-stored memories get graph links; a corroboration
        # reuses the existing node.
        if (
            outcome in {"added", "superseded", "contested"}
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
        await self._emit("write", memory_id=stored.id, user_id=uid, outcome=outcome)
        return stored

    async def search(
        self,
        query: str,
        top_k: int = 10,
        memory_type: MemoryType | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        as_of: datetime | None = None,
        compose: bool = False,
        *,
        user_id: str | None = None,
        entity: str | None = None,
        min_confidence: float | None = None,
        include_third_party: bool = True,
    ) -> list[MemoryItem]:
        """Semantic + graph recall.

        Defaults to *currently valid* memories only (Epic 2): a superseded
        fact is retained for audit but no longer recalled. Pass ``as_of``
        for a time-travel query — "what did we believe at moment T?".

        ``compose=True`` hops to sibling slots of the same entity so
        "when is the Denver meeting?" can surface ``meeting/time`` after
        hitting ``meeting/location``. Mem0-style store-and-recall cannot.
        """
        uid = self._resolve_user(user_id)
        await self._ensure_init()
        assert self._embedder is not None
        assert self._retrieval is not None
        assert self._pii is not None
        moment = as_of or self.clock()
        # Embed the redacted query so a lookup containing a real email
        # still retrieves '[EMAIL]' — and so raw PII never enters the
        # vector index on the read path either.
        processed_query, _query_pii = self._pii.process(query)
        scored: list[Any]
        try:
            query_embedding = (await self._embedder.embed([processed_query]))[0]
            scored = await self._retrieval.search_scored(
                query, query_embedding, top_k=top_k * 3, user_id=uid
            )
        except Exception as exc:  # noqa: BLE001
            if not self.config.lexical_on_embed_failure:
                raise
            _logger.warning("embedding failed (%s); lexical recall", exc)
            await self._emit("embed_fallback", user_id=uid, query=query, error=str(exc))
            scored = await self._lexical_scored(processed_query, uid, top_k=top_k * 3)
        floor = self.trust_policy.relevance_floor
        # Floor is on cosine (comparable across queries), not RRF×salience
        # (which is ~0.01–0.1 and would abstain on everything).
        scored = [s for s in scored if s.cosine >= floor]
        items = [s.item for s in scored if s.item.is_valid_at(moment)]
        if memory_type is not None:
            items = [m for m in items if m.memory_type == memory_type]
        if time_range is not None:
            start, end = time_range
            items = [m for m in items if m.event_time and start <= m.event_time <= end]
        if entity is not None:
            items = [m for m in items if m.entity_key == entity]
        if min_confidence is not None:
            items = [m for m in items if m.confidence >= min_confidence]
        if not include_third_party:
            items = [m for m in items if m.epistemic_source != "third_party"]
        items = items[:top_k]
        if compose:
            items = await self._compose_siblings(items, moment, extra=top_k, user_id=uid)
        if self._audit is not None:
            score_by_id = {s.item.id: s for s in scored}
            await self._audit.log(
                operation="SEARCH",
                user_id=uid,
                details={
                    "query": query,
                    "hits": len(items),
                    "returned_ids": [m.id for m in items],
                    "as_of": moment.isoformat(),
                    "relevance_floor": floor,
                    "abstained": len(items) == 0,
                    # Recall-side observability (Epic 6): every recall logs
                    # the full score decomposition AND the decision flags
                    # as they stood at recall time — computed properties
                    # change after confirm(), so a diary of scores without
                    # the decision is not an audit.
                    "scores": {
                        mid: {
                            "final": round(s.final_score, 6),
                            "rrf": round(s.rrf_score, 6),
                            "salience": round(s.salience, 6),
                            "age_days": round(s.age_days, 3),
                            "recurrence": s.recurrence,
                            "criticality_boost": s.criticality_boost,
                            "per_graph": s.per_graph,
                            "requires_confirmation": self.trust_policy.requires_confirmation(
                                s.item
                            ),
                            "action_trusted": self.trust_policy.is_trusted(s.item),
                            "confirmed": s.item.confirmed,
                            "contested": s.item.contested,
                            "independent_corroboration": s.item.independent_corroboration,
                        }
                        for mid, s in score_by_id.items()
                        if mid in {m.id for m in items}
                    },
                },
            )
        await self._emit("recall", user_id=uid, query=query, hits=len(items))
        return items

    async def _lexical_scored(
        self,
        query: str,
        user_id: str | None,
        top_k: int,
    ) -> list[Any]:
        from contextdb.dynamics.retrieval import ScoredMemory

        store = self._require_store()
        tokens = {part for part in query.lower().split() if part}
        candidates = await store.list_memories(user_id=user_id, limit=500)
        ranked: list[tuple[int, MemoryItem]] = []
        for item in candidates:
            overlap = len(tokens & set(item.content.lower().split()))
            if overlap:
                ranked.append((overlap, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [
            ScoredMemory(
                item=item,
                final_score=float(overlap),
                rrf_score=float(overlap),
                salience=1.0,
                age_days=0.0,
                recurrence=1.0,
                criticality_boost=1.0,
                cosine=0.0,
            )
            for overlap, item in ranked[:top_k]
        ]

    async def add_fast(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACTUAL,
        metadata: dict[str, Any] | None = None,
        entity_mentions: list[str] | None = None,
        *,
        user_id: str | None = None,
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
        uid = self._resolve_user(user_id)
        await self._ensure_init()
        assert self._pii is not None
        assert self._embedder is not None
        store = self._require_store()

        processed, pii_annotations = self._pii.process(content)
        shadow = self._pii_shadow(pii_annotations, processed)
        embed_text = shadow or processed
        embedding = (await self._embedder.embed([embed_text]))[0]
        now = self.clock()
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
            pii_shadow=shadow,
            user_id=uid,
            corroborated_by=[speaker_id(uid, self.session_id, self.agent_id)],
        )
        self._apply_slot(item, raw_text=content)
        self._screen_item(item)
        # Deterministic slot → trust engine (still no LLM). Unkeyed
        # fast-writes stay a plain append so the turn path stays cheap.
        if item.entity_key and item.attribute_key and self._trust is not None:
            stored, _outcome = await self._trust.write(item, user_id=uid)
        else:
            stored = await store.add(item)
            if self._audit is not None:
                await self._audit.log(
                    operation="CREATE",
                    memory_id=stored.id,
                    user_id=uid,
                    details={"memory_type": memory_type.value, "fast": True},
                )
        await self._emit("write", memory_id=stored.id, user_id=uid, outcome="fast")
        return stored

    async def confirm(self, memory_id: str, user_id: str | None = None) -> MemoryItem:
        """Write back an explicit user confirmation (closes the verify loop).

        After the agent asked and the user said yes, this graduates the
        fact: ``confirmed=True``, first-party, high confidence, speaker
        added to ``corroborated_by``. Under any stock :class:`TrustPolicy`
        the fact then passes ``recall_for_action``.
        """
        uid = self._resolve_user(user_id)
        await self._ensure_init()
        assert self._trust is not None
        item = await self._trust.confirm(memory_id, user_id=uid)
        await self._emit("confirm", memory_id=item.id, user_id=uid)
        return item

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
                    write_generation=raw.write_generation,
                )
                # Extractor is a security boundary: slot must be supported
                # by the raw text, and extracted trust cannot exceed what
                # the raw itself would have been assigned.
                self._apply_slot(new_item, raw_text=raw.content, extractor_bound=True)
                if new_item.epistemic_source == "user_stated" and not infer_slot(
                    raw.content
                ):
                    new_item.epistemic_source = "agent_inferred"
                self._screen_item(new_item)
                saved, outcome = await self._trust.write(new_item, user_id=self.user_id)
                if outcome in {"added", "superseded", "contested"}:
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
        deleted = await self._require_store().delete(memory_id, hard=hard)
        if deleted and self._audit is not None:
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
            self._apply_slot(item, raw_text=conversation, extractor_bound=True)
            self._screen_item(item)
            if item.entity_key and item.attribute_key:
                saved, outcome = await self._trust.write(item, user_id=self.user_id)
            else:
                saved = await store.add(item)
                outcome = "added"
            if (
                outcome in {"added", "superseded", "contested"}
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
        *,
        memory_id: str | None = None,
        attribute: str | None = None,
    ) -> int:
        """Bulk-delete memories.

        Age-only forgets use a single SQL ``DELETE`` to stay O(1) in Python
        memory. Entity-scoped forgets must inspect JSON-serialised
        ``entity_mentions`` and free-text content, so they stream memories in
        500-row pages rather than loading the full table.

        ``memory_id`` deletes one row. ``entity`` + ``attribute`` deletes the
        current occupants of that slot (``forget my address``).
        """
        uid = self._resolve_user(user_id)
        await self._ensure_init()
        store = self._require_store()

        if memory_id is not None:
            removed = await store.delete(memory_id, hard=True)
            count = 1 if removed else 0
            if removed and self._audit is not None:
                await self._audit.log(
                    operation="ERASE",
                    memory_id=memory_id,
                    user_id=uid,
                    details={"single": True},
                )
            await self._emit("forget", user_id=uid, memory_id=memory_id, count=count)
            return count

        if entity is not None and attribute is not None:
            rows = await store.list_by_slot(entity, attribute, user_id=uid)
            count = 0
            for row in rows:
                if await store.delete(row.id, hard=True):
                    count += 1
                    if self._audit is not None:
                        await self._audit.log(
                            operation="ERASE",
                            memory_id=row.id,
                            user_id=uid,
                            details={"slot": f"{entity}/{attribute}"},
                        )
            await self._emit("forget", user_id=uid, entity=entity, attribute=attribute, count=count)
            return count

        # Fast path: age-only deletes lower to a single SQL statement.
        if entity is None and older_than is not None:
            cutoff = self.clock() - older_than
            n_deleted = await store.delete_older_than(
                cutoff.isoformat(), user_id=uid, hard=True
            )
            if self._audit is not None:
                await self._audit.log(
                    operation="ERASE",
                    user_id=uid,
                    details={"bulk": True, "count": n_deleted, "older_than": older_than.days},
                )
            return n_deleted

        now = self.clock()
        needle = entity.lower() if entity is not None else None
        n_deleted = 0
        async for m in store.iter_memories(user_id=uid, batch_size=500):
            matches = True
            if needle is not None:
                ents = [e.lower() for e in m.entity_mentions]
                if needle not in ents and needle not in m.content.lower():
                    matches = False
            if older_than is not None and now - m.created_at < older_than:
                matches = False
            if matches:
                await store.delete(m.id, hard=True)
                n_deleted += 1
                if self._audit is not None:
                    await self._audit.log(
                        operation="ERASE",
                        memory_id=m.id,
                        user_id=uid,
                        details={"bulk": True},
                    )
        await self._emit("forget", user_id=uid, count=n_deleted)
        return n_deleted

    async def forget_user(self, user_id: str) -> int:
        """Verifiable forgetting (Epic 7): delete EVERYTHING for a user.

        Raw, archived, and consolidated/derived memories are hard-deleted —
        deletion that leaves derived memories behind is compliance theater.
        Derived memories are found by ``metadata.consolidated_from``
        intersection (to a fixpoint), so even cross-scope derivations are
        reached. A single FORGET audit entry is written whose details carry
        the sorted deletion set and its SHA-256 hash; the entry's own
        hash-chain signature therefore covers exactly what was deleted.

        The audit log itself is append-only and is NOT deleted — the FORGET
        entry is the proof of erasure. Returns the deletion count.
        """
        await self._ensure_init()
        store = self._require_store()

        targets: dict[str, MemoryItem] = {}
        async for m in store.iter_memories(user_id=user_id, status=None, batch_size=500):
            targets[m.id] = m

        # Fixpoint sweep for derived memories (consolidated_from overlap).
        while True:
            known = set(targets)
            added = 0
            async for m in store.iter_memories(status=None, batch_size=500):
                if m.id in known:
                    continue
                derived_from = m.metadata.get("consolidated_from") or []
                if isinstance(derived_from, list) and any(
                    src in known for src in derived_from
                ):
                    targets[m.id] = m
                    added += 1
            if added == 0:
                break

        # Walk graph edges: a forgotten fact that still has neighbors is
        # how "forgotten" memories regenerate — not from the row, from
        # the graph.
        for graph in self._graphs.values():
            for memory_id in list(targets):
                try:
                    neighbors = await graph.get_neighbors(memory_id, max_results=50)
                except Exception:  # noqa: BLE001
                    continue
                for nid, _weight in neighbors:
                    if nid in targets:
                        continue
                    neighbor = await store.get_raw(nid)
                    if neighbor is not None:
                        targets[nid] = neighbor
                try:
                    await graph.remove_node(memory_id)
                except Exception:  # noqa: BLE001
                    continue

        for memory_id in targets:
            await store.delete(memory_id, hard=True)

        deleted_ids = sorted(targets)
        deletion_hash = hashlib.sha256(
            "\n".join(deleted_ids).encode("utf-8")
        ).hexdigest()
        if self._audit is not None:
            await self._audit.log(
                operation="FORGET",
                user_id=user_id,
                details={
                    "deleted_count": len(deleted_ids),
                    "deleted_ids": deleted_ids,
                    "deletion_set_hash": deletion_hash,
                },
            )
        return len(deleted_ids)

    async def verify_forgotten(self, user_id: str) -> bool:
        """Re-search and assert zero residue after :meth:`forget_user`.

        Checks, in order: (1) no rows for the user in any lifecycle state;
        (2) every id from the latest FORGET deletion set is gone from the
        table; (3) no deleted id remains searchable in the vector index.
        """
        await self._ensure_init()
        store = self._require_store()
        if await store.count_any_status(user_id) > 0:
            return False

        deleted_ids: list[str] = []
        if self._audit is not None:
            entries = await self._audit.get_history(user_id=user_id, limit=1000)
            forgets = [e for e in entries if e.operation == "FORGET"]
            if forgets:
                raw_ids = forgets[-1].details.get("deleted_ids") or []
                deleted_ids = [str(mid) for mid in raw_ids]

        for memory_id in deleted_ids:
            if await store.get_raw(memory_id) is not None:
                return False
        if deleted_ids:
            residue = set(deleted_ids) & await store.index_ids()
            if residue:
                return False
        return True

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

    async def explain(self, memory_id: str) -> MemoryExplanation:
        """Reconstruct a memory's formation + recall history (Epic 6).

        Combines the store with the audit log: the memory itself, every
        write-side entry for its id, the full supersede chain in both
        directions, and every search that surfaced it (with the score
        components logged at recall time). The audit scan is O(log size);
        that is the price of append-only tamper-evidence.
        """
        await self._ensure_init()
        store = self._require_store()
        item = await store.get_raw(memory_id)

        writes: list[dict[str, Any]] = []
        surfaced_by: list[dict[str, Any]] = []
        chain: list[str] = [memory_id]

        if self._audit is not None:
            write_entries = await self._audit.get_history(memory_id=memory_id, limit=1000)
            writes = [
                e.model_dump(mode="json")
                for e in write_entries
                if self.user_id is None or e.user_id == self.user_id
            ]

            all_entries = await self._audit.get_history(limit=10000)
            if self.user_id is not None:
                all_entries = [e for e in all_entries if e.user_id == self.user_id]
            surfaced_by = [
                e.model_dump(mode="json")
                for e in all_entries
                if e.operation == "SEARCH"
                and memory_id in (e.details.get("returned_ids") or [])
            ]

            # Supersede edges from the audit log: old_id -> new_id.
            superseded_to_new: dict[str, str] = {}
            for e in all_entries:
                if e.operation == "SUPERSEDE" and e.memory_id:
                    new_id = e.details.get("superseded_by")
                    if isinstance(new_id, str):
                        superseded_to_new[e.memory_id] = new_id

            cursor = memory_id
            while True:
                predecessor = next(
                    (old for old, new in superseded_to_new.items() if new == cursor),
                    None,
                )
                if predecessor is None or predecessor in chain:
                    break
                chain.insert(0, predecessor)
                cursor = predecessor
            cursor = memory_id
            while True:
                successor = superseded_to_new.get(cursor)
                if successor is None or successor in chain:
                    break
                chain.append(successor)
                cursor = successor

        return MemoryExplanation(
            memory_id=memory_id,
            memory=item.model_dump(mode="json") if item is not None else None,
            writes=writes,
            supersede_chain=chain,
            surfaced_by=surfaced_by,
        )

    async def _compose_siblings(
        self,
        items: list[MemoryItem],
        moment: datetime,
        extra: int = 5,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        """Hop to other current slots of the same entity.

        "when is the Denver meeting?" hits ``meeting/location``; the hop
        surfaces ``meeting/time``. Cap keeps a noisy entity from dumping
        its whole profile into the prompt.
        """
        store = self._require_store()
        seen = {m.id for m in items}
        extras: list[MemoryItem] = []
        for item in items:
            if not item.entity_key:
                continue
            for sibling in await store.list_by_entity(item.entity_key, user_id=user_id):
                if sibling.id in seen or not sibling.is_valid_at(moment):
                    continue
                seen.add(sibling.id)
                extras.append(sibling)
                if len(extras) >= extra:
                    return items + extras
        return items + extras

    async def add_fast_many(
        self,
        contents: list[str],
        *,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        """LLM-free batch write. One embed call per item, no extraction."""
        stored: list[MemoryItem] = []
        for content in contents:
            stored.append(await self.add_fast(content, user_id=user_id))
        return stored

    async def add_many(
        self,
        items: list[dict[str, Any]],
        *,
        user_id: str | None = None,
    ) -> list[MemoryItem]:
        """Batch factual writes. Each dict is ``content`` plus optional trust fields.

        Items without ``source`` take the LLM-free ``add_fast`` path so a
        40-turn dump does not call an extractor.
        """
        stored: list[MemoryItem] = []
        for raw in items:
            if isinstance(raw, str):
                stored.append(await self.add_fast(raw, user_id=user_id))
                continue
            content = str(raw["content"])
            source = raw.get("source") or raw.get("epistemic_source")
            if source is None:
                stored.append(await self.add_fast(content, user_id=user_id))
                continue
            stored.append(
                await self.add(
                    content,
                    epistemic_source=source,
                    confidence=raw.get("confidence"),
                    action_relevant=raw.get("action_relevant"),
                    entity_key=raw.get("entity") or raw.get("entity_key"),
                    attribute_key=raw.get("attribute") or raw.get("attribute_key"),
                    user_id=user_id,
                )
            )
        return stored

    async def pending_confirmations(
        self,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Facts that are action-relevant and do not yet pass the trust policy."""
        uid = self._resolve_user(user_id)
        await self._ensure_init()
        store = self._require_store()
        rows = await store.list_memories(
            user_id=uid, memory_type=MemoryType.FACTUAL, limit=limit
        )
        return [item for item in rows if self.trust_policy.requires_confirmation(item)]

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
