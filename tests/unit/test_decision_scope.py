"""Decision-layer scoping and PII correctness (VerifyBeforeAct + audit log)."""

from __future__ import annotations

from pathlib import Path

import pytest

import contextdb
from contextdb.core.config import ContextDBConfig
from contextdb.core.exceptions import ConfigError
from contextdb.integrations.act import VerifyBeforeAct


def _cfg(tmp_path: Path, name: str = "decide.db", **overrides: object) -> ContextDBConfig:
    base: dict[str, object] = {
        "storage_url": f"sqlite:///{tmp_path}/{name}",
        "embedding_model": "mock",
        "embedding_dim": 32,
        "llm_model": "mock",
        "llm_api_key": "mock",
        "enable_entity_graph": False,
        "enable_multi_graph": False,
        "enable_auto_link": False,
        "enable_audit": True,
        "pii_action": "redact",
    }
    base.update(overrides)
    return ContextDBConfig(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_decide_scopes_per_call_user_on_shared_client(tmp_path: Path) -> None:
    """A shared client + decide(user_id=...) must only see that user's facts."""
    db = contextdb.init(config=_cfg(tmp_path))
    try:
        await db.factual.add(
            "The deploy window is Friday 2am",
            source="user_stated",
            confidence=0.95,
            action_relevant=True,
            entity="deploy",
            attribute="window",
            user_id="alice",
        )
        await db.factual.add(
            "Bob's deploy window is Sunday 3am",
            source="user_stated",
            confidence=0.95,
            action_relevant=True,
            entity="deploy",
            attribute="window",
            user_id="bob",
        )
        gate = VerifyBeforeAct(db)
        decision = await gate.decide("deploy window", user_id="alice")
        assert decision.kind == "act"
        assert decision.memories
        assert all(m.user_id == "alice" for m in decision.memories)
        assert all("Sunday" not in m.content for m in decision.memories)

        bob = await gate.decide("deploy window", user_id="bob")
        assert all(m.user_id == "bob" for m in bob.memories)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_confirm_pending_passes_user_scope(tmp_path: Path) -> None:
    """confirm_pending(user_id=...) flows into confirm(); a scoped client
    still rejects a foreign user_id."""
    db = contextdb.init(config=_cfg(tmp_path))
    try:
        wish = await db.factual.add(
            "I'd like to come in Thursday",
            source="user_stated",
            confidence=0.5,
            action_relevant=True,
            entity="caller",
            attribute="preferred_visit_day",
            user_id="alice",
        )
        gate = VerifyBeforeAct(db)
        confirmed = await gate.confirm_pending([wish.id], user_id="alice")
        assert confirmed[0].confirmed is True
    finally:
        await db.close()

    scoped = contextdb.init(user_id="alice", config=_cfg(tmp_path, name="scoped.db"))
    try:
        gate = VerifyBeforeAct(scoped)
        with pytest.raises(ConfigError, match="cannot operate as"):
            await gate.decide("anything", user_id="bob")
        with pytest.raises(ConfigError, match="cannot operate as"):
            await gate.confirm_pending(["some-id"], user_id="bob")
    finally:
        await scoped.close()


@pytest.mark.asyncio
async def test_decide_audit_log_never_stores_raw_pii_query(tmp_path: Path) -> None:
    """The DECIDE audit entry must carry the redacted query — the chain is
    append-only, so raw PII there could never be removed."""
    db = contextdb.init(user_id="alice", config=_cfg(tmp_path))
    try:
        gate = VerifyBeforeAct(db)
        await gate.decide("what is on file for jane.doe@example.com?")
        assert db.audit is not None
        entries = [e for e in await db.audit.get_history() if e.operation == "DECIDE"]
        assert entries
        for entry in entries:
            assert "jane.doe@example.com" not in str(entry.details)
            assert "[EMAIL]" in entry.details["query"]
        assert await db.audit.verify_chain()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_search_audit_log_never_stores_raw_pii_query(tmp_path: Path) -> None:
    """SEARCH audit entries persist the PII-processed query form only."""
    db = contextdb.init(user_id="alice", config=_cfg(tmp_path))
    try:
        await db.factual.add("My email is jane.doe@example.com", source="user_stated")
        await db.search("please look up jane.doe@example.com")
        assert db.audit is not None
        entries = [e for e in await db.audit.get_history() if e.operation == "SEARCH"]
        assert entries
        for entry in entries:
            assert "jane.doe@example.com" not in str(entry.details)
            assert "[EMAIL]" in entry.details["query"]
        assert await db.audit.verify_chain()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_deferred_read_audit_emits_redacted_details_and_keeps_write_audit(
    tmp_path: Path,
) -> None:
    db = contextdb.init(
        user_id="alice",
        config=_cfg(tmp_path, enable_read_audit=False),
    )
    recalls: list[dict[str, object]] = []

    async def capture(_event: str, payload: dict[str, object]) -> None:
        recalls.append(payload)

    db.on("read_audit", capture)
    try:
        stored = await db.factual.add(
            "My email is jane.doe@example.com",
            source="user_stated",
        )
        await db.search("please look up jane.doe@example.com")
        assert db.audit is not None
        entries = await db.audit.get_history()
        assert any(entry.operation == "CREATE" for entry in entries)
        assert all(entry.operation != "SEARCH" for entry in entries)
        assert await db.audit.verify_chain()

        assert len(recalls) == 1
        details = recalls[0]["audit_details"]
        assert isinstance(details, dict)
        assert details["query"] == "please look up [EMAIL]"
        assert "jane.doe@example.com" not in str(details)
        assert stored.id in details["returned_ids"]
        assert stored.id in details["scores"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_deferred_read_audit_sink_failure_fails_recall(
    tmp_path: Path,
) -> None:
    db = contextdb.init(
        user_id="alice",
        config=_cfg(tmp_path, enable_read_audit=False),
    )

    async def unavailable(_event: str, _payload: dict[str, object]) -> None:
        raise RuntimeError("durable read audit unavailable")

    db.on("read_audit", unavailable)
    try:
        with pytest.raises(RuntimeError, match="durable read audit unavailable"):
            await db.search("anything")
        assert db.audit is not None
        assert all(
            entry.operation != "SEARCH"
            for entry in await db.audit.get_history()
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_encrypt_pii_action_fails_closed_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pii_action='encrypt' without a key must fail at first use, not
    silently store plaintext annotation originals."""
    monkeypatch.delenv("CONTEXTDB_PII_KEY", raising=False)
    db = contextdb.init(
        user_id="alice",
        config=_cfg(tmp_path, name="enc.db", pii_action="encrypt", pii_encryption_key=None),
    )
    try:
        with pytest.raises(ConfigError, match="encrypt"):
            await db.factual.add("email me at foo@bar.com", source="user_stated")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_encrypt_pii_action_roundtrip_with_key(tmp_path: Path) -> None:
    """With a key, encrypt mode stores ciphertext originals end to end."""
    db = contextdb.init(
        user_id="alice",
        config=_cfg(
            tmp_path, name="enc2.db", pii_action="encrypt", pii_encryption_key="k" * 16
        ),
    )
    try:
        item = await db.factual.add(
            "email me at foo@bar.com", source="user_stated"
        )
        assert "[EMAIL]" in item.content
        assert item.pii_annotations
        assert item.pii_annotations[0].original != "foo@bar.com"
        assert "foo@bar.com" not in item.pii_annotations[0].original
    finally:
        await db.close()
