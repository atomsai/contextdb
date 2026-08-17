from __future__ import annotations

from pathlib import Path

import pytest

import contextdb
from contextdb.core.config import ContextDBConfig
from contextdb.serve import create_app


def _cfg(tmp_path: Path) -> ContextDBConfig:
    return ContextDBConfig(
        storage_url=f"sqlite:///{tmp_path}/serve.db",
        embedding_model="mock",
        embedding_dim=32,
        llm_model="mock",
        llm_api_key="mock",
        enable_entity_graph=False,
        enable_multi_graph=False,
        enable_auto_link=False,
    )


@pytest.mark.asyncio
async def test_http_remember_requires_source_and_scopes_user(tmp_path: Path) -> None:
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("starlette")
    from httpx import ASGITransport, AsyncClient

    db = contextdb.init(config=_cfg(tmp_path))
    app = create_app(db, token="secret")
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            denied = await client.post("/v1/remember", json={"content": "x"})
            assert denied.status_code == 401
            missing = await client.post(
                "/v1/remember",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u1"},
                json={"content": "x"},
            )
            assert missing.status_code == 400
            ok = await client.post(
                "/v1/remember",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u1"},
                json={"content": "Thursday works", "source": "user_stated"},
            )
            assert ok.status_code == 200
            recall = await client.post(
                "/v1/recall",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u2"},
                json={"query": "Thursday"},
            )
            assert recall.status_code == 200
            assert recall.json()["memories"] == []
            pending = await client.get(
                "/v1/pending_confirmations",
                headers={"Authorization": "Bearer secret", "X-ContextDB-User": "u1"},
            )
            assert pending.status_code == 200
    finally:
        await db.close()
        del httpx
