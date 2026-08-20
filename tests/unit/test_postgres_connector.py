from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pytest

from contextdb.connectors import (
    ConnectorCursor,
    PostgresConnector,
    PostgresConnectorConfig,
)

PG_DSN = os.environ.get("CONTEXTDB_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not PG_DSN,
    reason="CONTEXTDB_TEST_POSTGRES_URL not set; needs real PostgreSQL",
)


@pytest.mark.asyncio
async def test_postgres_connector_pages_by_timestamp_and_key() -> None:
    import asyncpg

    schema = f"connector_{time.time_ns()}"
    connection = await asyncpg.connect(PG_DSN)
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    try:
        await connection.execute(
            f"""
            CREATE TABLE "{schema}".customer_context (
                id INTEGER PRIMARY KEY,
                customer_id TEXT NOT NULL,
                context_text TEXT,
                updated_at TIMESTAMPTZ NOT NULL,
                deleted_at TIMESTAMPTZ
            )
            """
        )
        at = datetime(2026, 8, 20, tzinfo=timezone.utc)
        await connection.executemany(
            f"""
            INSERT INTO "{schema}".customer_context
            VALUES ($1, $2, $3, $4, $5)
            """,
            [
                (1, "user-1", "Tuesday", at, None),
                (10, "user-10", "Friday", at, None),
                (2, "user-2", None, at, at),
            ],
        )
        connector = PostgresConnector(
            PostgresConnectorConfig(
                dsn=PG_DSN,
                relation=f"{schema}.customer_context",
                primary_key="id",
                user_id="customer_id",
                content="context_text",
                updated_at="updated_at",
                deleted_at="deleted_at",
                page_size=2,
                ssl_mode="disable",
            )
        )
        await connector.validate()
        preview = await connector.preview()
        assert [record.source_key for record in preview] == ["1", "10", "2"]
        assert preview[-1].deleted is True
        remaining = [
            record
            async for page in connector.records(
                cursor=ConnectorCursor(at, "10"),
            )
            for record in page
        ]
        assert [record.source_key for record in remaining] == ["2"]
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()
