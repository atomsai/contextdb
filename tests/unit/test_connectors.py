from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contextdb.connectors import (
    ConnectorConfigError,
    ConnectorCursor,
    PostgresConnector,
    PostgresConnectorConfig,
)


def config(**updates: object) -> PostgresConnectorConfig:
    values = {
        "dsn": "postgresql://localhost/source",
        "relation": "public.customer_context",
        "primary_key": "id",
        "user_id": "customer_id",
        "content": "context_text",
        "updated_at": "updated_at",
        "deleted_at": "deleted_at",
        "ssl_mode": "disable",
    }
    values.update(updates)
    return PostgresConnectorConfig(**values)  # type: ignore[arg-type]


def test_postgres_connector_rejects_unsafe_identifiers() -> None:
    with pytest.raises(ConnectorConfigError):
        PostgresConnector(config(relation="public.users; DROP TABLE users"))
    with pytest.raises(ConnectorConfigError):
        PostgresConnector(config(primary_key="id DESC"))
    with pytest.raises(ConnectorConfigError):
        PostgresConnector(config(page_size=0))


def test_connector_cursor_is_portable() -> None:
    cursor = ConnectorCursor(
        updated_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
        primary_key="customer-1",
    )
    assert cursor.as_dict() == {
        "updated_at": "2026-08-20T00:00:00+00:00",
        "primary_key": "customer-1",
    }
