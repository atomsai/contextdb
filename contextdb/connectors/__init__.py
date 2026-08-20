"""Portable connector contracts and local reference adapters."""

from contextdb.connectors.base import (
    ConnectorConfigError,
    ConnectorCursor,
    ConnectorError,
    ConnectorReader,
    ConnectorRecord,
    ConnectorSourceError,
)
from contextdb.connectors.postgres import (
    PostgresConnector,
    PostgresConnectorConfig,
)

__all__ = [
    "ConnectorConfigError",
    "ConnectorCursor",
    "ConnectorError",
    "ConnectorReader",
    "ConnectorRecord",
    "ConnectorSourceError",
    "PostgresConnector",
    "PostgresConnectorConfig",
]
