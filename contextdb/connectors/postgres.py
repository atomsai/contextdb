"""Minimal local PostgreSQL/Supabase connector reference."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from contextdb.connectors.base import (
    ConnectorConfigError,
    ConnectorCursor,
    ConnectorRecord,
    ConnectorSourceError,
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _quoted(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ConnectorConfigError(
            f"invalid PostgreSQL identifier: {identifier!r}"
        )
    return f'"{identifier}"'


def _relation(value: str) -> str:
    parts = value.split(".")
    if len(parts) not in {1, 2}:
        raise ConnectorConfigError(
            "relation must be a table/view name with optional schema"
        )
    return ".".join(_quoted(part) for part in parts)


@dataclass(frozen=True)
class PostgresConnectorConfig:
    dsn: str
    relation: str
    primary_key: str
    user_id: str
    content: str
    updated_at: str
    deleted_at: str | None = None
    page_size: int = 100
    ssl_mode: str = "require"

    def validate(self) -> None:
        if urlparse(self.dsn).scheme not in {"postgres", "postgresql"}:
            raise ConnectorConfigError("dsn must use postgres/postgresql")
        if self.ssl_mode not in {
            "disable",
            "require",
            "verify-ca",
            "verify-full",
        }:
            raise ConnectorConfigError("unsupported PostgreSQL ssl_mode")
        if not 1 <= self.page_size <= 1_000:
            raise ConnectorConfigError("page_size must be between 1 and 1000")
        _relation(self.relation)
        for identifier in (
            self.primary_key,
            self.user_id,
            self.content,
            self.updated_at,
        ):
            _quoted(identifier)
        if self.deleted_at:
            _quoted(self.deleted_at)


class PostgresConnector:
    """Read mapped records locally; hosts own retries, secrets, and writes."""

    def __init__(self, config: PostgresConnectorConfig) -> None:
        config.validate()
        self.config = config

    async def _connect(self) -> Any:
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ConnectorConfigError(
                "Postgres connector requires pycontextdb[postgres]"
            ) from exc
        try:
            return await asyncpg.connect(
                self.config.dsn,
                ssl=self.config.ssl_mode,
                timeout=10,
                command_timeout=30,
            )
        except Exception as exc:
            raise ConnectorSourceError(
                "PostgreSQL source is unavailable"
            ) from exc

    async def validate(self) -> None:
        connection = await self._connect()
        try:
            await connection.fetchval(
                f"SELECT 1 FROM {_relation(self.config.relation)} LIMIT 1"
            )
        except Exception as exc:
            raise ConnectorSourceError(
                "configured relation is unavailable"
            ) from exc
        finally:
            await connection.close()

    async def preview(self, *, limit: int = 50) -> list[ConnectorRecord]:
        output: list[ConnectorRecord] = []
        async for page in self.records(limit=limit):
            output.extend(page)
        return output

    async def records(
        self,
        *,
        cursor: ConnectorCursor | None = None,
        limit: int = 100_000,
    ) -> AsyncIterator[list[ConnectorRecord]]:
        if not 1 <= limit <= 1_000_000:
            raise ConnectorConfigError("limit must be between 1 and 1000000")
        connection = await self._connect()
        current = cursor
        remaining = limit
        try:
            while remaining > 0:
                page_limit = min(self.config.page_size, remaining)
                rows = await self._fetch_page(
                    connection,
                    cursor=current,
                    limit=page_limit,
                )
                if not rows:
                    break
                records = [self._record(row) for row in rows]
                yield records
                last = records[-1]
                current = ConnectorCursor(
                    updated_at=last.updated_at,
                    primary_key=last.source_key,
                )
                remaining -= len(records)
                if len(records) < page_limit:
                    break
        finally:
            await connection.close()

    async def _fetch_page(
        self,
        connection: Any,
        *,
        cursor: ConnectorCursor | None,
        limit: int,
    ) -> list[Any]:
        config = self.config
        relation = _relation(config.relation)
        primary_key = _quoted(config.primary_key)
        user_id = _quoted(config.user_id)
        content = _quoted(config.content)
        updated_at = _quoted(config.updated_at)
        deleted_at = (
            _quoted(config.deleted_at)
            if config.deleted_at is not None
            else None
        )
        effective_updated = (
            f"GREATEST({updated_at}, COALESCE({deleted_at}, {updated_at}))"
            if deleted_at
            else updated_at
        )
        deleted_select = (
            f"{deleted_at} AS __deleted_at"
            if deleted_at
            else "NULL::timestamptz AS __deleted_at"
        )
        where = ""
        params: list[Any] = []
        if cursor is not None:
            where = (
                f"WHERE ({effective_updated} > $1 OR "
                f"({effective_updated} = $1 AND {primary_key}::text > $2))"
            )
            params.extend([cursor.updated_at, cursor.primary_key])
        params.append(limit)
        query = f"""
            SELECT
                {primary_key}::text AS __source_key,
                {user_id}::text AS __user_id,
                {content}::text AS __content,
                {effective_updated} AS __effective_updated,
                {deleted_select}
            FROM {relation}
            {where}
            ORDER BY {effective_updated}, {primary_key}::text
            LIMIT ${len(params)}
        """
        try:
            return list(await connection.fetch(query, *params))
        except Exception as exc:
            raise ConnectorSourceError(
                "failed to read PostgreSQL source"
            ) from exc

    def _record(self, row: Any) -> ConnectorRecord:
        updated_at = row["__effective_updated"]
        if not isinstance(updated_at, datetime):
            raise ConnectorSourceError(
                "updated_at mapping must produce a timestamp"
            )
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        source_key = str(row["__source_key"])
        user_id = str(row["__user_id"])
        content = str(row["__content"] or "")
        deleted = row["__deleted_at"] is not None
        if not source_key or not user_id:
            raise ConnectorSourceError(
                "source primary key and user_id must be non-empty"
            )
        if not deleted and not content:
            raise ConnectorSourceError(
                "source content must be non-empty"
            )
        return ConnectorRecord(
            source_key=source_key,
            source_version=updated_at.isoformat(),
            user_id=user_id,
            content=content,
            updated_at=updated_at,
            deleted=deleted,
            provenance={
                "provider": "postgres",
                "relation": self.config.relation,
                "source_key": source_key,
                "source_version": updated_at.isoformat(),
            },
        )
