"""Helpers for tests that need a real Postgres server.

Set ``CONTEXTDB_TEST_POSTGRES_URL`` to a database the test user may create
and drop databases on (the tests create a fresh database each, because the
hash-chained audit log is global to a database — a shared test database
would accumulate entries and fail whole-chain verification).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest


def postgres_base_url() -> str:
    url = os.environ.get("CONTEXTDB_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("set CONTEXTDB_TEST_POSTGRES_URL to run Postgres tests")
    return url


@asynccontextmanager
async def fresh_pg_database() -> AsyncIterator[str]:
    """Yield a URL for a freshly-created database; dropped on exit."""
    base = postgres_base_url()
    try:
        import asyncpg
    except ImportError:
        pytest.skip("asyncpg not installed (pip install 'pycontextdb[postgres]')")
    name = f"contextdb_test_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(base)
    try:
        await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()
    url = f"{base.rsplit('/', 1)[0]}/{name}"
    try:
        yield url
    finally:
        drop = await asyncpg.connect(base)
        try:
            await drop.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await drop.close()
