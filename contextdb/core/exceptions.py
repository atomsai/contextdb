"""Exception hierarchy for ContextDB.

All exceptions raised by ContextDB derive from :class:`ContextDBError`, so
callers can catch everything with a single ``except`` clause when they want to.
Specific subclasses exist for the common failure modes so callers can handle
them individually where it matters.
"""

from __future__ import annotations


class ContextDBError(Exception):
    """Base class for all ContextDB errors."""


class MemoryNotFoundError(ContextDBError):
    """Raised when a memory lookup by id returns no result."""


class StorageError(ContextDBError):
    """Raised when the underlying storage backend fails."""


class StaleReadError(StorageError):
    """Raised when a store cannot satisfy a requested consistency floor."""

    def __init__(
        self,
        *,
        required_version: int | None,
        current_version: int,
        required_wal_lsn: str | None = None,
        current_wal_lsn: str | None = None,
    ) -> None:
        self.required_version = required_version
        self.current_version = current_version
        self.required_wal_lsn = required_wal_lsn
        self.current_wal_lsn = current_wal_lsn
        super().__init__(
            "memory consistency floor is not available "
            f"(required_version={required_version}, "
            f"current_version={current_version}, "
            f"required_wal_lsn={required_wal_lsn}, "
            f"current_wal_lsn={current_wal_lsn})"
        )


class PrivacyError(ContextDBError):
    """Raised when a privacy constraint is violated (PII handling, retention)."""


class ConfigError(ContextDBError):
    """Raised when ContextDB is misconfigured."""


class SourceRequiredError(ConfigError):
    """Raised when a write omits epistemic ``source`` and the host requires it."""


class UnauthorizedError(ContextDBError):
    """Raised when a host API request fails configured authentication.

    The HTTP layer maps this to 401; every other :class:`ContextDBError`
    maps to 400.
    """


class ScopeConflictError(ContextDBError):
    """Raised when a request carries a user scope that conflicts with the
    authenticated scope (or two request-supplied scopes disagree).

    Silently picking one of two conflicting scopes is how cross-user writes
    happen, so the host APIs reject the request instead (HTTP 400).
    """
