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
