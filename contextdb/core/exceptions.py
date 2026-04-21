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
