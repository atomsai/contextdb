"""Translate the SQLite SQL the graphs and audit log already speak into Postgres."""

from __future__ import annotations

import re

_REPLACE_PK = {
    "semantic_edges": "(source_id, target_id)",
    "temporal_edges": "(source_id, target_id)",
    "causal_edges": "(source_id, target_id)",
    "memory_entity_edges": "(memory_id, entity_id)",
}

_REPLACE_SET = {
    "semantic_edges": (
        "weight = EXCLUDED.weight, metadata = EXCLUDED.metadata, "
        "created_at = EXCLUDED.created_at"
    ),
    "temporal_edges": (
        "relation = EXCLUDED.relation, weight = EXCLUDED.weight, "
        "time_diff_seconds = EXCLUDED.time_diff_seconds, "
        "metadata = EXCLUDED.metadata, created_at = EXCLUDED.created_at"
    ),
    "causal_edges": (
        "relation = EXCLUDED.relation, weight = EXCLUDED.weight, "
        "reasoning = EXCLUDED.reasoning, metadata = EXCLUDED.metadata, "
        "created_at = EXCLUDED.created_at"
    ),
    "memory_entity_edges": (
        "relation = EXCLUDED.relation, created_at = EXCLUDED.created_at"
    ),
}


def qmark_to_dollar(sql: str) -> str:
    """Turn ``?`` placeholders into ``$1, $2, ...``."""
    parts = sql.split("?")
    if len(parts) == 1:
        return sql
    out: list[str] = []
    for i, part in enumerate(parts[:-1]):
        out.append(part)
        out.append(f"${i + 1}")
    out.append(parts[-1])
    return "".join(out)


def translate_sqlite_sql(sql: str) -> str:
    """Best-effort SQLite → Postgres for the statements this package emits."""
    s = sql.strip()
    ignore = re.match(r"INSERT\s+OR\s+IGNORE\s+INTO\s+(\w+)", s, re.IGNORECASE)
    if ignore:
        s = re.sub(r"INSERT\s+OR\s+IGNORE", "INSERT", s, count=1, flags=re.IGNORECASE)
        s = s.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
        return qmark_to_dollar(s)
    replace = re.match(r"INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)", s, re.IGNORECASE)
    if replace:
        table = replace.group(1)
        s = re.sub(r"INSERT\s+OR\s+REPLACE", "INSERT", s, count=1, flags=re.IGNORECASE)
        pk = _REPLACE_PK.get(table)
        sets = _REPLACE_SET.get(table)
        if pk and sets:
            s = s.rstrip().rstrip(";") + f" ON CONFLICT {pk} DO UPDATE SET {sets}"
        return qmark_to_dollar(s)
    return qmark_to_dollar(s)


def split_script(script: str) -> list[str]:
    return [part.strip() for part in script.split(";") if part.strip()]
