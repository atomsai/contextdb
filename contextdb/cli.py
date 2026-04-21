"""Tiny command-line interface for ContextDB.

The CLI is intentionally minimal — enough to add/search/stats/export/import
from a shell, but not a replacement for the Python SDK. We use argparse
rather than click/typer so the library doesn't grow another dependency.

Usage:
    contextdb add "My birthday is March 5"
    contextdb search "when is my birthday"
    contextdb stats
    contextdb export dump.json
    contextdb import dump.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from contextdb import init
from contextdb.core.models import MemoryType


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contextdb")
    parser.add_argument("--storage-url", default=None, help="Override storage URL.")
    parser.add_argument("--user-id", default=None, help="Optional user_id scope.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a memory.")
    p_add.add_argument("content")
    p_add.add_argument(
        "--type",
        choices=[t.value for t in MemoryType],
        default=MemoryType.FACTUAL.value,
    )
    p_add.add_argument("--source", default="")

    p_search = sub.add_parser("search", help="Search memories.")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)

    sub.add_parser("stats", help="Print store statistics.")

    p_export = sub.add_parser("export", help="Export the store to JSON.")
    p_export.add_argument("path")

    p_import = sub.add_parser("import", help="Import a ContextDB JSON dump.")
    p_import.add_argument("path")

    return parser


async def _run(args: argparse.Namespace) -> int:
    kwargs: dict[str, Any] = {}
    if args.storage_url:
        kwargs["storage_url"] = args.storage_url
    client = init(user_id=args.user_id, **kwargs)
    try:
        if args.command == "add":
            item = await client.add(
                content=args.content,
                memory_type=MemoryType(args.type),
                source=args.source,
            )
            print(json.dumps({"id": item.id, "content": item.content}, indent=2))
        elif args.command == "search":
            hits = await client.search(args.query, top_k=args.top_k)
            print(
                json.dumps(
                    [{"id": m.id, "content": m.content} for m in hits], indent=2
                )
            )
        elif args.command == "stats":
            stats = await client.stats()
            print(json.dumps(stats, indent=2, default=str))
        elif args.command == "export":
            from contextdb.utils.migrations import JSONExporter

            exporter = JSONExporter(client)
            count = await exporter.export(Path(args.path))
            print(f"Exported {count} memories to {args.path}")
        elif args.command == "import":
            from contextdb.utils.migrations import JSONImporter

            importer = JSONImporter(client)
            count = await importer.import_path(Path(args.path))
            print(f"Imported {count} memories from {args.path}")
        else:
            return 1
        return 0
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
