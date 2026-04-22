"""ContextDB benchmark suite.

Runs six benchmarks that isolate storage/retrieval performance from external
API latency by using :class:`MockEmbedding` (dim=1536) and ``MockLLM``. No API
keys required.

Usage:
    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from contextdb import ContextDB, ContextDBConfig, MemoryItem, MemoryType
from contextdb.privacy.pii_detector import PIIDetector
from contextdb.store.sqlite_store import SQLiteStore
from contextdb.store.vector_index import NumpyIndex
from contextdb.utils.embeddings import MockEmbedding

EMBED_DIM = 1536


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}ms"


def _percentiles(samples: list[float]) -> tuple[float, float, float, float]:
    """Return (p50, p95, p99, mean) in seconds."""
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    p50 = sorted_samples[max(0, int(n * 0.50) - 1)]
    p95 = sorted_samples[max(0, int(n * 0.95) - 1)]
    p99 = sorted_samples[max(0, int(n * 0.99) - 1)]
    return p50, p95, p99, statistics.fmean(samples)


def _print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _make_config(db_path: Path) -> ContextDBConfig:
    return ContextDBConfig(
        storage_url=f"sqlite:///{db_path}",
        embedding_model="mock",
        embedding_dim=EMBED_DIM,
        llm_model="mock",
        enable_audit=False,
        enable_auto_link=False,
    )


async def bench_write_throughput(db_path: Path) -> dict[str, float]:
    """1000 sequential add()s against a fresh store."""
    _print_header("1. Write Throughput (1000 memories)")
    config = _make_config(db_path)
    async with ContextDB(config) as db:
        start = time.perf_counter()
        for i in range(1000):
            await db.add(f"memory {i}: benchmark write throughput sample")
        elapsed = time.perf_counter() - start

    throughput = 1000 / elapsed
    per_op = elapsed / 1000
    print("Total memories written:  1,000")
    print(f"Total time:              {elapsed:.2f}s")
    print(f"Throughput:              {throughput:.0f} writes/sec")
    print(f"Average per write:       {_fmt_ms(per_op)}")
    return {"throughput": throughput, "per_op": per_op}


async def bench_search_latency(db_path: Path) -> dict[str, float]:
    """p50/p95/p99 search latency across 100 queries over 1K memories."""
    _print_header("2. Search Latency (100 queries over 1,000 memories)")
    config = _make_config(db_path)
    async with ContextDB(config) as db:
        for i in range(1000):
            await db.add(f"memory {i}: facts about topic {i % 20}")

        samples: list[float] = []
        for i in range(100):
            query = f"topic {i % 20}"
            start = time.perf_counter()
            await db.search(query, top_k=10)
            samples.append(time.perf_counter() - start)

    p50, p95, p99, mean = _percentiles(samples)
    print(f"p50:    {_fmt_ms(p50)}")
    print(f"p95:    {_fmt_ms(p95)}")
    print(f"p99:    {_fmt_ms(p99)}")
    print(f"Mean:   {_fmt_ms(mean)}")
    return {"p50": p50, "p95": p95, "p99": p99, "mean": mean}


async def bench_search_latency_vs_scale() -> None:
    """Add + search latency at increasing store sizes."""
    _print_header("3. Search Latency vs Scale")
    sizes = [100, 500, 1000, 5000]
    print(f"{'Memories':>10} | {'Add (ms/op)':>12} | {'Search p50':>12} | {'Search p95':>12}")
    print("-" * 60)
    for n in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / f"scale_{n}.db"
            async with ContextDB(_make_config(db_path)) as db:
                add_start = time.perf_counter()
                for i in range(n):
                    await db.add(f"memory {i}: topic {i % 20} detail {i}")
                add_per = (time.perf_counter() - add_start) / n

                samples: list[float] = []
                for i in range(50):
                    q_start = time.perf_counter()
                    await db.search(f"topic {i % 20}", top_k=10)
                    samples.append(time.perf_counter() - q_start)
                p50, p95, _, _ = _percentiles(samples)
        print(
            f"{n:>10,} | {_fmt_ms(add_per):>12} | "
            f"{_fmt_ms(p50):>12} | {_fmt_ms(p95):>12}"
        )


def bench_pii_throughput() -> dict[str, float]:
    """1000 texts through redact + correctness spot-check."""
    _print_header("4. PII Detection & Redaction (1000 texts)")
    detector = PIIDetector(action="redact")
    sample = (
        "Contact me at test@example.com or 555-123-4567. SSN: 123-45-6789"
    )

    start = time.perf_counter()
    total_pii = 0
    for _ in range(1000):
        _, anns = detector.process(sample)
        total_pii += len(anns)
    elapsed = time.perf_counter() - start

    throughput = 1000 / elapsed
    avg_pii = total_pii / 1000
    redacted, anns = detector.process(sample)
    found_types = sorted({a.pii_type.value for a in anns})

    print(f"Throughput:              {throughput:,.0f} texts/sec")
    print(f"Average per text:        {_fmt_ms(elapsed / 1000)}")
    print(f"Average PII per text:    {avg_pii:.1f}")
    print()
    print("Correctness:")
    print(f"  Input:    {sample}")
    print(f"  Redacted: {redacted}")
    print(f"  PII found: {found_types}")
    return {"throughput": throughput, "per_text": elapsed / 1000}


def bench_vector_index() -> dict[str, float]:
    """10K × 1536d × 100 queries with self-retrieval accuracy check."""
    _print_header("5. Vector Index Performance (10K vectors, 1536d)")
    n = 10_000
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, EMBED_DIM)).astype(np.float32)
    ids = [f"v{i}" for i in range(n)]

    build_start = time.perf_counter()
    idx = NumpyIndex(dimension=EMBED_DIM)
    idx.add(ids, vectors)
    build_time = time.perf_counter() - build_start

    samples: list[float] = []
    correct = 0
    for i in range(100):
        target = i * 97 % n
        q = vectors[target]
        start = time.perf_counter()
        hits = idx.search(q, top_k=1)
        samples.append(time.perf_counter() - start)
        if hits and hits[0][0] == ids[target]:
            correct += 1

    p50, p95, _, _ = _percentiles(samples)
    accuracy = correct / 100
    print(f"Vectors indexed:         {n:,}")
    print(f"Dimensions:              {EMBED_DIM:,}")
    print(f"Index build time:        {build_time:.3f}s ({n / build_time:,.0f} vectors/sec)")
    print(f"Search p50:              {_fmt_ms(p50)}")
    print(f"Search p95:              {_fmt_ms(p95)}")
    print(f"Self-retrieval accuracy: {'PASS' if accuracy == 1.0 else f'FAIL ({accuracy:.0%})'}")
    return {"build_time": build_time, "p50": p50, "p95": p95, "accuracy": accuracy}


async def bench_customer_support_scenario(db_path: Path) -> dict[str, float]:
    """50 memories + searches + PII redaction roundtrip."""
    _print_header("6. End-to-End Scenario: Customer Support Agent")
    config = _make_config(db_path)
    async with ContextDB(config) as db:
        scenarios = [
            "Customer Alice reported AC leak on June 3 at 3pm",
            "Technician Bob scheduled for June 5 morning visit",
            "Replaced compressor on unit #A7; warranty noted",
            "Customer requested follow-up on humidity concerns",
            "Billing question: invoice #INV-2025-042 was updated",
        ]
        add_start = time.perf_counter()
        for i in range(50):
            text = scenarios[i % len(scenarios)] + f" — case {i}"
            mt = MemoryType.EXPERIENTIAL if i % 2 else MemoryType.FACTUAL
            await db.add(text, memory_type=mt)
        add_elapsed = time.perf_counter() - add_start

        search_samples: list[float] = []
        for q in [
            "AC leak customer",
            "warranty compressor",
            "humidity follow-up",
            "invoice update",
            "technician visit",
        ] * 4:
            s = time.perf_counter()
            await db.search(q, top_k=5)
            search_samples.append(time.perf_counter() - s)

        p50, _, _, mean = _percentiles(search_samples)

        pii_start = time.perf_counter()
        input_text = (
            "Customer John Smith, email john@acme.com, "
            "SSN 123-45-6789 called about billing"
        )
        stored = await db.add(input_text)
        pii_elapsed = time.perf_counter() - pii_start
        redacted_ok = "john@acme.com" not in stored.content and "[EMAIL]" in stored.content

    print(f"Add 50 memories:         {add_elapsed:.2f}s ({50 / add_elapsed:.0f}/sec)")
    print(f"Search p50:              {_fmt_ms(p50)}")
    print(f"Search mean:             {_fmt_ms(mean)}")
    print(f"PII add + redact:        {_fmt_ms(pii_elapsed)}")
    print(f"PII correctly redacted:  {'PASS' if redacted_ok else 'FAIL'}")
    print()
    print("Stored content verification:")
    print(f"  Input:  {input_text}")
    print(f"  Stored: {stored.content}")
    return {
        "add_rate": 50 / add_elapsed,
        "search_p50": p50,
        "pii_redacted": float(redacted_ok),
    }


async def _amain() -> int:
    print("ContextDB Benchmark Suite")
    print(f"MockEmbedding(dim={EMBED_DIM}), MockLLM, NumpyIndex, SQLite")
    print("No API keys required.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        await bench_write_throughput(tmp_path / "write.db")
        await bench_search_latency(tmp_path / "search.db")
        await bench_search_latency_vs_scale()
        bench_pii_throughput()
        bench_vector_index()
        await bench_customer_support_scenario(tmp_path / "cs.db")

    print()
    print("=" * 70)
    print("  All benchmarks complete.")
    print("=" * 70)
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())


# Silence unused imports referenced only by type checkers or examples.
_ = (MemoryItem, MockEmbedding, SQLiteStore)
