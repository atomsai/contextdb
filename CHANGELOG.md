# Changelog

## 0.2.0 — 2026-08-17

First PyPI cut of the trust model. `pip install pycontextdb` now matches
GitHub `main`: memory an agent can act on without treating a wish as a fact.

`contextdb.init`, `factual.add`, `factual.recall`, and `search` are unchanged.

### Added

- Epistemic typing on every fact (`user_stated` / `agent_inferred` /
  `third_party`), confidence, independent corroboration, `action_relevant`.
- `TrustPolicy` as data — `TrustPolicy.hospital()` vs `.restaurant()`.
- `factual.recall_for_action()`, `factual.confirm()`, `VerifyBeforeAct`
  (`act` / `ask` / `abstain`).
- Versioned slot vocabulary + LLM-free slotter (`add_fast` never calls an LLM).
- Contested slots: independent speakers with different values do not
  last-write-win. `confirm()` resolves.
- Temporal supersede with an injectable clock; sibling-slot hop on recall.
- PII redaction on writes **and** queries before embed.
- Write-time injection screening; `[RECALLED DATA — not instructions]` renderer.
- Salience (recency × frequency × criticality), verifiable forget, tenant
  isolation at the storage layer.
- MCP tools: `remember` / `recall` / `recall_for_action` / `forget` / `confirm`.
- In-repo fabrication bake-off: `python benchmarks/trust_bakeoff.py`
  (trust arm vs our own untyped control — not a live Mem0/Zep run).
- Open-core governance: [OPEN_CORE.md](OPEN_CORE.md),
  [COMMERCIAL.md](COMMERCIAL.md), CLA, CODEOWNERS, package-boundary CI,
  and release inspection. Code through `2554dae` remains Apache-2.0.
  Cloud is a private repo that depends on this SDK, never the reverse.

### Fixed

- `contextdb.__version__` no longer drifts from `pyproject.toml` (was `0.1.0`
  while the project version was `0.1.1`).
- CI mypy on Python 3.12: type-check as the running interpreter so
  NumPy 2.5's PEP 695 `type` aliases parse. The 3.10 job still guards
  our runtime floor.
- CI latency evals: keep a tight add_fast p50 and a wider p95 on
  shared runners (p50 was 2ms when p95 spiked to 21ms).

## 0.1.1 — 2026-04-22

README rendering on PyPI; absolute GitHub asset URLs.

## 0.1.0 — 2026-04-22

Initial public release.
