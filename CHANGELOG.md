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

### Fixed

- `contextdb.__version__` no longer drifts from `pyproject.toml` (was `0.1.0`
  while the project version was `0.1.1`).
- CI mypy on Python 3.12: skip NumPy stubs so NumPy 2.5's PEP 695 `type`
  aliases cannot red the job while we still type-check as 3.10.

## 0.1.1 — 2026-04-22

README rendering on PyPI; absolute GitHub asset URLs.

## 0.1.0 — 2026-04-22

Initial public release.
