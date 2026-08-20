## Summary

<!-- What changed, and why. Evals win over prose. -->

## Open-core classification

Select exactly one and justify placement against `OPEN_CORE.md`.

- [ ] No capability change — docs/tests/build/maintenance only.
- [ ] Semantic — portable offline correctness.
- [ ] Contract — interface/schema/protocol.
- [ ] Reference — minimal single-node implementation.
- [ ] Operated — multi-tenant/process/region/people/SLA coordination; this PR
      does not belong in the public SDK.

**Why this repository is the correct side of the boundary:**

<!-- For mixed features, identify the open contract and private implementation. -->

## Checklist

- [ ] `pytest tests/evals/ -v` passes, or I explain why not.
- [ ] `python scripts/check_open_core_boundary.py` passes.
- [ ] `contextdb.init`, `factual.add`, `factual.recall`, and `search` keep working; new fields have backward-compatible defaults.
- [ ] No new runtime dependency without a discussion first.
- [ ] No hosted auth, entitlements, billing, control-plane, enterprise,
      deployment, migration-orchestration, or fleet code entered the SDK.
- [ ] Substantial contribution: I agree to the CLA in CONTRIBUTING.md (legal name and email in this description). Trivial typo PRs may skip this.
