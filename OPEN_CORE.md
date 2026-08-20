# ContextDB open-core constitution

**Status: binding for all future work.** Changes require an explicit
architecture decision and maintainer approval.

> Open the single-node semantics and reference engine. Keep multi-tenant coordination, operated guarantees, governance, and proof private.

This repository is the public Apache-2.0 engine. It must remain complete,
useful offline, inspectable, and safe without a hosted account, license key,
paid flag, time bomb, or phone-home.

Existing Apache grants are permanent. New code is reviewed prospectively
against this constitution before merge and again before package publication.

## Four change classes

Every capability-changing contribution is classified as exactly one:

1. **Semantic** — portable correctness required offline: open implementation.
2. **Contract** — interface, schema, protocol, or compatibility test: open.
3. **Reference** — minimal single-node implementation proving a contract: open.
4. **Operated** — coordinates tenants, processes, regions, people, or an SLA:
   private Cloud implementation, not this repository.

Docs, tests, and maintenance with no capability change declare that explicitly.
Mixed features are split. This repository may contain the contract and minimal
reference implementation. Distributed schedulers, controllers, managed
workers, deployment topology, and operations stay private.

## What belongs here

- Memory and epistemic data models.
- `TrustPolicy`, act/ask/abstain, confirmation, and local explanation.
- PII-before-embed, injection defenses, and public adversarial evals.
- Local CRUD, SQLite, and a basic Postgres reference adapter.
- Atomic mutation/version/audit behavior required for correct storage.
- Deletion semantics and local verification.
- Portable consistency-token, embedding-provider, and connector contracts.
- Minimal reference implementations and compatibility tests.

Safety and correctness are never commercially gated or intentionally weaker
than the hosted service.

## What does not belong here

- Hosted gateway, tenancy, API-key auth, entitlements, quotas, billing, or
  abuse controls.
- Organization/project/team control plane or governance console.
- Multi-worker routing, replica orchestration, failover, autoscaling, backups,
  incidents, or availability guarantees.
- Managed model fleet, batching service, backfill control, or deployment code.
- Managed connector sync, checkpoints, retries, observability, or support SLAs.
- Organization Action Ledger, signed compliance exports, or retention service.
- Policy rollout/canary/approval controllers.
- SSO, SCIM, KMS/BYOK, private networking, residency, or contractual controls.

Thin generated Cloud clients, public API schemas, connector interfaces, and
common community adapters are public distribution artifacts because they grow
the ecosystem; they may live in separate public packages and contain no hosted
implementation.

## Contribution and release gate

- Pull requests declare one change class and justify repository placement.
- Ambiguous changes stop for boundary review; publication is never the default.
- Contract tests may be shared; private implementations are not copied here.
- Every PyPI release receives an open-core diff review.
- `scripts/check_open_core_boundary.py` and CI enforce mechanical invariants.

Private code can be opened later. An Apache grant cannot be clawed back.

## Commercial model

ContextDB Cloud sells operation, coordination, governance, and proof of this
open correctness model. The SDK is not a crippled free tier. If Cloud cannot
win while this engine is useful, Cloud must become stronger rather than making
the SDK weaker.

See [COMMERCIAL.md](COMMERCIAL.md) for trademark rules. Apache-2.0 permits
commercial use of the code; it does not grant rights to ContextDB trademarks.
