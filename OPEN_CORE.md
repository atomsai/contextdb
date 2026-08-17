# ContextDB open-core boundary

**Decision:** ContextDB remains open-core.

All code public through commit `2554dae` remains Apache-2.0. This
repository stays Apache-2.0. New work in *this* tree is the community
SDK. Hosted operations live in a private `contextdb-cloud` repository
that **depends on** this SDK and that this SDK **must never** depend on.

The evals in `tests/evals/test_trust_model.py` are the spec for trust
and safety. Monetize operating, proving, and governing that spec — not
hiding it.

## Public — `atomsai/contextdb` (this repo)

Apache-2.0. Anyone may use, modify, and ship it under that license.

- Epistemic types, provenance, and corroboration
- `TrustPolicy`, `VerifyBeforeAct`, `decide()`, and confirmation APIs
- Local SQLite runtime and backend interfaces
- `add_fast`, recall, deletion, and export
- Safe Pipecat, LiveKit, and agent adapters
- Prompt rendering and injection defenses
- Baseline policy implementations (`TrustPolicy.hospital()`, `.restaurant()`)
- Trust evals, fabrication benchmarks, and synthetic fixtures
- Community SDK, CLI, and documentation

Local feature flags (`enable_multi_graph`, `enable_rl_manager`, …) are
**optional pathways in the open SDK**. They are not entitlements. They
must never consult a license key, a cloud control plane, or a “paid”
boolean.

## Private — `contextdb-cloud`

Not in this tree. Cloud depends on `pycontextdb`. The SDK does not
import it. Cloud source, bootstrap installers, and operator runbooks
do not belong in this repository.

- Hosted, multi-tenant runtime
- Managed distributed storage and indexing
- Organization, project, API-key, and entitlement control plane
- SSO, SCIM, RBAC, and approval workflows
- Policy deployment, versioning, canaries, and rollback
- Audit retention, signed exports, and compliance evidence
- Managed human-confirmation workflows and SLAs
- Customer-transcript evaluation and trust reports
- Advanced proprietary ranking / consolidation models
- Regional hosting, backups, HA, observability, and support
- PyAI-native managed memory and voice-path enforcement

## Hard rules

1. **Dependency direction.** Cloud may import `contextdb`. `contextdb`
   must not import cloud, billing, or entitlement packages.
2. **No premium-in-Apache.** Do not place cloud code in this repo behind
   feature flags. If it is here, it is Apache-2.0.
3. **Entitlements are server-side.** Never a local boolean, license
   file, or compiled-in key. A checkout of this repo can do everything
   the SDK documents, offline.
4. **Trust and safety stay open.** Typing, corroboration, contest,
   `VerifyBeforeAct`, PII-before-embed, injection rendering, forget, and
   the evals that lock those behaviors are not paid features.
5. **Trademark.** Apache-2.0 is a copyright and patent license, not a
   trademark license. A public credit line is optional, not a condition
   of commercial use. See [COMMERCIAL.md](COMMERCIAL.md) and [NOTICE](NOTICE).
6. **Releases.** PyPI publishes only from a GitHub Release after package
   inspection and the `pypi` environment approval. See below.
7. **CLA.** Substantial external contributions require a signed CLA so
   future dual-licensing of *new* work remains possible. Code through
   `2554dae` is already Apache-2.0 and stays that way. See
   [CONTRIBUTING.md](CONTRIBUTING.md).

## What we monetize

Operating the trust bar in production (hosted runtime, isolation,
uptime), proving it on customer transcripts, and governing it (SSO,
policy canaries, signed audit, confirmation SLAs). Not the correctness
of `decide()`.

## Release and accidental publication

- `scripts/check_open_core.py` — import direction, forbidden paths,
  no local entitlement fields, sdist/wheel denylist.
- CI job **Open-core boundary** runs the source checks on every push
  and pull request to `main`.
- `publish.yml` builds the distributions, inspects them, checks the
  release tag against `pyproject.toml`, then publishes only after the
  GitHub Environment `pypi` is approved.
- Hatch excludes secrets, keys, and stray scratch files from sdist and
  wheel (the 0.1.0 release shipped a token in an `.rtf`; that must not
  happen again).
- [`.github/CODEOWNERS`](.github/CODEOWNERS) requires a maintainer
  review on governance, license, and the SDK tree.

Configure the `pypi` environment with **required reviewers** so a
published tag cannot reach PyPI without a human.

## Moving a feature across the boundary

- **SDK → cloud:** delete it from this repo in a public commit. Do not
  leave a stub that phones home.
- **Cloud → SDK:** land it here as Apache-2.0 with an eval. No
  “disabled unless entitled” branch.

## Related

- [COMMERCIAL.md](COMMERCIAL.md) — product split, trademark, contact
- [CONTRIBUTING.md](CONTRIBUTING.md) — CLA and review
- [CHANGELOG.md](CHANGELOG.md) — 0.2.0 is the first trust-model PyPI cut
