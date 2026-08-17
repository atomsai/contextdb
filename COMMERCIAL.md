# ContextDB commercial terms

ContextDB is **open-core**. The software in this repository is
Apache-2.0. Hosted operations, compliance, and managed enforcement are
a separate product.

This is not a substitute for a customer contract. It states what is
free, what is sold, and how the name may be used.

## Free (this repository)

`pip install pycontextdb` (and this GitHub tree) includes:

- The full local SDK and CLI
- Epistemic typing, `TrustPolicy`, `VerifyBeforeAct`, `confirm()`
- SQLite runtime, backend interfaces, `add_fast`, recall, forget, export
- Stock adapters (Pipecat, LiveKit, MCP, …) and prompt-injection defenses
- Baseline policies and the public eval / bake-off suite

There is **no license key**. There is **no local paid boolean**. If a
flag exists in `ContextDBConfig`, it is an optional local pathway, not
an entitlement.

## Paid (ContextDB Cloud)

Sold from the private `contextdb-cloud` control plane, enforced
**server-side**:

- Hosted multi-tenant runtime and managed storage / indexing
- Organizations, projects, API keys, entitlements
- SSO, SCIM, RBAC, approval workflows
- Policy deployment, versioning, canaries, rollback
- Audit retention, signed exports, compliance evidence
- Managed confirmation workflows and SLAs
- Customer-transcript evaluation and trust reports
- Proprietary ranking / consolidation models
- Regional hosting, backups, HA, observability, support
- PyAI-native managed memory and voice-path enforcement

Cloud **depends on** the open SDK. The SDK **never** depends on cloud.
See [OPEN_CORE.md](OPEN_CORE.md).

## Trademark

**ContextDB** and the ContextDB logo are trademarks of Atoms AI.

The Apache License 2.0 grants rights to the *software*. It does **not**
grant rights to the *marks* (Apache License §6).

You may:

- Say you use ContextDB
- Cite the project and link to this repository
- State compatibility (“works with ContextDB”) when that is true

You may not, without written permission:

- Use ContextDB or confusingly similar names as the name of a competing
  product, hosted service, or company
- Use the logo or word mark to imply Atoms AI endorsement
- Register domains, social handles, or trademarks that trade on the mark

File notices and the [NOTICE](NOTICE) file exist so redistributors keep
this distinction visible.

## Attribution (optional)

Apache-2.0 does **not** require a public credit line. Commercial use of
this SDK does **not** require a link to Atoms AI, PyAI, or any other
site.

If you ship a product that uses ContextDB and you want to credit the
project, the preferred form is a visible **Built with ContextDB** (or
**Uses ContextDB**) link to the canonical project page:

https://github.com/atomsai/contextdb

That is a courtesy, not a license condition. Redistributors must still
keep `LICENSE` and [NOTICE](NOTICE) as Apache requires.

If you use the ContextDB name or logo, follow the trademark rules
above. ContextDB Cloud customer contracts may require a credit line;
that is a service term, not part of this license.

Research citations should use the paper:
https://zenodo.org/records/19647089

## Dual licensing

Code published through `2554dae` is Apache-2.0 forever, including forks
already taken. New substantial contributions to *this* repo are accepted
only under the CLA in [CONTRIBUTING.md](CONTRIBUTING.md) so Atoms AI can
relicense *future* work if needed. That does not revoke Apache-2.0 on
anything already released.

## Contact

- Product and Cloud: Gaurav Sharma — gaurav@saaslabs.co
- Issues (SDK): https://github.com/atomsai/contextdb/issues
- Trademark misuse: the same address, subject `ContextDB trademark`
