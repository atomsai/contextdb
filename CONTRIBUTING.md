# Contributing to ContextDB

Thank you for wanting to improve ContextDB. This repository is the
**open-core SDK**. Read [OPEN_CORE.md](OPEN_CORE.md) before you write
code. If the change belongs in hosted multi-tenant operations, it does
not belong here.

## What we accept here

Trust and safety correctness, local runtime fixes, adapters, docs, and
evals. Evals win over prose: `pytest tests/evals/ -v`.

We do **not** accept:

- Cloud control-plane, billing, SSO, or entitlement code
- Local license keys or `if paid:` branches
- Imports of `contextdb_cloud` or any private cloud package
- Changes that loosen PII-before-embed or the hash-chained audit log

## Contributor License Agreement

Trivial changes (typos, broken links, comment nits) do not need a CLA.

**Substantial contributions** (new or material code, evals, or docs that
are not typo-fixes) require a signed CLA **before** merge. The CLA lets
Atoms AI relicense *future* contributions if dual-licensing ever
matters. It does not change the Apache-2.0 grant on code already
public through `2554dae`.

1. Read [legal/CLA-individual.md](legal/CLA-individual.md) (or
   [legal/CLA-entity.md](legal/CLA-entity.md) if you contribute on
   behalf of an employer).
2. Open a pull request whose description includes:

   > I have read and agree to the ContextDB CLA
   > (legal/CLA-individual.md or legal/CLA-entity.md).
   > Legal name: \<name\>. Email: \<email\>.

3. Maintainers record acceptances under `legal/signed/` (gitignored
   from the sdist; not a public dump of personal data).

We may enable a CLA bot later. The requirement is the signature, not
the bot.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
ruff check .
mypy contextdb --strict
pytest
python scripts/check_open_core.py
```

Do not break `contextdb.init`, `factual.add`, `factual.recall`, or
`search`. New fields need backward-compatible defaults.

## Review

[`.github/CODEOWNERS`](.github/CODEOWNERS) routes reviews. Governance
files (`OPEN_CORE.md`, `COMMERCIAL.md`, `LICENSE`, `legal/`, workflows)
need a maintainer.

## License of your contribution

By opening a substantial PR you agree the contribution is your original
work (or you have the right to submit it) and you offer it under
Apache-2.0 **and** the CLA.
