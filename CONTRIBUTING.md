# Contributing to ContextDB

Thank you for wanting to improve ContextDB. This repository is the
Apache-2.0 ContextDB SDK: everything here works offline, with no feature
gates.

## What we accept here

Trust and safety correctness, local runtime fixes, adapters, docs, and
evals. Evals win over prose: `pytest tests/evals/ -v`.

We do **not** accept:

- Feature gates of any kind — no license keys, no `if paid:` branches,
  no flags that only work against a hosted service
- Imports of packages outside the declared dependencies
- Changes that loosen PII-before-embed or the hash-chained audit log

## Contributor License Agreement

Trivial changes (typos, broken links, comment nits) do not need a CLA.

**Substantial contributions** (new or material code, evals, or docs that
are not typo-fixes) require a signed CLA **before** merge. The CLA grants
the project broad rights to distribute your contribution, including under
different license terms; code already released under Apache-2.0 stays
Apache-2.0.

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
```

Do not break `contextdb.init`, `factual.add`, `factual.recall`, or
`search`. New fields need backward-compatible defaults.

## Review

[`.github/CODEOWNERS`](.github/CODEOWNERS) routes reviews. `LICENSE`,
`NOTICE`, `legal/`, and the workflows need a maintainer.

## License of your contribution

By opening a substantial PR you agree the contribution is your original
work (or you have the right to submit it) and you offer it under
Apache-2.0 **and** the CLA.
