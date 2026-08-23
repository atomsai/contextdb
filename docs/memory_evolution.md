# Explicit memory evolution

`ContextDB.evolve()` and `db.factual.evolve()` apply a caller-selected
operation with deterministic storage semantics. They do not choose an
operation, call an LLM planner, or consult the optional RL memory manager.

```python
result = await db.factual.evolve(
    "update",
    "The meeting is at 4pm",
    source="user_stated",
    entity="meeting",
    attribute="time",
    user_id="caller-7",
)

await db.require_consistency(
    min_memory_version=result.consistency_token.memory_version,
    min_wal_lsn=result.consistency_token.primary_wal_lsn,
)
```

The operation values are `add`, `update`, `delete`, and `noop`.
`MemoryEvolutionResult` reports the requested and applied operation, a closed
outcome, lineage IDs, the resulting memory when one exists, and a consistency
token read after the mutation commits.

## ADD

ADD requires nonempty content. Content follows the normal factual write path:
PII processing happens before embedding, explicit or inferred slot keys are
canonicalized, and injection-shaped content is demoted.

For an empty slot, ADD stores a new memory. If the same speaker repeats the
same canonical slot value, the applied operation is NOOP: no row is inserted
or updated and the memory version does not advance. The audit entry contains
only the fixed operation and reason. A new independent speaker asserting the
same value corroborates the current memory and applies as UPDATE.

ADD fails with `EvolutionOperationConflictError` when the slot already holds a
different value. Use UPDATE to correct an occupied slot.

## UPDATE

UPDATE requires nonempty content plus either `entity` and `attribute`, or a
`target_memory_id` that is itself currently valid and whose slot can be
resolved in the caller's scope. A superseded predecessor cannot be used as an
UPDATE target even when its lifecycle status remains `ACTIVE`. The slot must
have a current occupant.

A changed value from the same speaker creates a new memory row and closes the
predecessor with `valid_until` and `superseded_by`. It never edits corrected
content in place. The predecessor remains indexed so `as_of` recall continues
to work.

A different value from an independent speaker preserves the existing
contested-slot behavior: both values remain current and untrusted for action
until confirmation resolves the contest. The applied operation is UPDATE and
the outcome is `contested`.

For an applied UPDATE, `previous_memory_ids` contains the occupants observed
under the slot lock before the operation. The `memory` field contains the
current corroborated memory or the newly created successor.

## DELETE

DELETE requires `target_memory_id` or `entity` and `attribute`. IDs are
resolved only inside the caller's user, tenant, and agent scope. Missing and
foreign IDs both raise `EvolutionTargetNotFoundError`.

A slot-only DELETE is allowed only when exactly one current occupant exists.
Contested or otherwise ambiguous slots require an explicit target ID. The
resolved memory is hard-deleted through the transactional erasure path, an
`ERASE` audit is appended, and the vector is removed. The result contains only
the deleted ID and no memory content.

An explicit DELETE target may intentionally identify a superseded historical
row. This erases that exact row while leaving the current head unchanged.
Slot-only DELETE never selects historical rows.

DELETE does not create a tombstone or retain a content row. It does not change
the broader `forget_user()` and `verify_forgotten()` contracts.

## NOOP

NOOP requires a nonempty reason of at most 256 characters. Optional target or
slot references are scope-checked. The reason is passed through the configured
PII processor before the content-free `NOOP` audit entry is appended.

NOOP does not change memory rows, vectors, or the memory version. Its result
token therefore reports the same memory version visible before the operation
when no unrelated concurrent mutation advances that project.

## Transactions and limitations

On Postgres, the memory rows, project/global revision changes, vector-cache
invalidation, and audit append run in one transaction. An audit failure rolls
back the memory operation. Per-slot advisory locks serialize concurrent
updates across processes.

SQLite is the single-process reference backend. Its per-slot lock preserves
deterministic slot decisions, but multi-step row and audit writes do not have
the Postgres rollback guarantee. Use Postgres when atomic multi-row evolution
is required.

The API applies an operation selected by the caller. It does not infer intent
from prose, merge arbitrary memories, plan a sequence of operations, or
coordinate distributed workers.

## Open-core placement and checks

This unit is open portable correctness under `OPEN_CORE.md`:

- the operation enums, result, and typed errors are the public Contract;
- slot, lineage, audit, and deletion behavior are the Semantic implementation;
- SQLite and basic Postgres behavior are the single-node Reference.

The pull-request template still requires exactly one classification. For this
combined unit, select **Semantic** as the dominant class and describe the
Contract and Reference facets in the justification. Run the full unit and eval
suites, Ruff, strict mypy, package build, release artifact inspection, and
`python scripts/check_open_core_boundary.py`. Real Postgres tests run only
when `CONTEXTDB_TEST_POSTGRES_URL` is available. No operated Cloud capability
belongs in this change.
