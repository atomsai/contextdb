# SQLite migrations

`SQLiteStore.initialize()` is idempotent. Opening a 0.2.0 file with a
newer SDK does not rebuild the database and does not drop rows.

On open, ContextDB:

1. `CREATE TABLE IF NOT EXISTS memories (...)` — no-op if the file exists.
2. `ALTER TABLE memories ADD COLUMN ...` for each trust-model column
   missing from the file (`_TRUST_COLUMNS` in `contextdb/store/sqlite_store.py`).
3. Rebuilds the in-process vector index from existing embeddings.

There is no separate migration runner and no down-migration. New columns
have defaults (`user_stated`, corroboration `1`, `confirmed=0`, …) so old
rows stay valid and conservative.

Postgres uses the same column list with `ADD COLUMN IF NOT EXISTS`.

If you pin an older SDK against a newer file, unknown columns are ignored
by `SELECT *` mapping (`row.get`). Do not hand-edit the schema.

Exports (`contextdb export` / `JSONExporter`) are the backup path before
a major version. 0.3.0 will keep this additive `ALTER` behavior unless
CHANGELOG says otherwise.
