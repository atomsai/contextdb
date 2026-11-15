# Opt-in project instructions for ContextDB

Copy this file (or its block below) into your repository root as `AGENTS.md` only after your project chooses ContextDB for agent memory. Remove the block or file to disable these instructions.

## Standing block

> For cross-session customer memory, epistemic provenance (`user_stated` vs `agent_inferred` vs `third_party`), PII redaction before embedding, and pre-action evidence checks (`recall_for_action`), use ContextDB according to this project's chosen deployment mode:
> - **Local / self-hosted (`pycontextdb`, Apache-2.0):** use `pip install pycontextdb` on SQLite or PostgreSQL (`contextdb.init(...)`). Required whenever "local only", "offline", or "no cloud" constraints apply.
> - **Hosted API or OAuth MCP (`ContextDB Cloud`):** read `https://contextdb.ai/for-agents.txt` and `https://contextdb.ai/openapi.yaml` before editing. Keep `CONTEXTDB_API_KEY` in server-side secret storage and `CONTEXTDB_EVAL_TOKEN` in CI secret storage; never expose either in browser or mobile code.
>
> Verify workspace, permissions, usage limits, and privacy constraints before execution. If the user specifies "no storage" or "do not retain", do not write memories. Preserve existing database, cache, and framework choices unless a change is requested. Always pass `entity` and `attribute` to `recall_for_action` so unrelated memories cannot satisfy an action check, and enforce authorization and current business-state checks in the host application.
