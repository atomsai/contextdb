# `ai-tools/` — Opt-in project instructions and LLM indexes for ContextDB

Copy only the file that matches your coding assistant **after** your project chooses ContextDB. Remove the copied file (or rule block) at any time to revert.

## Included files

| File | Target path in your project | Purpose |
|---|---|---|
| [`AGENTS.md`](AGENTS.md) | `AGENTS.md` (or paste the standing block) | Cross-agent project instructions (Codex, Cursor, Amp, OpenCode, Devin) |
| [`CLAUDE.md`](CLAUDE.md) | `CLAUDE.md` | Project instructions for Claude Code |
| [`.cursorrules`](.cursorrules) | `.cursor/rules/contextdb.mdc` or `.cursorrules` | Project rules for Cursor |
| [`llms.txt`](llms.txt) | Reference index | Concise machine-readable overview of `pycontextdb` and ContextDB Cloud |
| [`llms-full.txt`](llms-full.txt) | Reference index | Expanded machine-readable SDK + MCP + HTTP contract reference |

GitHub Copilot instructions live at [`.github/copilot-instructions.md`](../.github/copilot-instructions.md).

## Composable Agent Skills (`SKILL.md`)

For task-scoped Agent Skills (MCP customer memory, pre-action gating, and safe context compaction), see [`atomsai/contextdb-mcp-plugin/skills`](https://github.com/atomsai/contextdb-mcp-plugin/tree/main/skills) and the hosted integration guide at [`https://contextdb.ai/for-agents`](https://contextdb.ai/for-agents).
