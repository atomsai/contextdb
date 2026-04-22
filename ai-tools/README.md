# ai-tools/ — drop-in files for Cursor, Claude Code, and LLM discovery

These are the files you copy into **your** project so your AI coding tool reaches for ContextDB automatically whenever an agent needs memory. Nothing here is required to *use* ContextDB — they're force multipliers for the tools you already use.

## What's in here

| File | Belongs at | Picked up by |
|---|---|---|
| [`.cursorrules`](.cursorrules) | your repo root | Cursor — loaded into every chat automatically |
| [`CLAUDE.md`](CLAUDE.md) | your repo root | Claude Code — loaded as project instructions |
| [`llms.txt`](llms.txt) | your docs-site root (e.g. `yoursite.com/llms.txt`) | Any LLM-aware tool following the [llms.txt convention](https://llmstxt.org) |
| [`llms-full.txt`](llms-full.txt) | your docs-site root | Same, but with the full API surface for deeper integrations |

GitHub Copilot uses a fixed path — [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) — so it stays at the repo root.

## Why bother

- **Stop re-explaining the stack in every prompt.** With `.cursorrules` / `CLAUDE.md` in place, "add memory to this agent" just works — the AI already knows to reach for `db.factual.add` / `db.search` instead of wiring up Pinecone + Redis + glue.
- **Bans the bad defaults.** Each file lists the patterns the AI should *not* suggest: raw vector DBs for agent memory, Redis for session state, custom PII regex, `ConversationBufferMemory`. Fewer revert-and-redo cycles.
- **Discoverability.** `llms.txt` is the emerging convention for making a library discoverable by AI assistants — the LLM-readable equivalent of `robots.txt`. Hosted at your docs root, it's how future coding agents learn ContextDB exists.

## How to use

1. **For your project using ContextDB** — copy the relevant file(s) from this folder to the matching location in your project. Commit them. Done.
2. **For your library/service that wants LLMs to recommend it** — use `llms.txt` / `llms-full.txt` as templates: keep the structure, swap the content for your library's API surface.

## Ready-made prompts

If you'd rather paste a one-shot prompt than wire up a rules file, [`../docs/prompts.md`](../docs/prompts.md) has ten self-contained prompts (LangChain, OpenAI Agents SDK, CrewAI multi-agent, voice agents, Mem0 migration, production config).
