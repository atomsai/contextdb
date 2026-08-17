"""Framework adapters: LangChain, OpenAI function-calling, CrewAI, AutoGen,
plus realtime hosts (Pipecat, LiveKit) and the shared prompt renderer."""

from __future__ import annotations

__all__ = [
    "act",
    "autogen_memory",
    "crewai_memory",
    "langchain_memory",
    "livekit",
    "openai_tools",
    "pipecat",
    "prompting",
]
