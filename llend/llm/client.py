"""LLM client abstraction — provider-agnostic interface for agent prompting.

The Responder (and future agents) depend on ``LLMClient``, never on a
concrete provider SDK.  This keeps the harness portable across LLM backends
and makes agent tests mock-friendly.

Spec references
===============
- **Spec 003 §2.4** → Responder delegates all LLM calls through this abstraction
- **Spec 003 §5.3** → ``LLMClient`` is injected into ``ResponderAgent`` at construction
- **Spec 001 §3.1** → the replaceable-backend pattern applies to LLM providers too
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field


class LLMStreamEvent(BaseModel):
    """A single event from a streaming LLM response.  Platform-agnostic.

    Maps provider-specific events (Anthropic content blocks, OpenAI deltas,
    etc.) into a unified type so the Responder never depends on SDK internals.
    """

    type: str  # "text" | "tool_call" | "done" | "error"
    text_delta: str | None = None  # Text content chunk (type="text")
    tool_name: str | None = None  # Tool requested (type="tool_call")
    tool_input: dict[str, Any] | None = None  # Tool arguments (type="tool_call")
    tool_call_id: str | None = None  # Unique tool call ID (type="tool_call")
    finish_reason: str | None = None  # "end_turn"|"max_tokens"|"tool_use" (type="done")
    error_message: str | None = None  # Error description (type="error")


class LLMClient(ABC):
    """Abstract base for LLM provider clients.

    Implementations wrap provider-specific SDKs (Anthropic, OpenAI, etc.) and
    expose a unified interface so agents never depend on a particular provider.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> str:
        """Non-streaming generation.  Returns the full response text."""
        ...

    @abstractmethod
    async def stream_generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Streaming generation.  Yields events as they arrive from the provider."""
        ...


class AnthropicClient(LLMClient):
    """LLMClient backed by the Anthropic (Claude) API.

    Resolves the API key from the constructor argument, then the
    ``ANTHROPIC_API_KEY`` environment variable.  Raises ``RuntimeError``
    if neither is set.

    Parameters
    ----------
    api_key:
        Anthropic API key.  If ``None``, reads from ``ANTHROPIC_API_KEY`` env var.
    model:
        Claude model ID.  Default: ``claude-sonnet-4-20250514``.
    max_tokens:
        Maximum output tokens per request.  Default: 4096.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = None  # anthropic.AsyncAnthropic — lazy init

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily create the underlying AsyncAnthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic  # type: ignore[import-untyped]

            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Pass api_key= to AnthropicClient() or set the env var."
                )
            self._client = AsyncAnthropic(api_key=key)
        return self._client

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> str:
        """Non-streaming call.  Concatenates all text content blocks."""
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    async def stream_generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Streaming call — maps Anthropic SDK events to ``LLMStreamEvent``."""
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        try:
            async with client.messages.stream(**kwargs) as stream:
                current_tool_name: str | None = None
                current_tool_id: str | None = None
                current_tool_input: dict[str, Any] = {}

                async for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield LLMStreamEvent(
                                type="text",
                                text_delta=event.delta.text,
                            )

                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool_name = event.content_block.name
                            current_tool_id = event.content_block.id
                            current_tool_input = event.content_block.input or {}

                    elif event.type == "content_block_stop":
                        if current_tool_name is not None:
                            yield LLMStreamEvent(
                                type="tool_call",
                                tool_name=current_tool_name,
                                tool_call_id=current_tool_id,
                                tool_input=current_tool_input,
                            )
                            current_tool_name = None
                            current_tool_id = None
                            current_tool_input = {}

                    elif event.type == "message_stop":
                        yield LLMStreamEvent(
                            type="done",
                            finish_reason="end_turn",
                        )

        except Exception as exc:
            yield LLMStreamEvent(
                type="error",
                error_message=str(exc),
            )
