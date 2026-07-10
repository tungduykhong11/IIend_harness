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

import json as _json
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
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Non-streaming generation.  Returns the full response text.

        When *tools* are provided, the LLM may emit tool-call blocks in the
        response.  Implementations should pass tools through to the provider
        API (native function-calling) and serialize any tool-call response
        into JSON in the returned text.
        """
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
        tools: list[dict[str, Any]] | None = None,
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
        if tools:
            kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)
        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
            elif block.type == "tool_use":
                # Serialize tool_use as JSON so the Executor's text parser
                # can extract it.  Native function-calling via API is more
                # reliable than text-parsing JSON from the system prompt.
                parts.append(_json.dumps({
                    "name": block.name,
                    "id": block.id,
                    "arguments": block.input,
                }))
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


# ---------------------------------------------------------------------------
# DeepSeekClient  —  Spec 005 §4
# ---------------------------------------------------------------------------


class DeepSeekClient(LLMClient):
    """LLMClient backed by DeepSeek (OpenAI-compatible API).  Spec 005 §4.3.

    DeepSeek V4 Pro exposes an OpenAI-compatible chat completions endpoint.
    This client wraps the ``openai`` SDK's ``AsyncOpenAI``, following the
    same lazy-init pattern as ``AnthropicClient``.

    Parameters
    ----------
    api_key:
        DeepSeek API key.  If ``None``, reads from ``DEEPSEEK_API_KEY`` env var.
    model:
        Model ID.  Default: ``deepseek-chat`` (V4 Pro).
    base_url:
        API base URL or custom endpoint.  Default: ``https://api.deepseek.com``.
    max_tokens:
        Maximum output tokens per request.  Default: 4096.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self._model = model
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client: Any = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily create the underlying AsyncOpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]

            if not self._api_key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY environment variable is not set. "
                    "Pass api_key= to DeepSeekClient() or set the env var."
                )
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Non-streaming generation.  Returns the full response text.  §4.3."""
        client = self._ensure_client()
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # If the LLM returned native tool calls, serialize them as JSON
        # so the Executor's text parser can extract them.
        if choice.message.tool_calls:
            parts: list[str] = []
            if choice.message.content:
                parts.append(choice.message.content)
            for tc in choice.message.tool_calls:
                parts.append(_json.dumps({
                    "name": tc.function.name,
                    "id": tc.id,
                    "arguments": (
                        _json.loads(tc.function.arguments)
                        if tc.function.arguments else {}
                    ),
                }))
            return "\n".join(parts)

        return choice.message.content or ""

    async def stream_generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Streaming generation with proper tool-call delta accumulation.  §4.3 (fixed).

        OpenAI-compatible streaming sends tool call arguments in fragments
        across multiple chunks.  We accumulate by ``(index)`` and only yield
        a ``tool_call`` event once the arguments JSON is complete.
        """
        client = self._ensure_client()
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": msgs,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        # Accumulate streaming tool call deltas by index.
        # A single tool call's name, id, and arguments arrive across
        # multiple chunks — we accumulate until each is complete.
        tool_call_acc: dict[int, dict[str, Any]] = {}

        def _try_finalize(idx: int) -> LLMStreamEvent | None:
            tc = tool_call_acc.get(idx)
            if tc is None:
                return None
            if not (tc.get("id") and tc.get("name")):
                return None
            import json as _json

            try:
                parsed = _json.loads(tc["arguments"])
            except (_json.JSONDecodeError, TypeError):
                return None  # still receiving fragments
            # Complete — yield and clear
            del tool_call_acc[idx]
            return LLMStreamEvent(
                type="tool_call",
                tool_name=tc["name"],
                tool_call_id=tc["id"],
                tool_input=parsed,
            )

        finish_reason = "end_turn"

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta

                # Text content
                if delta.content:
                    yield LLMStreamEvent(type="text", text_delta=delta.content)

                # Tool call deltas — accumulate by index  §4.3 stream fix
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_acc:
                            tool_call_acc[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        acc = tool_call_acc[idx]
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] = tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments
                        event = _try_finalize(idx)
                        if event is not None:
                            yield event

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            # End of stream — emit incomplete tool calls as errors
            for idx in sorted(tool_call_acc.keys()):
                tc = tool_call_acc[idx]
                yield LLMStreamEvent(
                    type="error",
                    error_message=(
                        f"Tool call '{tc.get('name', 'unknown')}' "
                        f"(id={tc.get('id', '?')}): arguments stream ended "
                        f"before JSON was complete"
                    ),
                )
            tool_call_acc.clear()

            yield LLMStreamEvent(type="done", finish_reason=finish_reason)

        except Exception as exc:
            yield LLMStreamEvent(type="error", error_message=str(exc))


# ---------------------------------------------------------------------------
# Provider factory  —  Spec 005 §4.5
# ---------------------------------------------------------------------------


def create_llm_client(provider: str, **kwargs: Any) -> LLMClient:
    """Factory for LLM providers.  Spec 005 §4.5.

    ``provider`` is one of: ``"anthropic"``, ``"deepseek"``, ``"openai"``.
    Additional keyword arguments are forwarded to the client constructor.
    """
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "deepseek":
        return DeepSeekClient(**kwargs)
    elif provider == "openai":
        raise NotImplementedError(
            "OpenAI provider not yet implemented — deferred to v1."
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
