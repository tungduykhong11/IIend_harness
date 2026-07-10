"""ExecutorAgent — the task execution worker.  Spec 005 §2.

Each Executor is spawned fresh for a single task, runs a ReAct (tool-use)
loop with an LLM, invokes actions via ``ActionDispatcher``, and returns
``task.result`` to the Orchestrator.

Architecture (Spec 005 §2.4)
-----------------------------
1. Wait for ``task.dispatch`` on the inbox queue
2. Build system prompt from ``skill_context`` (§2.3)
3. ReAct loop (§2.4):
   - LLM decides: call tool OR return answer
   - If tool → ActionDispatcher.dispatch() → result → feed back to LLM
   - If answer → format output, exit loop
4. Send ``task.result`` (status=DONE or DONE_WITH_CONCERNS)
5. COMPLETE → DEAD (§2.2 lifecycle)
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import Any
from uuid import UUID

from llend.llm.client import LLMClient
from llend.registry.action_dispatcher import ActionDispatcher, ActionDispatchError
from llend.registry.models import ActionBinding
from llend.runtime.lifecycle import AgentType
from llend.runtime.message import (
    AgentErrorCode,
    Message,
    MsgType,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM API call fails.  Spec 005 §2.6."""


# ---------------------------------------------------------------------------
# System prompt template  —  Spec 005 §2.3
# ---------------------------------------------------------------------------

EXECUTOR_SYSTEM_PROMPT = """You are an EXECUTOR agent. Your job is to complete ONE task using the tools provided.

## Task
- Skill: {skill_name} — {skill_description}
- Task spec: {task_spec}

**IMPORTANT: The task spec above contains the EXACT parameters to use.**
Do NOT invent your own query, URL, or search terms.  Use the values from
the task spec exactly as given.  If the task spec says "iPhone 15", do NOT
search for "laptop" or any other product.  If a required parameter is
missing from the task spec, report it as a concern — do not substitute
your own defaults.

## Available Tools
{tool_descriptions}

## Instructions
1. Read the task spec carefully. Understand what output is expected.
2. Use the available tools to gather data or perform actions.
3. If you need data from a tool, CALL it. Don't guess — use tools for real data.
4. When you have everything you need, produce the final output.
5. If something is wrong or uncertain, report it in "concerns".
6. Output must match the expected schema: {output_schema}

## Output Format
Respond with JSON:
{{
  "status": "done" | "done_with_concerns" | "partial" | "error",
  "output": <task output — must match the expected schema>,
  "concerns": ["list of concerns, if any"]
}}"""


# ---------------------------------------------------------------------------
# ExecutorAgent  —  Spec 005 §2
# ---------------------------------------------------------------------------


class ExecutorAgent:
    """Execute ONE task using a ReAct tool-use loop.  Spec 005 §2.

    Stateless beyond the current task — spawned fresh, killed after
    ``task.result``.

    Parameters
    ----------
    runtime:
        The runtime that spawned this agent — used to send messages and
        register the message handler.
    instance_id:
        The unique instance id assigned during ``runtime.spawn()``.
    session_id:
        The current session id (§2.1).
    llm_client:
        Any ``LLMClient`` implementation (Anthropic, DeepSeek, …).
    max_tool_calls:
        Safety limit on ReAct loop iterations (§7.1).  Default: 20.
    """

    def __init__(
        self,
        runtime: Any,  # AgentRuntime — avoid circular import
        instance_id: str,
        session_id: UUID,
        llm_client: LLMClient,
        *,
        max_tool_calls: int = 20,
    ) -> None:
        self._runtime = runtime
        self._instance_id = instance_id
        self._session_id = session_id
        self._llm = llm_client
        self._max_tool_calls = max_tool_calls

        # Per-task state (set when task.dispatch arrives)
        self._task_id: str = ""
        self._skill_name: str = ""
        self._task_spec: dict[str, Any] = {}
        self._skill_context: dict[str, Any] = {}
        self._action_dispatcher: ActionDispatcher | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle  §2.2
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Register the message handler and wait for ``task.dispatch``.

        Called by the runtime's agent factory after spawn (§5.2).
        The agent processes exactly ONE task and then exits.
        """
        await self._runtime.register_handler(self._instance_id, self._handler)

        # Wait for task.dispatch on the inbox queue.
        # The runtime's spawn() creates a queue for this instance;
        # messages are delivered via send() → handler callback.
        handle = self._runtime._agents.get(self._instance_id)
        if handle is None:
            logger.error("Executor %s: no handle found after spawn", self._instance_id)
            return

        try:
            # Block until we receive task.dispatch  §2.2
            msg = await handle.queue.get()
            if msg.msg_type != MsgType.TASK_DISPATCH:
                logger.warning(
                    "Executor %s: expected task.dispatch, got %s",
                    self._instance_id, msg.msg_type.value,
                )
                await self._send_error(AgentErrorCode.UNKNOWN, "Unexpected message type")
                return

            await self._execute_task(msg)

        except asyncio.CancelledError:
            logger.info("Executor %s cancelled", self._instance_id)
        except Exception:
            logger.exception("Executor %s crashed", self._instance_id)
            await self._send_error(
                AgentErrorCode.CRASH,
                "Executor crashed — see logs for traceback",
            )

    # ------------------------------------------------------------------
    # Message handler  (fire-and-forget from runtime.send)
    # ------------------------------------------------------------------

    async def _handler(self, message: Message) -> None:
        """Callback registered with runtime — not used directly.

        The agent blocks on ``handle.queue.get()`` for task.dispatch.
        Subsequent messages (if any) would be queued but are irrelevant
        since the agent dies after one task.
        """

    # ------------------------------------------------------------------
    # Task execution  §2.4
    # ------------------------------------------------------------------

    async def _execute_task(self, dispatch_msg: Message) -> None:
        """Process a single ``task.dispatch`` message.  §2.4."""
        payload = dispatch_msg.payload
        self._task_id = payload.get("task_id", "")
        self._skill_name = payload.get("skill_name", "unknown")
        self._task_spec = payload.get("task_spec", {})
        self._skill_context = payload.get("skill_context", {})

        logger.info(
            "Executor %s: executing task %s skill=%s",
            self._instance_id, self._task_id, self._skill_name,
        )

        # Build ActionDispatcher from skill_context  §2.5
        bindings_raw = self._skill_context.get("action_bindings", {})
        action_bindings: dict[str, ActionBinding] = {}
        for name, raw in bindings_raw.items():
            if isinstance(raw, dict):
                action_bindings[name] = ActionBinding(**raw)
        handler = self._skill_context.get("handler")
        logger.warning(
            "EXECUTOR %s: handler=%s, skill=%s, custom_actions=%s",
            self._instance_id,
            type(handler).__name__ if handler else "NONE",
            self._skill_name,
            [n for n, b in action_bindings.items() if b.source == "custom"],
        )
        self._action_dispatcher = ActionDispatcher(action_bindings, handler=handler)

        # Build system prompt  §2.3
        system_prompt = self._build_system_prompt()

        # Build tool definitions for LLM  §2.5
        tools = self._build_tool_definitions(action_bindings)

        # Run ReAct loop  §2.4
        try:
            result = await self._react_loop(system_prompt, tools)
        except LLMError as exc:
            # §2.6: LLM API error
            logger.exception("Executor %s: LLM API error", self._instance_id)
            await self._send_error(AgentErrorCode.LLM_ERROR, str(exc))
            return
        except Exception as exc:
            logger.exception("Executor %s: ReAct loop crashed", self._instance_id)
            await self._send_error(AgentErrorCode.CRASH, str(exc))
            return

        # Send task.result  §2.2
        await self._send_result(result)

    # ------------------------------------------------------------------
    # System prompt  §2.3
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the Executor system prompt from skill_context.  §2.3."""
        skill_md = self._skill_context.get("skill_md", "")
        output_schema = self._skill_context.get("output_schema")
        schema_text = _json.dumps(output_schema, indent=2) if output_schema else "(any)"

        # Extract one-line description from skill_md frontmatter
        description = self._skill_name
        if skill_md:
            for line in skill_md.split("\n"):
                if line.startswith("description:"):
                    description = line.split(":", 1)[1].strip()
                    break

        # Build human-readable tool list for the prompt
        action_bindings = self._skill_context.get("action_bindings", {})
        tool_lines: list[str] = []
        for name in self._skill_context.get("allowed_actions", action_bindings.keys()):
            tool_lines.append(f"  - {name}")
        tool_descriptions = "\n".join(tool_lines) if tool_lines else "(no tools available)"

        return EXECUTOR_SYSTEM_PROMPT.format(
            skill_name=self._skill_name,
            skill_description=description,
            task_spec=_json.dumps(self._task_spec, ensure_ascii=False),
            tool_descriptions=tool_descriptions,
            output_schema=schema_text,
        )

    # ------------------------------------------------------------------
    # Tool definitions  §2.5
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tool_definitions(
        action_bindings: dict[str, ActionBinding],
    ) -> list[dict[str, Any]]:
        """Build OpenAI function-calling tool definitions.  §2.5.

        Uses ``ActionBinding.input_schema`` when available; falls back to
        an empty parameters object.
        """
        tools: list[dict[str, Any]] = []
        for name, binding in action_bindings.items():
            params_schema: dict[str, Any] = {
                "type": "object",
                "properties": {},
            }

            if binding.input_schema:
                params_schema = binding.input_schema
            elif binding.config:
                # Legacy fallback: extract simple params from config for basic tools
                props: dict[str, Any] = {}
                for key in binding.config:
                    if isinstance(binding.config[key], (str, int, float, bool)):
                        type_map = {
                            str: "string", int: "integer",
                            float: "number", bool: "boolean",
                        }
                        props[key] = {
                            "type": type_map.get(type(binding.config[key]), "string"),
                            "description": key,
                        }
                if props:
                    params_schema["properties"] = props

            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Action: {name}",
                    "parameters": params_schema,
                },
            })
        return tools

    # ------------------------------------------------------------------
    # ReAct loop  §2.4
    # ------------------------------------------------------------------

    async def _react_loop(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the tool-use loop.  §2.4.

        Returns ``{status, output, concerns}`` on success.
        Raises on unrecoverable error (caught by _execute_task).
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Execute: {_json.dumps(self._task_spec)}"},
        ]

        for iteration in range(self._max_tool_calls):
            # Call LLM with tools  §2.4
            response_text = await self._llm.generate(messages, tools=tools if tools else None)

            # Check if LLM wants to call a tool (function-calling response).
            # In non-streaming mode, the LLM may return tool_calls in the
            # response alongside or instead of text content.  We handle both.
            tool_calls = await self._extract_tool_calls(messages, response_text, tools)

            if tool_calls:
                # Execute each tool call and feed results back to LLM  §2.4
                for tc in tool_calls:
                    try:
                        assert self._action_dispatcher is not None
                        result = await self._action_dispatcher.dispatch(
                            tc["name"], tc.get("arguments", {}),
                        )
                        messages.append({
                            "role": "user",
                            "content": f"Tool result for {tc['name']}: {_json.dumps(result, default=str)}",
                        })
                    except ActionDispatchError as exc:
                        # §2.6: Tool call fails → let LLM decide: retry or error
                        messages.append({
                            "role": "user",
                            "content": f"Tool error for {tc['name']}: {exc}",
                        })
                continue  # Back to LLM with tool results

            # No tool calls — LLM produced final answer  §2.4
            parsed = self._parse_output(response_text)
            if not parsed.get("parsed_ok", True):
                # §2.6: Output doesn't parse as JSON
                await self._send_error(
                    AgentErrorCode.VALIDATION_ERROR,
                    f"Output could not be parsed as JSON. Raw text: {parsed.get('raw_text', response_text)[:500]}",
                )
                return parsed
            return parsed

        # Exhausted tool call limit  §2.6
        return {
            "status": "error",
            "output": None,
            "concerns": [f"Exceeded max tool calls ({self._max_tool_calls})"],
        }

    # ------------------------------------------------------------------
    # Tool call extraction
    # ------------------------------------------------------------------

    async def _extract_tool_calls(
        self,
        messages: list[dict[str, Any]],
        response_text: str,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Determine whether the LLM response contains tool calls.

        LLMs (especially DeepSeek) often output natural-language explanation
        followed by one or more JSON tool-call blocks.  We scan the entire
        response for JSON objects — any block matching a known tool name is
        treated as a tool call.  If none are found, the response is the
        final answer.
        """
        tool_names = {t["function"]["name"] for t in tools}
        tool_calls: list[dict[str, Any]] = []

        for obj in _find_json_objects(response_text):
            if "name" in obj and "arguments" in obj:
                if obj["name"] in tool_names:
                    tool_calls.append({
                        "name": obj["name"],
                        "id": obj.get("id", ""),
                        "arguments": obj["arguments"],
                    })

        return tool_calls

    # ------------------------------------------------------------------
    # Output parsing  §2.4
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(text: str) -> dict[str, Any]:
        """Parse the LLM's final JSON output.  §2.4.

        Returns ``{status, output, concerns, parsed_ok}``.  When
        ``parsed_ok`` is False, the caller must send ``agent.error``
        with ``VALIDATION_ERROR`` per §2.6.

        Uses ``_find_json_objects`` to locate JSON anywhere in the
        response — handles markdown fences (`` ```json ``` ``) and
        natural-language text mixed with JSON blocks.
        """
        for obj in _find_json_objects(text):
            if isinstance(obj, dict) and "output" in obj:
                return {
                    "status": obj.get("status", "done"),
                    "output": obj["output"],
                    "concerns": obj.get("concerns", []),
                    "parsed_ok": True,
                }

        # Also try parsing the whole text as pure JSON (no fences)
        try:
            parsed = _json.loads(text.strip())
            if isinstance(parsed, dict) and "output" in parsed:
                return {
                    "status": parsed.get("status", "done"),
                    "output": parsed["output"],
                    "concerns": parsed.get("concerns", []),
                    "parsed_ok": True,
                }
        except (_json.JSONDecodeError, TypeError):
            pass

        # §2.6: Output doesn't parse as JSON → signal error
        return {
            "status": "error",
            "output": None,
            "concerns": ["LLM output could not be parsed as valid JSON"],
            "parsed_ok": False,
            "raw_text": text,
        }

    # ------------------------------------------------------------------
    # Result / error dispatch  §2.6
    # ------------------------------------------------------------------

    async def _send_result(self, result: dict[str, Any]) -> None:
        """Send ``task.result`` to the Orchestrator.  §2.2."""
        status_raw = result.get("status", "done")
        try:
            status = TaskStatus(status_raw)
        except ValueError:
            status = TaskStatus.DONE

        msg = Message(
            session_id=self._session_id,
            sender=AgentType.EXECUTOR.value,
            sender_instance=self._instance_id,
            recipient=AgentType.ORCHESTRATOR.value,
            msg_type=MsgType.TASK_RESULT,
            payload={
                "task_id": self._task_id,
                "status": status.value,
                "output": result.get("output"),
                "concerns": result.get("concerns", []),
            },
        )
        await self._runtime.send(msg)
        logger.info(
            "Executor %s: task.result sent status=%s", self._instance_id, status.value,
        )

    async def _send_error(self, code: AgentErrorCode, detail: str) -> None:
        """Send ``agent.error`` to the Orchestrator.  §2.6."""
        msg = Message(
            session_id=self._session_id,
            sender=AgentType.EXECUTOR.value,
            sender_instance=self._instance_id,
            recipient=AgentType.ORCHESTRATOR.value,
            msg_type=MsgType.AGENT_ERROR,
            payload={
                "error_code": code.value,
                "detail": detail,
                "recoverable": code != AgentErrorCode.CRASH,
            },
        )
        await self._runtime.send(msg)
        logger.info(
            "Executor %s: agent.error sent code=%s", self._instance_id, code.value,
        )


# ---------------------------------------------------------------------------
# JSON extraction helper  (module-level)
# ---------------------------------------------------------------------------


def _find_json_objects(text: str) -> list[dict[str, Any]]:
    """Find all top-level JSON objects in *text*, handling mixed NL + JSON.

    LLMs often output natural-language text followed by one or more JSON
    blocks.  This function scans for ``{...}`` pairs (tracking brace depth)
    and returns every successfully parsed dict.

    Used by ``ExecutorAgent._extract_tool_calls()`` to find tool-call
    blocks embedded anywhere in the LLM response.
    """
    results: list[dict[str, Any]] = []
    depth = 0
    start: int | None = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                segment = text[start : i + 1]
                try:
                    obj = _json.loads(segment)
                    if isinstance(obj, dict):
                        results.append(obj)
                except (_json.JSONDecodeError, TypeError):
                    pass
                start = None
        if depth < 0:  # unbalanced braces — reset
            depth = 0
            start = None

    return results
