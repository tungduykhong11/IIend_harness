"""Action Dispatcher — resolves and executes action calls within an Executor.  Spec 002 §4.4.

Skills declare *actions*; the ``ActionDispatcher`` turns those declarations into
concrete function calls at runtime.  It sits inside the Executor process and is
the bridge between LLM tool-use decisions and actual Python execution.

Spec references
===============
- **§4.4** → Action Execution Model — dispatcher flow, global vs custom routing
- **§4.2** → Action sources (global tool bridge + custom handler)
- **§4.3** → Custom action discovery (handler class conventions)
- **§6.2** → ``ActionBinding`` (the resolved mapping consumed here)
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any

from llend.registry.models import ActionBinding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ActionDispatchError — raised when an action cannot complete
# ---------------------------------------------------------------------------


class ActionDispatchError(Exception):
    """Raised by ``ActionDispatcher`` when all retries are exhausted.  §4.4 step 5.

    The Executor catches this and sends ``agent.error(TOOL_ERROR)`` to the
    Orchestrator with the detail from this exception.
    """

    def __init__(self, action_name: str, attempts: int, last_error: str) -> None:
        self.action_name = action_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Action {action_name!r} failed after {attempts} attempt(s): {last_error}"
        )


# ---------------------------------------------------------------------------
# ActionDispatcher  —  Spec 002 §4.4
# ---------------------------------------------------------------------------


class ActionDispatcher:
    """Resolves and executes action calls within an Executor.  Spec 002 §4.4.

    The dispatcher is given a dict of ``ActionBinding`` objects (merged from
    global tool bridge + custom handler) and an optional handler instance.
    When the LLM decides to call a tool, the Executor calls ``dispatch()``
    with the action name and arguments.

    Flow (§4.4 steps 2–5):

    1. LLM emits ``tool_use`` block → Executor calls ``dispatch(action, args)``
    2. Dispatcher looks up the binding, routes to global or custom
    3. Global: ``importlib.import_module(tool)`` → ``getattr(mod, function)``
    4. Custom: ``getattr(handler, function)``
    5. Result returned to LLM context
    6. On exception → retry up to ``binding.retry`` times
    7. Retries exhausted → raise ``ActionDispatchError``
    """

    def __init__(
        self,
        action_bindings: dict[str, ActionBinding],
        handler: object | None = None,
    ) -> None:
        """*action_bindings* is the merged global + custom bindings dict from
        ``SkillRegistry.resolve()``.  *handler* is the instantiated handler
        class (if ``handler.py`` exists), or ``None``.
        """
        self._bindings = action_bindings
        self._handler = handler

    # ------------------------------------------------------------------
    # §4.4 step 2–7 — dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, action_name: str, arguments: dict[str, Any]) -> Any:
        """Resolve and execute *action_name* with *arguments*.  §4.4.

        Returns the action's result on success.  Raises
        ``ActionDispatchError`` if all retries are exhausted.
        """
        binding = self._bindings.get(action_name)
        if binding is None:
            raise ActionDispatchError(
                action_name, 1, f"Unknown action {action_name!r}"
            )

        max_attempts = 1 + max(binding.retry, 0)
        last_error: str = ""

        for attempt in range(1, max_attempts + 1):
            try:
                # §4.4 step 3: route by source
                return await self._execute(binding, arguments)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Action %r attempt %d/%d failed: %s",
                    action_name, attempt, max_attempts, last_error,
                )
                if attempt < max_attempts:
                    continue

        # §4.4 step 5: retries exhausted → Executor sends agent.error
        raise ActionDispatchError(action_name, max_attempts, last_error)

    # ------------------------------------------------------------------
    # §4.4 step 3 — source routing
    # ------------------------------------------------------------------

    async def _execute(self, binding: ActionBinding, arguments: dict[str, Any]) -> Any:
        """Route to global or custom source and execute with timeout.  §4.4.

        Global bindings are resolved via ``importlib.import_module`` then
        ``getattr`` walking the dotted function path.  Custom bindings are
        resolved via ``getattr(self._handler, binding.function)``.

        Both paths enforce ``binding.timeout_ms`` via ``asyncio.wait_for``.
        """
        if binding.source == "global":
            if binding.tool is None:
                raise ActionDispatchError(
                    binding.action_name, 1, "Global binding has no 'tool' module"
                )
            # §4.4: importlib.import_module(tool) → getattr for dotted path
            mod = importlib.import_module(binding.tool)
            func = mod
            for attr in binding.function.split("."):
                func = getattr(func, attr)
        elif binding.source == "custom":
            if self._handler is None:
                raise ActionDispatchError(
                    binding.action_name, 1, "Custom binding but no handler instance"
                )
            func = getattr(self._handler, binding.function)
        else:
            raise ActionDispatchError(
                binding.action_name, 1, f"Unknown source {binding.source!r}"
            )

        # §4.4: enforce timeout via asyncio.wait_for
        if binding.timeout_ms and binding.timeout_ms > 0:
            result = await asyncio.wait_for(
                self._call(func, arguments, binding),
                timeout=binding.timeout_ms / 1000.0,
            )
        else:
            result = await self._call(func, arguments, binding)

        return result

    # ------------------------------------------------------------------
    # Internal — function call
    # ------------------------------------------------------------------

    @staticmethod
    async def _call(
        func: object, arguments: dict[str, Any], binding: ActionBinding
    ) -> Any:
        """Invoke *func* with *arguments*, merging in ``binding.config``.  §4.4.

        Config from the binding is merged AFTER the call arguments so the LLM
        can override config defaults.
        """
        merged = {**binding.config, **arguments}

        # Support both sync and async callables
        if asyncio.iscoroutinefunction(func):
            return await func(**merged)
        else:
            return func(**merged)
