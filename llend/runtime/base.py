"""AgentRuntime — abstract interface for an agent execution backend.

Every runtime backend (LangGraph, asyncio, future Ray/Celery) must implement
this ABC.  Skills and agents only depend on this interface — never on a
concrete runtime.  This is the **replaceable backend** pattern described in
Spec 001 §3.1.

Spec reference
==============
- **§3.1** — "The AgentRuntime is an ABC. v0 ships LangGraphRuntime …
  v1 could add RayRuntime or CeleryRuntime without changing skill code."
- **§3.2** — the four-method interface: ``spawn``, ``send``, ``interrupt``, ``kill``
  (plus ``shutdown`` — practical addition for graceful cleanup)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from llend.runtime.message import Message

#: Signature for a message handler callback registered by an agent.
#: Receives the delivered ``Message`` and returns nothing (fire-and-forget).
MessageHandler = Callable[[Message], Awaitable[None]]


class AgentRuntime(ABC):
    """Abstract agent execution backend.  Spec 001 §3.2.

    Owns agent lifecycle (§3.3), message routing (§2.4), and the
    interrupt / human-in-the-loop primitive (§3.4).
    """

    # ---- §3.2: spawn ------------------------------------------------

    @abstractmethod
    async def spawn(self, agent_type: str, context: dict) -> str:
        """Create a new agent instance and return its *instance_id*.  §3.2.

        The concrete runtime allocates resources (asyncio tasks, queues,
        LangGraph threads, …), sets the agent's initial state to INIT
        then RUNNING (§3.3), and starts any background tasks (e.g.
        heartbeat, §2.2 ``agent.heartbeat`` with 30s idle threshold).
        """
        ...

    # ---- §3.2: send ------------------------------------------------

    @abstractmethod
    async def send(self, message: Message) -> None:
        """Route *message* to its intended recipient.  §3.2, §2.4.

        The runtime resolves ``recipient`` / ``recipient_instance`` to an
        actual agent inbox and delivers the message (§2.4 routing rule).
        If the message has expired before delivery (§2.3) the runtime
        drops it and sends an ``agent.error(TIMEOUT)`` back to the sender.
        """
        ...

    # ---- §3.2: interrupt -------------------------------------------

    @abstractmethod
    async def interrupt(self, instance_id: str, prompt: str, options: list[str]) -> str:
        """Pause *instance_id*, ask the human *prompt* with the given
        *options*, and block until a decision is returned.  §3.2, §3.4.

        The runtime saves a checkpoint (§3.4), transitions the agent to
        INTERRUPT (§3.3), notifies the human channel (§3.4 step 3), and
        awaits a response.  Returns the human's chosen option string.

        Raises ``InterruptTimeoutError`` if TTL expires before response (§3.4 ¶2).
        """
        ...

    # ---- §3.2: kill -------------------------------------------------

    @abstractmethod
    async def kill(self, instance_id: str) -> None:
        """Terminate the agent *instance_id*.  §3.2.

        Must be **idempotent** — calling ``kill()`` on an already-dead
        agent (§3.3: DEAD) is a no-op, not an error.
        """
        ...

    # ---- §3.3.1 / Spec 003: register_handler -----------------------

    @abstractmethod
    async def register_handler(
        self, instance_id: str, handler: MessageHandler
    ) -> None:
        """Register an async callback that fires for every message delivered to
        *instance_id*.  Spec 003 §5.1.

        Only one handler per instance is supported; calling again replaces the
        previous handler.  The handler is invoked via ``asyncio.create_task``
        (fire-and-forget) so the runtime's ``send()`` is never blocked by
        handler logic.

        Handlers MUST be reentrant-safe: they may be called concurrently.
        """
        ...

    # ---- shutdown (practical addition, not in spec) ----------------

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully stop all agents, cancel pending tasks, and release
        resources.  After ``shutdown()`` the runtime must reject further
        ``send()`` calls.

        Not in the spec — practical addition for resource management.
        """
        ...
