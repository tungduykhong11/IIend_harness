"""AgentRuntime — abstract interface for an agent execution backend.

Every runtime backend (asyncio, Ray, Celery, …) must implement this ABC.
Skills and agents only depend on this interface — never on a concrete runtime.
"""

from abc import ABC, abstractmethod

from llend_harness.runtime.message import Message


class AgentRuntime(ABC):
    """Abstract agent execution backend.

    Owns agent lifecycle (spawn, kill), message routing (send), and the
    interrupt / human-in-the-loop primitive.
    """

    @abstractmethod
    async def spawn(self, agent_type: str, context: dict) -> str:
        """Create a new agent instance and return its *instance_id*.

        The concrete runtime allocates resources (asyncio tasks, queues, …),
        sets the agent's initial state to INIT then RUNNING, and starts
        any background tasks (e.g. heartbeat).
        """
        ...

    @abstractmethod
    async def send(self, message: Message) -> None:
        """Route *message* to its intended recipient.

        The runtime resolves ``recipient`` / ``recipient_instance`` to an
        actual agent inbox and delivers the message.  If the message has
        expired before delivery the runtime drops it and sends an
        ``agent.error`` back to the sender.
        """
        ...

    @abstractmethod
    async def interrupt(self, instance_id: str, prompt: str, options: list[str]) -> str:
        """Pause *instance_id*, ask the human *prompt* with the given
        *options*, and block until a decision is returned.

        The runtime saves a checkpoint, transitions the agent to INTERRUPT,
        notifies the human channel, and awaits a response.  The return value
        is the human's chosen option string.
        """
        ...

    @abstractmethod
    async def kill(self, instance_id: str) -> None:
        """Terminate the agent *instance_id*.

        Must be **idempotent** — calling ``kill()`` on an already-dead agent
        is a no-op, not an error.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully stop all agents, cancel pending tasks, and release
        resources.  After ``shutdown()`` the runtime must reject further
        ``send()`` calls.
        """
        ...
