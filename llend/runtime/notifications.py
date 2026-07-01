"""Notification channels for human-in-the-loop interrupts.  Spec 001 §3.4.

When an agent raises ``interrupt.raise`` the runtime **notifies the human**
(§3.4 ¶1, step 3).  v0 ships with ``ConsoleNotificationChannel`` (prints to
stdout).  Future channels — Telegram, Discord, WebSocket (§3.4 ¶3) — are
plugged in the same way: implement the ABC and pass to the runtime.

Spec reference
==============
- **§3.4 ¶1 step 3** — "Notifies human via configured channel"
- **§3.4 ¶3** — channel list: Telegram, Discord, WebSocket (future)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from llend.runtime.checkpoint import Checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract channel  —  Spec 001 §3.4 ¶1 step 3
# ---------------------------------------------------------------------------


class NotificationChannel(ABC):
    """Contract for a human-notification transport.  Spec 001 §3.4.

    The runtime calls these hooks at well-defined points in the interrupt
    lifecycle.  Implementations must be **non-blocking** — they are called
    inside the runtime's event loop.  Use ``asyncio.create_task`` for
    long-running work like HTTP calls.
    """

    @abstractmethod
    async def notify_interrupt(self, checkpoint: Checkpoint) -> None:
        """Called when an agent raises ``interrupt.raise`` (§3.4 step 3).

        The channel should deliver *checkpoint.interrupt_message* together
        with the available *checkpoint.interrupt_options* to the human.
        """
        ...

    @abstractmethod
    async def notify_interrupt_timeout(self, checkpoint: Checkpoint) -> None:
        """Called when a checkpoint's TTL expires without a human response (§3.4 ¶2).

        The channel should inform the human that the interrupt has been
        auto-terminated — no further response will be accepted.
        """
        ...


# ---------------------------------------------------------------------------
# Console (v0 default)  —  Spec 001 §3.4 ¶3
# ---------------------------------------------------------------------------


class ConsoleNotificationChannel(NotificationChannel):
    """Print interrupt notifications to stdout — v0 default.

    Useful during development.  Production deployments should use a
    channel that reaches the human: Telegram, Discord, WebSocket (§3.4).
    """

    def __init__(self, separator_width: int = 60) -> None:
        self._sep = "─" * separator_width

    async def notify_interrupt(self, checkpoint: Checkpoint) -> None:
        """Print a formatted interrupt prompt to stdout (§3.4 step 3)."""
        options_text = "\n".join(
            f"  [{chr(65 + i)}] {opt}"  # A, B, C, …
            for i, opt in enumerate(checkpoint.interrupt_options)
        )
        print(
            f"\n{self._sep}\n"
            f"⚡ INTERRUPT — {checkpoint.agent_instance} ({checkpoint.agent_type})\n"
            f"{self._sep}\n"
            f"\n{checkpoint.interrupt_message}\n"
            f"\n{options_text}\n"
            f"\n⏳ TTL: {checkpoint.ttl_seconds}s  "
            f"ID: {checkpoint.interrupt_id}\n"
            f"{self._sep}\n",
            flush=True,
        )

    async def notify_interrupt_timeout(self, checkpoint: Checkpoint) -> None:
        """Print a timeout notification to stdout (§3.4 ¶2)."""
        print(
            f"\n{self._sep}\n"
            f"⌛ INTERRUPT TIMEOUT — {checkpoint.agent_instance}\n"
            f"{self._sep}\n"
            f"\nInterrupt {checkpoint.interrupt_id} expired after "
            f"{checkpoint.ttl_seconds}s without a human response.\n"
            f"Agent has been terminated.\n"
            f"\n{self._sep}\n",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Multi-channel fan-out  —  Spec 001 §3.4 ¶3
# ---------------------------------------------------------------------------


class MultiChannel(NotificationChannel):
    """Fan-out to multiple ``NotificationChannel`` instances (§3.4 — multiple channels).

    Calls each channel in order.  A failure in one channel does **not**
    prevent subsequent channels from being called — errors are logged
    and swallowed (degraded notification is better than no notification).
    """

    def __init__(self, *channels: NotificationChannel) -> None:
        self._channels = channels

    async def notify_interrupt(self, checkpoint: Checkpoint) -> None:
        for ch in self._channels:
            try:
                await ch.notify_interrupt(checkpoint)
            except Exception:
                logger.exception("notification channel %r failed on notify_interrupt", ch)

    async def notify_interrupt_timeout(self, checkpoint: Checkpoint) -> None:
        for ch in self._channels:
            try:
                await ch.notify_interrupt_timeout(checkpoint)
            except Exception:
                logger.exception(
                    "notification channel %r failed on notify_interrupt_timeout", ch
                )
