"""Checkpoint model for interrupt / human-in-the-loop.  Spec 001 §3.4.

When an agent raises ``interrupt.raise`` the runtime freezes its state into a
``Checkpoint``.  LangGraph handles its own internal checkpoint via
``InMemorySaver`` / ``SqliteSaver`` — the ``Checkpoint`` Pydantic model
carries interrupt-specific *metadata* through the graph state.

Disk persistence (§3.4 ¶2) stores each checkpoint as JSON under
``~/.llend/sessions/{session_id}/checkpoints/{interrupt_id}.json``.

Spec reference
==============
- **§3.4 ¶1** — 5-step interrupt flow: freeze, save, notify, block, resume
- **§3.4 ¶2** — TTL timeout → ERROR state → Orchestrator decides
- **§3.4 schema** — ``Checkpoint`` fields (interrupt_id … resolved_at)
- **Q2** — JSON format (human-readable), not pickle
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Interrupt timeout error  —  Spec 001 §3.4 ¶2
# ---------------------------------------------------------------------------


class InterruptTimeoutError(Exception):
    """Raised inside ``runtime.interrupt()`` when the TTL expires without a
    human response.  Spec 001 §3.4 ¶2.

    The caller (Executor / Reviewer / Orchestrator) catches this to decide:
    retry, skip, or escalate — as described in the spec:

      "If human doesn't respond in TTL (default 24h), interrupt times out
       → ERROR state → Orchestrator decides: retry / skip / escalate."
    """

    def __init__(self, interrupt_id: UUID, ttl_seconds: int) -> None:
        self.interrupt_id = interrupt_id
        self.ttl_seconds = ttl_seconds
        super().__init__(
            f"Interrupt {interrupt_id} timed out after {ttl_seconds}s — "
            f"no human response received."
        )


# ---------------------------------------------------------------------------
# Checkpoint model  —  Spec 001 §3.4 schema
# ---------------------------------------------------------------------------

_DEFAULT_BASE_DIR: Path = Path.home() / ".llend"


def set_default_base_dir(path: Path) -> None:
    """Override the default base directory for checkpoint persistence.

    Useful in tests to isolate on-disk state from the real filesystem.
    """
    global _DEFAULT_BASE_DIR
    _DEFAULT_BASE_DIR = path


def get_default_base_dir() -> Path:
    """Return the current default base directory."""
    return _DEFAULT_BASE_DIR


class Checkpoint(BaseModel):
    """A frozen snapshot of an agent blocked on a human decision.

    All fields match the Spec 001 §3.4 schema exactly.
    Disk path: ``<base_dir>/sessions/<session_id>/checkpoints/<interrupt_id>.json``
    """

    # ---- §3.4 schema fields ----

    interrupt_id: UUID
    session_id: UUID
    agent_instance: str          # e.g. "executor-task1-run1"
    agent_type: str              # "executor" | "reviewer" | "responder"
    agent_state: str = "INTERRUPT"  # always "INTERRUPT" when checkpointed

    reply_chain: list[UUID] = Field(default_factory=list)
    task_context: dict[str, Any] = Field(default_factory=dict)

    interrupt_message: str       # the human-facing question
    interrupt_options: list[str] = Field(default_factory=list)  # choices presented

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: int = 86400     # 24h default (Q1: configurable per interrupt ✅)

    human_response: str | None = None  # filled on resume
    resolved_at: datetime | None = None

    # ------------------------------------------------------------------
    # Derived properties  —  §3.4 checks
    # ------------------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        """True when the TTL has elapsed since *created_at*.  §3.4 ¶2.

        Used by the TTL monitor to auto-terminate stale interrupts.
        """
        elapsed = (datetime.now(UTC) - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    @property
    def is_resolved(self) -> bool:
        """True after the human has responded (§3.4 ¶5)."""
        return self.human_response is not None

    @property
    def age_seconds(self) -> float:
        """Seconds since this checkpoint was created."""
        return (datetime.now(UTC) - self.created_at).total_seconds()

    # ------------------------------------------------------------------
    # Disk persistence  —  Spec 001 §3.4 ¶2
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoints_dir(session_id: UUID, base_dir: Path | None = None) -> Path:
        """Directory where checkpoints for *session_id* are stored.

        §3.4 ¶2: ``~/.llend/sessions/{session_id}/checkpoints/``
        """
        base = base_dir or _DEFAULT_BASE_DIR
        return base / "sessions" / str(session_id) / "checkpoints"

    def disk_path(self, base_dir: Path | None = None) -> Path:
        """Full on-disk path for this checkpoint's JSON file.

        §3.4 schema: ``{interrupt_id}.json``
        """
        return self._checkpoints_dir(self.session_id, base_dir) / f"{self.interrupt_id}.json"

    def save(self, base_dir: Path | None = None) -> Path:
        """Persist this checkpoint to disk as JSON.  §3.4 ¶2 step 2.

        Returns the file path.  Creates parent directories if needed.
        Q2: JSON format (human-readable, debuggable).
        """
        path = self.disk_path(base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(
        cls,
        session_id: UUID,
        interrupt_id: UUID,
        base_dir: Path | None = None,
    ) -> Checkpoint | None:
        """Load a checkpoint from disk (§3.4 ¶5), or ``None`` if not found."""
        path = cls._checkpoints_dir(session_id, base_dir) / f"{interrupt_id}.json"
        if not path.exists():
            return None
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def delete(self, base_dir: Path | None = None) -> None:
        """Remove this checkpoint's JSON file from disk.  Idempotent.

        Called when an agent is killed to clean up stale state (§3.3: any → DEAD).
        """
        path = self.disk_path(base_dir)
        if path.exists():
            path.unlink()

    @classmethod
    def list_for_session(
        cls,
        session_id: UUID,
        base_dir: Path | None = None,
    ) -> list[UUID]:
        """Return interrupt IDs of all checkpoints on disk for a session.

        Useful for audit / debugging — "show all pending interrupts for session X".
        """
        dir_path = cls._checkpoints_dir(session_id, base_dir)
        if not dir_path.exists():
            return []
        return sorted(
            UUID(p.stem) for p in dir_path.glob("*.json") if p.stem != ""
        )
