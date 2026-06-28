"""Checkpoint system for interrupt / human-in-the-loop.

When an agent raises an interrupt the runtime freezes its state into a
``Checkpoint``, persists it as JSON, and blocks until a human responds.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Checkpoint model
# ---------------------------------------------------------------------------


class Checkpoint(BaseModel):
    """A frozen snapshot of an agent blocked on a human decision."""

    interrupt_id: UUID
    session_id: UUID
    agent_instance: str
    agent_type: str  # "executor" | "reviewer" | "responder"
    agent_state: str = "INTERRUPT"  # always INTERRUPT when checkpointed

    reply_chain: list[UUID] = Field(default_factory=list)
    task_context: dict[str, Any] = Field(default_factory=dict)

    interrupt_message: str
    interrupt_options: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: int = 86400  # 24 h default, configurable per interrupt

    human_response: str | None = None
    resolved_at: datetime | None = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        """True when the TTL has elapsed since *created_at*."""
        elapsed = (datetime.now(UTC) - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    @property
    def is_resolved(self) -> bool:
        """True after the human has responded."""
        return self.human_response is not None

    @property
    def age_seconds(self) -> float:
        """Seconds since this checkpoint was created."""
        return (datetime.now(UTC) - self.created_at).total_seconds()


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _base_path(session_id: UUID) -> Path:
    """Return the checkpoints directory for *session_id*."""
    return Path.home() / ".llend" / "sessions" / str(session_id) / "checkpoints"


def _checkpoint_path(interrupt_id: UUID, session_id: UUID) -> Path:
    """Return the full path to the checkpoint JSON file."""
    return _base_path(session_id) / f"{interrupt_id}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(checkpoint: Checkpoint) -> Path:
    """Persist *checkpoint* as JSON on disk.

    Creates parent directories if needed.  Returns the file path.
    """
    file_path = _checkpoint_path(checkpoint.interrupt_id, checkpoint.session_id)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    return file_path


def load_checkpoint(interrupt_id: UUID, session_id: UUID) -> Checkpoint | None:
    """Load a checkpoint from disk.

    Returns ``None`` when the file does not exist or the checkpoint has
    expired (expired checkpoints are silently removed).
    """
    file_path = _checkpoint_path(interrupt_id, session_id)
    if not file_path.exists():
        return None

    data = file_path.read_text(encoding="utf-8")
    checkpoint = Checkpoint.model_validate_json(data)

    # Silently clean up expired checkpoints
    if checkpoint.is_expired:
        file_path.unlink(missing_ok=True)
        return None

    return checkpoint


def delete_checkpoint(interrupt_id: UUID, session_id: UUID) -> bool:
    """Delete a checkpoint file from disk.

    Returns ``True`` if the file was deleted, ``False`` if it did not exist.
    """
    file_path = _checkpoint_path(interrupt_id, session_id)
    existed = file_path.exists()
    file_path.unlink(missing_ok=True)
    return existed


def cleanup_expired_checkpoints(session_id: UUID, max_age_seconds: int = 86400) -> int:
    """Remove all checkpoints for *session_id* older than *max_age_seconds*.

    Returns the number of files removed.
    """
    base = _base_path(session_id)
    if not base.exists():
        return 0

    removed = 0
    cutoff = datetime.now(UTC).timestamp() - max_age_seconds

    for child in base.iterdir():
        if child.is_file() and child.suffix == ".json":
            if child.stat().st_mtime < cutoff:
                child.unlink()
                removed += 1

    return removed
