"""Orchestrator configuration — all tunable knobs for session orchestration.

Uses a Pydantic BaseModel (not BaseSettings — we read settings.toml manually
to avoid coupling to a particular env-var convention).  All values have
sensible defaults so the Orchestrator can start with zero configuration.

Spec references
===============
- **§13.1** → ``OrchestratorConfig`` — every field documented here
- **§13.2** → ``UserProfile`` (already in ``llend.responder.memory``)
- **§17** → model selection decisions (classification=Haiku, synthesis=Sonnet)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class OrchestratorConfig(BaseModel):
    """All tunable settings for the Orchestrator and session execution.  §13.1.

    Loaded from ``llend/settings.toml`` if present; otherwise every field
    falls back to the documented default.
    """

    # -- Model selection  §13.1 [orchestrator] ----------------------------

    classification_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Cheap & fast model for message classification.  §17.",
    )
    summarization_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Cheap model for TaskResultSummary generation.  §17.",
    )
    synthesis_model: str = Field(
        default="claude-sonnet-5",
        description="Capable model for final session synthesis.  §17.",
    )

    # -- Execution  §13.1 [execution] -------------------------------------

    max_retries_mandatory: int = Field(
        default=5,
        description="Max Executor→Reviewer loops for mandatory skills.  §4.4.",
    )
    max_retries_strict: int = Field(
        default=3,
        description="Max Executor→Reviewer loops for strict skills.  §4.4.",
    )
    max_retries_suggested: int = Field(
        default=1,
        description="Max Executor→Reviewer loops for suggested skills.  §4.4.",
    )
    task_timeout_default: int = Field(
        default=300,
        description="Default seconds before an Executor task is killed.  §13.1.",
    )
    review_timeout_default: int = Field(
        default=120,
        description="Default seconds before a Reviewer is killed.  §13.1.",
    )
    allow_parallel: bool = Field(
        default=False,
        description="Gate parallel task execution.  v0: sequential only.  §5.4.",
    )

    # -- Responder  §13.1 [responder] -------------------------------------

    responder_enabled: bool = Field(
        default=True,
        description="Whether to spawn a Responder at session start.  §13.1.",
    )
    tool_auto_approve_timeout_ms: int = Field(
        default=10000,
        description="Cheap threshold for auto-approving Responder tool requests.  §9.2.",
    )
    max_tool_requests_per_turn: int = Field(
        default=3,
        description="Warn if Responder exceeds this many tool requests in one turn.  §9.1.",
    )

    # -- Session  §13.1 [session] -----------------------------------------

    output_dir: str = Field(
        default="output",
        description="Directory for session artifacts (relative to cwd, or absolute).  §13.1.",
    )
    checkpoint_dir: str = Field(
        default="~/.llend/checkpoints",
        description="Directory for interrupt checkpoints.  §13.1.",
    )
    max_session_duration: int = Field(
        default=3600,
        description="Safety kill-switch — seconds before session auto-terminates.  §13.1.",
    )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: Path | str | None = None) -> "OrchestratorConfig":
        """Load settings from a TOML file, falling back to defaults.

        Looks for the ``[orchestrator]``, ``[execution]``, ``[responder]``,
        and ``[session]`` sections (§13.1) and merges their values into the
        config.  Missing sections or keys are silently ignored.
        """
        kwargs: dict[str, Any] = {}
        toml_path: Path | None = None

        if path is not None:
            toml_path = Path(path)
        else:
            # Try project-local, then cwd-local
            candidates = [
                Path("llend/settings.toml"),
                Path("settings.toml"),
            ]
            for c in candidates:
                if c.exists():
                    toml_path = c
                    break

        if toml_path is not None and toml_path.exists():
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore[no-redef]
                except ImportError:
                    tomllib = None  # type: ignore[assignment]

            if tomllib is not None:
                raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
                for section in ("orchestrator", "execution", "responder", "session"):
                    if section in raw and isinstance(raw[section], dict):
                        for key, value in raw[section].items():
                            # TOML uses hyphens; pydantic uses underscores
                            kwargs[key.replace("-", "_")] = value

        return cls(**kwargs)

    def get_max_retries(self, enforcement: str) -> int:
        """Return the max retry count for a given enforcement level.  §4.4.

        Parameters
        ----------
        enforcement:
            One of ``"mandatory"``, ``"strict"``, ``"suggested"``.
        """
        mapping: dict[str, int] = {
            "mandatory": self.max_retries_mandatory,
            "strict": self.max_retries_strict,
            "suggested": self.max_retries_suggested,
        }
        return mapping.get(enforcement, self.max_retries_suggested)
