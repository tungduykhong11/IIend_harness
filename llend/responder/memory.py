"""User profile persistence for the Responder.  Allows preferences to survive
across sessions without needing a vector database or external service.

Spec references
===============
- **§9.1** → ``UserProfile`` — persisted preferences (platforms, budget, categories, persona)
- **§9.2** → ``UserProfile.load()`` / ``UserProfile.save()`` — JSON persistence
- **§9.3** → Default path ``~/.llend/user_profile.json``
- **§9.4** → Preference extraction — Orchestrator updates profile after each session
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from llend.responder.persona import Persona


class UserProfile(BaseModel):
    """Persistent user preferences loaded at session start.  §9.1.

    Stored as JSON at ``~/.llend/user_profile.json``.  The Orchestrator
    extracts preference signals after each session and updates the profile
    via ``UserProfile.save()``.  §9.4.
    """

    preferred_platforms: list[str] = Field(default_factory=list)
    budget_conscious: bool = False
    favorite_categories: list[str] = Field(default_factory=list)
    persona_preference: Persona = Persona.AUTO
    custom_notes: dict[str, str] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))  # §9.1

    # ------------------------------------------------------------------
    # persistence helpers  §9.2
    # ------------------------------------------------------------------

    @classmethod
    def _default_path(cls) -> Path:
        """Return the default profile path: ``~/.llend/user_profile.json``.  §9.3."""
        return Path.home() / ".llend" / "user_profile.json"

    @classmethod
    def load(cls, path: Path | None = None) -> "UserProfile":
        """Load the user profile from disk.  §9.2.

        Returns a default ``UserProfile`` if the file does not exist or is corrupt.
        """
        target = path or cls._default_path()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return cls()

    def save(self, path: Path | None = None) -> Path:
        """Persist the profile to disk as JSON.  §9.2.

        Creates parent directories if needed.  Updates ``last_updated`` before
        writing.
        """
        target = path or self._default_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now(UTC)
        target.write_text(
            self.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
        return target
