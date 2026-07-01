"""Tool Bridge — global action→tool binding resolution.

Reads ``mappings.toml``, validates imports at startup (fail-fast), and serves
action bindings to ``SkillRegistry`` during skill resolution.

Spec references
===============
- **§5** → Tool Bridge concept & TOML format
- **§5.2** → ``ToolBridge`` class (resolve / resolve_all / list_actions / validate_mapping)
- **§4.4** → ``ActionBinding`` consumed by Executor's ``ActionDispatcher``
"""

from __future__ import annotations

import importlib
import logging
import tomllib
from pathlib import Path
from typing import Any

from llend.registry.models import ActionBinding

logger = logging.getLogger(__name__)


class ToolBridge:
    """Resolves global action names → concrete tool implementations.  Spec 002 §5.2.

    Wraps ``mappings.toml`` with startup validation (fails fast if a tool is
    not importable) and provides a query interface for ``SkillRegistry``.
    """

    def __init__(self, mappings_path: Path, *, validate: bool = True) -> None:
        """Load and optionally validate all action bindings from *mappings_path*.

        When *validate* is ``True`` (default), raises ``ValueError`` immediately
        if a configured tool cannot be imported (fail-fast — §5.3).  Pass
        ``validate=False`` for development or when tools are added incrementally;
        call ``validate_mapping()`` individually later.
        """
        self._mappings_path = mappings_path
        self._bindings: dict[str, ActionBinding] = {}

        raw = self._load_toml(mappings_path)
        self._parse_bindings(raw)

        if validate:
            self._validate_all()

    # ------------------------------------------------------------------
    # §5.2 — resolve / resolve_all / list_actions
    # ------------------------------------------------------------------

    def resolve(self, action_name: str) -> ActionBinding | None:
        """Look up a global action by name.  Returns ``None`` if not found.  §5.2."""
        return self._bindings.get(action_name)

    def resolve_all(self) -> dict[str, ActionBinding]:
        """All registered global actions with their bindings.  §5.2."""
        return dict(self._bindings)

    def list_actions(self) -> list[str]:
        """All registered global action names.  §5.2."""
        return list(self._bindings.keys())

    # ------------------------------------------------------------------
    # §5.2 — validate_mapping
    # ------------------------------------------------------------------

    def validate_mapping(self, action_name: str) -> bool:
        """Check that the mapped tool is importable and the function exists.  §5.2.

        Uses ``importlib.import_module(tool)`` → ``getattr(module, function)``.
        Returns ``True`` if importable, ``False`` otherwise.

        Called at startup for every binding (fail-fast).  ``SkillRegistry`` may
        also call this before adding a binding during hot-reload.
        """
        binding = self._bindings.get(action_name)
        if binding is None or binding.source != "global" or binding.tool is None:
            return False

        try:
            mod = importlib.import_module(binding.tool)
        except ImportError:
            logger.warning("Tool %r for action %r is not installed", binding.tool, action_name)
            return False

        # Walk dotted function path (e.g. "AsyncWebCrawler.arun")
        try:
            obj = mod
            for attr in binding.function.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            logger.warning(
                "Function %r not found on tool %r (action %r)",
                binding.function,
                binding.tool,
                action_name,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Internal — TOML loading & parsing
    # ------------------------------------------------------------------

    def _load_toml(self, path: Path) -> dict[str, Any]:
        """Read the TOML file.  Uses stdlib ``tomllib`` (Python ≥ 3.11)."""
        raw_text = path.read_text(encoding="utf-8")
        return tomllib.loads(raw_text)

    def _parse_bindings(self, raw: dict[str, Any]) -> None:
        """Parse ``[actions.*]`` sections into ``ActionBinding`` instances.

        Expected TOML structure (§5.1)::

            [actions.fetch_web_page]
            tool = "crawl4ai"
            function = "AsyncWebCrawler.arun"
            timeout_ms = 30000
            retry = 3

            [actions.fetch_web_page.config]
            stealth_mode = true
            user_agent = "llend-harness/0.1"
        """
        actions_table = raw.get("actions", {})
        if not actions_table:
            return

        for action_name, entry in actions_table.items():
            if not isinstance(entry, dict):
                logger.warning("Skipping malformed action %r — expected table", action_name)
                continue

            binding = ActionBinding(
                action_name=action_name,
                source="global",
                tool=entry.get("tool"),
                function=entry.get("function", ""),
                timeout_ms=entry.get("timeout_ms", 30000),
                retry=entry.get("retry", 0),
                config=entry.get("config", {}),
            )

            if not binding.tool or not binding.function:
                logger.warning(
                    "Action %r is missing required 'tool' or 'function' field", action_name
                )
                continue

            self._bindings[action_name] = binding

    def _validate_all(self) -> None:
        """Run ``validate_mapping()`` on every binding at startup.  §5.2."""
        failed: list[str] = []
        for action_name in self._bindings:
            if not self.validate_mapping(action_name):
                failed.append(action_name)

        if failed:
            names = ", ".join(failed)
            raise ValueError(
                f"ToolBridge startup: {len(failed)} action(s) have unimportable tools: {names}"
            )

        logger.info(
            "ToolBridge loaded %d action(s) from %s",
            len(self._bindings), self._mappings_path,
        )
