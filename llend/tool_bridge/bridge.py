"""Tool Bridge — global action→tool binding resolution.

Reads ``mappings.toml``, validates imports at startup (fail-fast), and serves
action bindings to ``SkillRegistry`` during skill resolution.  Supports config
merging (TOML → env vars → per-skill overrides) and hot-reload.

Spec references
===============
- **§5** → Tool Bridge concept & TOML format
- **§5.2** → ``ToolBridge`` class (resolve / resolve_all / list_actions / validate_mapping)
- **§5.3** → Config merging (TOML + env + per-skill) & hot-reload
- **§4.4** → ``ActionBinding`` consumed by Executor's ``ActionDispatcher``
"""

from __future__ import annotations

import importlib
import logging
import os
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

        When *validate* is ``True`` (default), logs warnings for unimportable
        tools but does **not** crash — the harness can still start and run
        skills that don't depend on the missing tools.  Pass ``validate=False``
        to skip validation entirely; call ``validate_mapping()`` individually
        later.
        """
        self._mappings_path = mappings_path
        self._bindings: dict[str, ActionBinding] = {}

        raw = self._load_toml(mappings_path)
        self._parse_bindings_into(raw, self._bindings)

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
        except Exception:
            logger.warning(
                "Tool %r for action %r could not be imported",
                binding.tool, action_name, exc_info=True,
            )
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

        # Auto-populate input_schema from function signature if not already set.
        # This lets the Executor build proper tool definitions for the LLM.
        if binding.input_schema is None:
            binding.input_schema = _signature_to_schema(obj)

        return True

    # ------------------------------------------------------------------
    # §5.3 — config merging (TOML + env vars + per-skill overrides)
    # ------------------------------------------------------------------

    def resolve_with_overrides(
        self,
        action_name: str,
        env_overrides: dict[str, Any] | None = None,
        skill_overrides: dict[str, Any] | None = None,
    ) -> ActionBinding | None:
        """Resolve *action_name* with merged config.  §5.3.

        Precedence (highest to lowest):
        1. *skill_overrides* — per-skill config in handler.py or skill.md
        2. *env_overrides* — ``LLEND_TOOL_<ACTION>_<KEY>=value`` env vars
        3. TOML ``[actions.<name>.config]`` — static config in mappings.toml
        """
        binding = self.resolve(action_name)
        if binding is None:
            return None

        merged_config: dict[str, Any] = dict(binding.config)

        # Layer 2: env var overrides (§5.3)
        if env_overrides:
            merged_config.update(env_overrides)

        # Also check process environment for LLEND_TOOL_* vars
        prefix = f"LLEND_TOOL_{action_name.upper()}_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                merged_config[config_key] = self._coerce_env_value(value)

        # Layer 3: per-skill overrides (§5.3)
        if skill_overrides:
            merged_config.update(skill_overrides)

        return ActionBinding(
            action_name=binding.action_name,
            source=binding.source,
            tool=binding.tool,
            function=binding.function,
            handler_class=binding.handler_class,
            timeout_ms=binding.timeout_ms,
            retry=binding.retry,
            config=merged_config,
        )

    # ------------------------------------------------------------------
    # §5.3 — hot-reload
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-read ``mappings.toml`` and re-validate all bindings.  §5.3.

        Called by ``SkillRegistry.watch()`` when the TOML file changes.
        Validation failures are logged as warnings — the old bindings
        remain in place until the new ones pass validation.
        """
        logger.info("Hot-reload: re-reading %s", self._mappings_path)
        try:
            raw = self._load_toml(self._mappings_path)
            new_bindings: dict[str, ActionBinding] = {}
            self._parse_bindings_into(raw, new_bindings)

            # Validate new bindings
            failed: list[str] = []
            for name, binding in new_bindings.items():
                if binding.tool and self._check_importable(binding.tool, binding.function):
                    continue
                failed.append(name)

            if failed:
                logger.warning(
                    "Hot-reload: %d action(s) not importable, keeping old bindings: %s",
                    len(failed), ", ".join(failed),
                )
                # Keep failed actions from old bindings
                for name in failed:
                    if name in self._bindings:
                        new_bindings[name] = self._bindings[name]

            self._bindings = new_bindings
            logger.info("Hot-reload: %d action(s) loaded", len(self._bindings))
        except Exception:
            logger.exception("Hot-reload: failed to reload %s", self._mappings_path)

    @property
    def mappings_path(self) -> Path:
        """Path to the currently loaded mappings file.  §5.3."""
        return self._mappings_path

    def _load_toml(self, path: Path) -> dict[str, Any]:
        """Read the TOML file.  Uses stdlib ``tomllib`` (Python ≥ 3.11)."""
        raw_text = path.read_text(encoding="utf-8")
        return tomllib.loads(raw_text)

    def _parse_bindings_into(
        self, raw: dict[str, Any], target: dict[str, ActionBinding]
    ) -> None:
        """Parse ``[actions.*]`` sections into *target* dict.  §5.1.

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

            target[action_name] = binding

    def _validate_all(self) -> None:
        """Run ``validate_mapping()`` on every binding at startup.  §5.2.

        Missing tools are logged as warnings rather than crashing startup —
        the harness can still run skills that don't depend on the unavailable
        tools.
        """
        failed: list[str] = []
        for action_name in self._bindings:
            if not self.validate_mapping(action_name):
                failed.append(action_name)

        if failed:
            names = ", ".join(failed)
            logger.warning(
                "ToolBridge: %d action(s) have unimportable tools: %s. "
                "Skills depending on these actions will fail at runtime.",
                len(failed), names,
            )

        logger.info(
            "ToolBridge loaded %d action(s) from %s",
            len(self._bindings), self._mappings_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_importable(tool: str, function: str) -> bool:
        """Check *tool* module is importable and *function* exists on it.  §5.3."""
        try:
            mod = importlib.import_module(tool)
        except Exception:
            return False
        obj = mod
        for attr in function.split("."):
            try:
                obj = getattr(obj, attr)
            except AttributeError:
                return False
        return True


# ---------------------------------------------------------------------------
# signature → JSON Schema helper  (module-level)
# ---------------------------------------------------------------------------


def _signature_to_schema(func: object) -> dict[str, Any]:
    """Auto-generate a JSON Schema ``input_schema`` from *func*'s signature.

    Inspects parameter names, types, and defaults to produce a schema the
    LLM can use when deciding to call this tool.  ``**kwargs`` is ignored.
    """
    import inspect

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return {"type": "object", "properties": {}}

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue

        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            json_type = "string"
        else:
            # Handle generic types like list[str], dict[str, Any], str | None
            origin = getattr(annotation, "__origin__", None)
            if origin is not None:
                # It's a generic — use the origin type
                json_type = type_map.get(origin, "string")
            else:
                json_type = type_map.get(annotation, "string")

        prop: dict[str, Any] = {"type": json_type, "description": name}

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    } if required else {
        "type": "object",
        "properties": properties,
    }

    @staticmethod
    def _coerce_env_value(value: str) -> int | float | bool | str:
        """Coerce an env-var string value to the most specific type.  §5.3."""
        # bool
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        # int
        try:
            return int(value)
        except ValueError:
            pass
        # float
        try:
            return float(value)
        except ValueError:
            pass
        return value
