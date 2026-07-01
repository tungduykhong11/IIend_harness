"""Skill Registry — discovery, validation, resolution, and hot-reload.

Scans ``skills_dir`` recursively for skill.md files, parses their YAML
frontmatter, validates them against the tool bridge, and resolves them into
fully-populated ``Skill`` objects ready for dispatch.

Spec references
===============
- **§6** → Skill Registry concept
- **§6.1** → ``SkillRegistry`` class (discover / validate / resolve / watch)
- **§4.3** → custom action discovery
- **§9** → skill_context payload construction
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from llend.registry.models import ActionBinding, ResolutionError, Skill, SkillMeta, ValidationIssue
from llend.registry.parser import parse_inputs
from llend.registry.validator import (
    _discover_custom_actions,
    _discover_handler_class,
    _import_module_from_path,
    validate_skill,
)
from llend.tool_bridge.bridge import ToolBridge

logger = logging.getLogger(__name__)

# YAML frontmatter delimiter — §2.2
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class SkillRegistry:
    """Discover, validate, and resolve skills.  Spec 002 §6.1.

    Constructor takes *skills_dir* (path to the skills root) and a *tool_bridge*
    for global action resolution.
    """

    def __init__(self, skills_dir: Path, tool_bridge: ToolBridge) -> None:
        self._skills_dir = skills_dir
        self._tool_bridge = tool_bridge
        self._metas: dict[str, SkillMeta] = {}  # discovered metadata, keyed by name
        self._skills: dict[str, Skill] = {}  # resolved skills, keyed by name

    # ------------------------------------------------------------------
    # §6.1 — Discovery
    # ------------------------------------------------------------------

    async def discover(self) -> list[SkillMeta]:
        """Scan *skills_dir* recursively for skill.md files, parse frontmatter.

        Returns metadata for all discovered skills.  Does NOT validate or resolve.
        Call ``validate()`` and ``resolve()`` separately.  §6.1.
        """
        self._metas.clear()
        discovered: list[SkillMeta] = []

        for md_path in sorted(self._skills_dir.rglob("skill.md")):
            try:
                meta = self._parse_skill_md(md_path)
                discovered.append(meta)
                self._metas[meta.name] = meta
                logger.info(
                    "Discovered skill %r v%s at %s",
                    meta.name, meta.version, md_path.parent,
                )
            except Exception:
                logger.exception("Failed to parse %s — skipping", md_path)

        return discovered

    # ------------------------------------------------------------------
    # §6.1 — Validation
    # ------------------------------------------------------------------

    def validate(self, meta: SkillMeta) -> list[ValidationIssue]:
        """Check that *meta* is usable.  §6.1.

        Validates five things:
        1. All declared actions resolve (global bridge OR custom handler)
        2. Input/output types are parseable
        3. If models.py exists, output model is importable and is a BaseModel
        4. handler.py is importable, has expected class, methods have docstrings
        5. Dependencies reference existing skill names

        Returns a list of ``ValidationIssue``.  Empty list = valid.
        Errors block ``resolve()``; warnings allow it.
        """
        known = set(self._metas.keys())
        return validate_skill(meta, self._tool_bridge, self._skills_dir, known_names=known)

    # ------------------------------------------------------------------
    # §6.1 — Resolution
    # ------------------------------------------------------------------

    def resolve(self, name: str, version: str | None = None) -> Skill:
        """Fully resolve a skill ready for dispatch.  §6.1.

        1. Look up the discovered SkillMeta
        2. Validate — if any error-severity issues exist, raise ``ResolutionError``
        3. Merge global tool bridge actions + custom handler actions
        4. Load Pydantic models if models.py exists
        5. Instantiate handler if handler.py exists
        6. Return ``Skill`` with all ``action_bindings`` populated
        """
        meta = self._metas.get(name)
        if meta is None:
            raise ResolutionError(name, [
                ValidationIssue(
                    severity="error", field="name",
                    message=f"Skill {name!r} not discovered",
                )
            ])

        # If version is None, resolve latest (current behaviour: only one version)
        if version is not None and version != meta.version:
            raise ResolutionError(name, [
                ValidationIssue(
                    severity="error",
                    field="version",
                    message=f"Version {version!r} not found (have {meta.version!r})",
                )
            ])

        # Validate first
        issues = self.validate(meta)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise ResolutionError(name, issues)

        skill_dir = self._skills_dir / name
        skill_md = (skill_dir / "skill.md").read_text(encoding="utf-8")

        # Merge action bindings: global + custom
        action_bindings = self._merge_action_bindings(meta, skill_dir)

        # Load output schema from models.py
        output_schema, input_schemas = self._load_schemas(meta, skill_dir)

        # Instantiate handler if present
        handler = self._instantiate_handler(skill_dir, meta.name)

        skill = Skill(
            name=meta.name,
            version=meta.version,
            description=meta.description,
            inputs=meta.inputs,
            outputs=meta.outputs,
            actions=meta.actions,
            dependencies=meta.dependencies,
            enforcement=meta.enforcement,
            path=skill_dir,
            skill_md=skill_md,
            output_schema=output_schema,
            input_schemas=input_schemas,
            action_bindings=action_bindings,
            handler=handler,
        )

        self._skills[name] = skill
        logger.info("Resolved skill %r v%s", name, meta.version)
        return skill

    def resolve_all(self) -> dict[str, Skill]:
        """Resolve all discovered skills.  Keyed by name.  §6.1."""
        for name in list(self._metas):
            if name not in self._skills:
                try:
                    self.resolve(name)
                except ResolutionError:
                    logger.warning("Skipping %r — resolution failed", name)
        return dict(self._skills)

    # ------------------------------------------------------------------
    # §6.1 — Query
    # ------------------------------------------------------------------

    def list_skills(self) -> dict[str, list[SkillMeta]]:
        """Skills grouped by subdirectory (category).  §6.1."""
        grouped: dict[str, list[SkillMeta]] = {}
        for meta in self._metas.values():
            # Use the immediate parent dir of the skill as category
            skill_dir = self._skills_dir / meta.name
            cat = skill_dir.parent.name if skill_dir.parent != self._skills_dir else "__root__"
            grouped.setdefault(cat, []).append(meta)
        return grouped

    def get(self, name: str) -> Skill | None:
        """Get a previously resolved skill by name.  §6.1."""
        return self._skills.get(name)

    # ------------------------------------------------------------------
    # §6.1 — Hot Reload
    # ------------------------------------------------------------------

    async def watch(self) -> None:
        """Start filesystem watcher on *skills_dir* and *mappings.toml*.  §6.1, §5.3.

        On skill file change: re-discover affected skill, re-validate, update cache.
        On mappings.toml change: call ``ToolBridge.reload()``.
        Runs as a background asyncio task.  Uses ``watchdog`` if installed;
        gracefully degrades to a no-op otherwise.
        """
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.info("watchdog not installed — skill hot-reload disabled")
            return

        registry_ref = self

        class SkillChangeHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.is_directory:
                    return
                path = Path(event.src_path)
                # §6.1: skill file changes
                if path.name in ("skill.md", "handler.py", "models.py"):
                    logger.info("Hot-reload: change detected in %s", path)
                    skill_dir = path.parent
                    md_path = skill_dir / "skill.md"
                    if md_path.exists():
                        try:
                            meta = registry_ref._parse_skill_md(md_path)
                            registry_ref._metas[meta.name] = meta
                            # Invalidate cached resolved skill
                            registry_ref._skills.pop(meta.name, None)
                            logger.info("Hot-reload: re-parsed %r", meta.name)
                        except Exception:
                            logger.exception("Hot-reload: failed to re-parse %s", md_path)
                # §5.3: mappings.toml changes
                elif path.name == "mappings.toml":
                    logger.info("Hot-reload: mappings.toml changed, reloading")
                    registry_ref._tool_bridge.reload()

        observer = Observer()
        observer.schedule(SkillChangeHandler(), str(self._skills_dir), recursive=True)
        # Also watch tool_bridge mappings.toml — §5.3
        mappings_dir = str(self._tool_bridge.mappings_path.parent)
        observer.schedule(SkillChangeHandler(), mappings_dir, recursive=False)
        observer.start()
        logger.info("Hot-reload watcher started on %s + %s", self._skills_dir, mappings_dir)

        # Run until cancelled
        try:
            import asyncio
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            observer.stop()
            observer.join()
            logger.info("Hot-reload watcher stopped")

    # ------------------------------------------------------------------
    # Internal — skill.md parsing
    # ------------------------------------------------------------------

    def _parse_skill_md(self, md_path: Path) -> SkillMeta:
        """Parse YAML frontmatter from a skill.md file.  §2.2–§2.3."""
        text = md_path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.search(text)
        if not match:
            raise ValueError(f"No YAML frontmatter found in {md_path}")

        raw_yaml = match.group(1)
        fm = yaml.safe_load(raw_yaml)
        if not isinstance(fm, dict):
            raise ValueError(f"Frontmatter in {md_path} is not a YAML mapping")

        # Parse inputs string → dict
        raw_inputs = fm.get("inputs", "")
        if isinstance(raw_inputs, str):
            inputs = parse_inputs(raw_inputs)
        else:
            inputs = raw_inputs or {}

        # Ensure actions is a list
        actions = fm.get("actions", [])
        if isinstance(actions, str):
            actions = [a.strip() for a in actions.split(",")]

        # Ensure dependencies is a list
        deps = fm.get("dependencies", [])
        if isinstance(deps, str):
            deps = [d.strip() for d in deps.split(",") if d.strip()]

        return SkillMeta(
            name=fm.get("name", md_path.parent.name),
            version=str(fm.get("version", "0.1.0")),
            description=str(fm.get("description", "")),
            inputs=inputs,
            outputs=str(fm.get("outputs", "None")),
            actions=list(actions),
            dependencies=list(deps),
            enforcement=fm.get("enforcement", "suggested"),
        )

    # ------------------------------------------------------------------
    # Internal — action binding merge
    # ------------------------------------------------------------------

    def _merge_action_bindings(
        self, meta: SkillMeta, skill_dir: Path
    ) -> dict[str, ActionBinding]:
        """Merge global tool bridge bindings with custom handler actions.  §4.2."""
        bindings: dict[str, ActionBinding] = {}

        custom_actions = _discover_custom_actions(skill_dir)
        handler_cls = _discover_handler_class(
            _import_module_from_path("handler", skill_dir / "handler.py"),
            meta.name,
        ) if (skill_dir / "handler.py").exists() else None

        for action_name in meta.actions:
            # Try global first
            global_binding = self._tool_bridge.resolve(action_name)
            if global_binding is not None:
                bindings[action_name] = global_binding
                continue

            # Try custom
            if action_name in custom_actions:
                bindings[action_name] = ActionBinding(
                    action_name=action_name,
                    source="custom",
                    function=action_name,
                    handler_class=handler_cls.__name__ if handler_cls else None,
                )
                continue

            # Unresolved — should have been caught by validate()
            pass

        return bindings

    # ------------------------------------------------------------------
    # Internal — schema loading
    # ------------------------------------------------------------------

    def _load_schemas(
        self, meta: SkillMeta, skill_dir: Path
    ) -> tuple[dict | None, dict[str, dict]]:
        """Load output_schema + input_schemas from models.py.  §3.3."""
        output_schema: dict | None = None
        input_schemas: dict[str, dict] = {}

        primitives = {"str", "int", "float", "bool", "list", "dict", "list[dict]", "None"}
        models_path = skill_dir / "models.py"

        if models_path.exists():
            module = _import_module_from_path("models", models_path)
            if module is not None:
                # Output schema
                output = meta.outputs.strip()
                if output not in primitives:
                    model_cls = getattr(module, output, None)
                    if model_cls is not None:
                        try:
                            output_schema = model_cls.model_json_schema()
                        except Exception:
                            logger.exception("Failed to generate JSON schema for %r", output)

                # Input schemas — generate JSON Schema for Pydantic models,
                # simple type descriptors for primitives  §3.3
                for param_name, type_spec in meta.inputs.items():
                    # Strip default value part for model lookup
                    type_name = type_spec.split("=")[0].strip()
                    model_cls = getattr(module, type_name, None)
                    if model_cls is not None:
                        try:
                            input_schemas[param_name] = model_cls.model_json_schema()
                            continue
                        except Exception:
                            logger.exception(
                                "Failed to generate JSON schema for input %r", param_name
                            )
                    input_schemas[param_name] = {"type": type_spec}
            else:
                # No models.py — generate primitive input descriptors
                for param_name, type_spec in meta.inputs.items():
                    input_schemas[param_name] = {"type": type_spec}

        return output_schema, input_schemas

    # ------------------------------------------------------------------
    # Internal — handler instantiation
    # ------------------------------------------------------------------

    def _instantiate_handler(self, skill_dir: Path, skill_name: str) -> object | None:
        """Instantiate the handler class from handler.py.  §4.3."""
        handler_path = skill_dir / "handler.py"
        if not handler_path.exists():
            return None

        module = _import_module_from_path("handler", handler_path)
        if module is None:
            return None

        handler_cls = _discover_handler_class(module, skill_name)
        if handler_cls is None:
            return None

        try:
            return handler_cls()
        except Exception:
            logger.exception("Failed to instantiate handler for %r", skill_name)
            return None
