"""Input/output wiring — connecting upstream task outputs to downstream inputs.

When task N produces a ``ScrapeResult`` and task N+1 expects ``dataset: list[dict]``,
the Orchestrator needs to bridge them.  This module implements the four
auto-unwrap rules from Spec 004 §6.2.

Spec references
===============
- **§6.1** → The wiring problem
- **§6.2** → Auto-unwrap convention (4 rules)
- **§6.3** → Wiring in practice (code sketch)
- **§6.4** → Type coercion (Pydantic → dict)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from llend.registry.pipeline import TaskSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-unwrap  —  §6.2
# ---------------------------------------------------------------------------


def has_single_list_field(obj: Any) -> bool:
    """Return True if *obj* is a Pydantic model with exactly one field of list type.  §6.2."""
    if not isinstance(obj, BaseModel):
        return False
    fields = list(type(obj).model_fields.keys())
    if len(fields) != 1:
        return False
    annotation = type(obj).model_fields[fields[0]].annotation
    if annotation is None:
        return False
    origin = getattr(annotation, "__origin__", None)
    return origin is list


def get_list_field(obj: BaseModel) -> list[Any]:
    """Extract the single list field from a Pydantic model.  §6.2 rule 2."""
    field_name = next(iter(type(obj).model_fields))
    return getattr(obj, field_name)


def wire_upstream_output(
    upstream_output: Any,
    upstream_skill_name: str,
    downstream_task: TaskSpec,
) -> dict[str, Any]:
    """Connect an upstream task's output to a downstream task's input.  §6.2.

    Applies the four auto-unwrap rules in order:

    1. **Exact match**: If upstream output type name == downstream input type
       name → pass directly.
    2. **Wrapper unwrap**: If upstream output is a Pydantic model with a
       **single list field** → unwrap it.
    3. **Named field match**: If upstream output has a field whose name
       matches the downstream input name → extract that field.
    4. **Pass-through**: If none of the above → pass the entire upstream
       output object.

    Returns a dict of ``{param_name: wired_value}`` suitable for merging
    into ``TaskSpec.task_spec``.
    """
    wired: dict[str, Any] = {}

    if not downstream_task.input_from or upstream_skill_name not in downstream_task.input_from:
        return wired

    ref_name = f"{upstream_skill_name}_ref"

    # Rule 1: Exact match — if upstream output type name == downstream input
    # type name, pass directly.  §6.2
    upstream_type_name = type(upstream_output).__name__
    for input_param, input_type_spec in downstream_task.task_spec.items():
        if isinstance(input_type_spec, str) and input_type_spec == upstream_type_name:
            wired[input_param] = upstream_output
            logger.debug(
                "Exact-match: %s → %s.%s",
                upstream_type_name,
                downstream_task.skill_name,
                input_param,
            )
            return wired
    # The downstream task_spec already has {dep}_ref keys from the pipeline;
    # we enhance them with the actual wired data.
    if has_single_list_field(upstream_output):
        # Rule 2: Wrapper unwrap — §6.2
        unwrapped = get_list_field(upstream_output)
        logger.debug(
            "Auto-unwrap: %s → list[%d items] for %s",
            upstream_type_name,
            len(unwrapped),
            downstream_task.skill_name,
        )
        wired[ref_name] = unwrapped
    elif isinstance(upstream_output, BaseModel):
        # Rule 3: Named field match — check if any input param name matches
        # a field on the upstream output model
        fields = type(upstream_output).model_fields
        matched_any = False
        for input_param in downstream_task.task_spec:
            if input_param in fields:
                wired[input_param] = getattr(upstream_output, input_param)
                matched_any = True
                logger.debug(
                    "Named-field match: %s.%s → %s.%s",
                    upstream_type_name,
                    input_param,
                    downstream_task.skill_name,
                    input_param,
                )
        if not matched_any:
            # Rule 4: Pass-through
            wired[ref_name] = upstream_output
            logger.debug(
                "Pass-through: %s → %s (no matching fields)",
                upstream_type_name,
                downstream_task.skill_name,
            )
    else:
        # Rule 4: Pass-through (non-Pydantic output)
        wired[ref_name] = upstream_output
        logger.debug(
            "Pass-through (raw): %s → %s",
            type(upstream_output).__name__,
            downstream_task.skill_name,
        )

    return wired


# ---------------------------------------------------------------------------
# Type coercion  —  §6.4
# ---------------------------------------------------------------------------


def coerce_to_expected_type(data: Any, expected_type: str) -> Any:
    """Best-effort type coercion for downstream consumption.  §6.4.

    If downstream expects ``list[dict]`` but receives ``list[ProductListing]``
    (Pydantic models), calls ``.model_dump()`` on each item.
    """
    if expected_type in ("list[dict]", "list[dict[str, Any]]") and isinstance(data, list):
        coerced: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, BaseModel):
                coerced.append(item.model_dump())
            elif isinstance(item, dict):
                coerced.append(item)
            else:
                coerced.append({"value": item})
        return coerced
    return data  # pass-through
