"""Responder tool approval gate — the Orchestrator's guard for tool requests.

When the Responder sends ``respond.request_tool``, the Orchestrator evaluates
whether the request should be auto-approved, rejected, or escalated to the
human for a decision.

Spec references
===============
- **§9.1** → The approval flow (4-step decision tree)
- **§9.2** → Cheap vs expensive heuristic table
- **§6.1 (Spec 003)** → Responder tool request flow
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from llend.registry.models import ActionBinding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GateDecision(BaseModel):
    """The Orchestrator's decision on a Responder tool request.  §9.1."""

    approved: bool
    reason: str = ""
    needs_human: bool = False
    cached_result: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Cheap vs Expensive  —  §9.2
# ---------------------------------------------------------------------------


_EXPENSIVE_ACTIONS = frozenset({"fetch_web_page", "crawl", "crawl_multi_page", "scrape"})


def _is_cheap(
    binding: ActionBinding,
    skill_name: str,
    tool_params: dict[str, Any] | None = None,
    auto_approve_timeout_ms: int = 10000,
) -> bool:
    """Determine whether a tool request qualifies as "cheap".  §9.2.

    Returns True if:
    - ``timeout_ms < auto_approve_timeout_ms`` (default 10s)
    - Skill has no ``fetch_web_page`` or ``crawl*`` actions
    - Not classified as expensive

    Returns False if:
    - Skill is ``data_provider`` with any ``platform=*``
    - Skill has ``max_items > 100``
    """
    params = tool_params or {}

    # data_provider with platform=* is always expensive  §9.2
    if skill_name == "data_provider" and "platform" in params:
        return False

    # Timeout heuristic
    if binding.timeout_ms < auto_approve_timeout_ms:
        return True

    # Action-based heuristic
    action_lower = binding.action_name.lower()
    for expensive in _EXPENSIVE_ACTIONS:
        if expensive in action_lower:
            return False

    # Config-based heuristic
    config = binding.config or {}
    if config.get("max_items", 0) > 100:
        return False

    return True


# ---------------------------------------------------------------------------
# Tool Approval Gate  —  §9.1
# ---------------------------------------------------------------------------


@dataclass
class ToolApprovalGate:
    """Gatekeeper for Responder tool requests.  §9.1.

    Tracks tool requests per conversation turn so it can flag excessive
    requests (§9.1 step 4).  Also maintains a simple in-memory cache of
    recent tool results (§9.1 step 3).

    Parameters
    ----------
    auto_approve_timeout_ms:
        Threshold for the cheap heuristic.  Default 10 000 ms per §13.1.
    max_requests_per_turn:
        Warn threshold — if the Responder exceeds this, escalate to human.
        Default 3 per §13.1.
    """

    auto_approve_timeout_ms: int = 10000
    max_requests_per_turn: int = 3

    # Per-turn tracking
    _request_count: int = field(default=0, init=False)
    _recent_results: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def reset_turn(self) -> None:
        """Reset the per-turn request counter.  Call on each new user message."""
        self._request_count = 0

    def evaluate(
        self,
        suggested_skill: str,
        tool_params: dict[str, Any] | None,
        registry: Any,  # SkillRegistry — avoid circular import
    ) -> GateDecision:
        """Evaluate a Responder tool request through the 4-step gate.  §9.1.

        1. Skill exists in registry?
        2. Is it cheap? (timeout < 10s, no crawl/scrape)
        3. Already ran with same params this session?
        4. 3+ tool requests in this turn?
        """
        self._request_count += 1
        params = tool_params or {}

        # Step 1: Skill exists?  §9.1
        skill = registry.get(suggested_skill)
        if skill is None:
            return GateDecision(
                approved=False,
                reason=f"Skill '{suggested_skill}' not available in registry.",
            )

        # Step 2: Is it cheap?  §9.1, §9.2
        binding = skill.action_bindings.get(suggested_skill) if skill.action_bindings else None
        if binding is not None and _is_cheap(binding, suggested_skill, params, self.auto_approve_timeout_ms):
            cache_key = self._make_cache_key(suggested_skill, params)
            if cache_key in self._recent_results:
                # Step 3: Cache hit  §9.1
                return GateDecision(
                    approved=True,
                    reason="Cached result from earlier run.",
                    cached_result=self._recent_results[cache_key],
                )
            return GateDecision(
                approved=True,
                reason=f"Cheap operation (timeout < {self.auto_approve_timeout_ms}ms).",
            )

        # Step 3: Cache check (for non-cheap too)  §9.1
        cache_key = self._make_cache_key(suggested_skill, params)
        if cache_key in self._recent_results:
            return GateDecision(
                approved=True,
                reason="Cached result from earlier run.",
                cached_result=self._recent_results[cache_key],
            )

        # Step 4: Excessive requests?  §9.1
        if self._request_count >= self.max_requests_per_turn:
            return GateDecision(
                approved=False,
                needs_human=True,
                reason=(
                    f"Responder has made {self._request_count} tool requests "
                    f"in this turn (max {self.max_requests_per_turn}). "
                    "Needs human approval."
                ),
            )

        # Expensive operation — escalate to human  §9.1
        return GateDecision(
            approved=False,
            needs_human=True,
            reason=(
                f"Responder wants to use '{suggested_skill}'. "
                "This is an expensive operation. Human approval required."
            ),
        )

    def cache_result(self, skill_name: str, params: dict[str, Any], result: dict[str, Any]) -> None:
        """Store a tool result in the in-memory cache.  §9.1 step 3."""
        cache_key = self._make_cache_key(skill_name, params)
        self._recent_results[cache_key] = result
        # Prune old entries if cache grows too large
        if len(self._recent_results) > 50:
            # Drop oldest (simplistic — dict insertion order preserved in 3.7+)
            oldest = next(iter(self._recent_results))
            del self._recent_results[oldest]

    @staticmethod
    def _make_cache_key(skill_name: str, params: dict[str, Any]) -> str:
        """Deterministic cache key from skill name + params."""
        raw = json.dumps({"skill": skill_name, "params": params}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
