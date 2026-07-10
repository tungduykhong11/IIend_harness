"""Reviewer agent — the adversarial "checker" in the llend harness topology.

Each Reviewer is spawned fresh after an Executor completes, receives
``task.review`` with a pre-constructed adversarial system prompt, performs
a single LLM call, and returns ``task.verdict`` to the Orchestrator.

Spec references
===============
- **Spec 005 §3** → ReviewerAgent thin wrapper implementation
- **Spec 001 §2.2** → ``task.review`` / ``task.verdict`` message types
- **Spec 004 §4.5** → adversarial system prompt (built by Orchestrator)
"""

from llend.reviewer.agent import ReviewerAgent

__all__ = ["ReviewerAgent"]
