"""Orchestrator package — the central hub that ties the harness together.  Spec 004.

Public API:
- ``OrchestratorAgent`` — the main long-lived coordinating agent
- ``OrchestratorConfig`` — all tunable settings
- ``SessionManager`` / ``SessionState`` — session lifecycle management
- ``ToolApprovalGate`` — Responder tool request gating
- ``ProgressReporter`` / ``ProgressEvent`` — human-visible progress updates
- ``execute_task_loop`` — the Executor → Reviewer → adjudicate cycle
"""

from llend.orchestrator.agent import OrchestratorAgent
from llend.orchestrator.config import OrchestratorConfig
from llend.orchestrator.executor import execute_task_loop, REVIEWER_SYSTEM_PROMPT
from llend.orchestrator.gate import GateDecision, ToolApprovalGate
from llend.orchestrator.progress import ProgressEvent, ProgressReporter
from llend.orchestrator.session import SessionManager, SessionState

__all__ = [
    "OrchestratorAgent",
    "OrchestratorConfig",
    "SessionManager",
    "SessionState",
    "ToolApprovalGate",
    "GateDecision",
    "ProgressReporter",
    "ProgressEvent",
    "execute_task_loop",
    "REVIEWER_SYSTEM_PROMPT",
]
