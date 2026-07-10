"""Executor agent — the "do-er" in the llend harness topology.

Each Executor is spawned fresh for a single task, runs a ReAct (tool-use)
loop with an LLM, invokes actions via ActionDispatcher, and returns
``task.result`` to the Orchestrator.  The agent is **stateless** beyond
the current task.

Spec references
===============
- **Spec 005 §2** → ExecutorAgent role, lifecycle, ReAct loop
- **Spec 001 §2.2** → ``task.dispatch`` / ``task.result`` message types
- **Spec 002 §4.4** → ActionDispatcher (tool invocation inside Executor)
"""

from llend.executor.agent import ExecutorAgent

__all__ = ["ExecutorAgent"]
