"""CLI entry point — ``python -m llend``.  Spec 005 §6.

Bootstrap sequence (§6.1)
--------------------------
1. Load settings from llend/settings.toml (or use defaults)
2. Create LLMClient via provider factory (DEEPSEEK_API_KEY or ANTHROPIC_API_KEY)
3. Create AsyncioRuntime
4. Create ToolBridge from mappings.toml
5. Create SkillRegistry, discover + resolve skills
6. Create SkillPipeline
7. Register ExecutorAgent and ReviewerAgent factories with runtime
8. Create OrchestratorAgent (injects runtime, registry, llm_client, pipeline)
9. Orchestrator.start(session_goal=...)
10. CLI read-eval-print loop:
    - Send SESSION_START once to initiate the session
    - Read human input (stdin)
    - Send USER_MESSAGE with user text
    - Wait for response (Responder reply or task completion)
    - Print output to stdout
11. On "exit" or Ctrl+C → Orchestrator.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from llend.llm.client import create_llm_client
from llend.orchestrator.agent import OrchestratorAgent
from llend.orchestrator.config import OrchestratorConfig
from llend.registry.pipeline import SkillPipeline
from llend.registry.registry import SkillRegistry
from llend.runtime.asyncio_runtime import AsyncioRuntime
from llend.runtime.lifecycle import AgentType
from llend.runtime.message import Message, MsgType
from llend.tool_bridge.bridge import ToolBridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Load ``.env`` file from project root or cwd, if present.

    Looks for ``python-dotenv``; silently skips if not installed.
    Search order: ``llend/.env``, ``.env`` (cwd).
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
    except ImportError:
        return

    candidates = [
        Path(__file__).parent / ".env",
        Path(".env"),
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            logger.info("Loaded .env from %s", p)
            return


# ---------------------------------------------------------------------------
# main  §6.2
# ---------------------------------------------------------------------------


async def main() -> None:
    """Bootstrap the harness and run the CLI REPL.  Spec 005 §6.2."""
    # 0. Load .env file if present (project root or cwd)
    _load_dotenv()

    # 1. Load settings
    config = OrchestratorConfig.from_toml()

    # 2. Resolve provider and create LLM client
    provider = os.environ.get("LLEND_PROVIDER", "deepseek")
    try:
        llm_client = create_llm_client(
            provider,
            model=os.environ.get(
                "LLEND_MODEL",
                "deepseek-chat" if provider == "deepseek" else config.classification_model,
            ),
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            f"Set {provider.upper()}_API_KEY environment variable or configure another provider.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Create runtime  §6.1 step 3
    runtime = AsyncioRuntime()

    # 4. Create ToolBridge  §6.1 step 4
    mappings_path = Path("llend/tool_bridge/mappings.toml")
    if not mappings_path.exists():
        # Try relative to package
        pkg_mappings = Path(__file__).parent / "tool_bridge" / "mappings.toml"
        if pkg_mappings.exists():
            mappings_path = pkg_mappings

    tool_bridge = ToolBridge(mappings_path)

    # 5. Create SkillRegistry, discover + resolve  §6.1 step 5
    skills_dir = Path("llend/skills")
    if not skills_dir.exists():
        skills_dir = Path(__file__).parent / "skills"

    registry = SkillRegistry(skills_dir, tool_bridge)
    await registry.discover()
    registry.resolve_all()

    # 6. Create SkillPipeline  §6.1 step 6
    pipeline = SkillPipeline(registry)

    # 7. Register agent factories with runtime  §6.1 step 7, Spec 005 §5.3
    def _make_executor_factory():
        from llend.executor.agent import ExecutorAgent

        async def factory(handle, context):
            executor = ExecutorAgent(
                runtime=runtime,
                instance_id=handle.instance_id,
                session_id=runtime.session_id,
                llm_client=llm_client,
            )
            await executor.start()

        return factory

    def _make_reviewer_factory():
        from llend.reviewer.agent import ReviewerAgent

        async def factory(handle, context):
            reviewer = ReviewerAgent(
                runtime=runtime,
                instance_id=handle.instance_id,
                session_id=runtime.session_id,
                llm_client=llm_client,
            )
            await reviewer.start()

        return factory

    def _make_responder_factory():
        from llend.responder.agent import ResponderAgent
        from llend.responder.persona import Persona

        async def factory(handle, context):
            persona_raw = context.get("persona", "auto")
            try:
                persona = Persona(persona_raw)
            except ValueError:
                persona = Persona.AUTO
            responder = ResponderAgent(
                runtime=runtime,
                instance_id=handle.instance_id,
                session_id=runtime.session_id,
                llm_client=llm_client,
                persona=persona,
            )
            await responder.start()

        return factory

    runtime.register_agent_type(AgentType.EXECUTOR.value, _make_executor_factory())
    runtime.register_agent_type(AgentType.REVIEWER.value, _make_reviewer_factory())
    runtime.register_agent_type(AgentType.RESPONDER.value, _make_responder_factory())

    # 8. Create OrchestratorAgent  §6.1 step 8
    orch = OrchestratorAgent(
        runtime=runtime,
        registry=registry,
        llm_client=llm_client,
        config=config,
        pipeline=pipeline,
        on_progress=lambda ev: print(f"  {ev.message}"),
    )
    await orch.start(session_goal="Interactive session")

    # 9. Orchestrator.start() already sends session.start internally (§11.1).
    #    We do NOT send a second one — that would be double-delivery.

    # 10. CLI REPL  §6.1 step 10
    print("llend harness ready. Type your request or 'exit'.")
    try:
        while True:
            try:
                user_input = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            # Each input uses user.message (Spec 001 §2.2 fix)
            msg = Message(
                session_id=runtime.session_id,
                sender="human",
                sender_instance="cli",
                recipient=AgentType.ORCHESTRATOR.value,
                msg_type=MsgType.USER_MESSAGE,
                payload={"text": user_input},
            )
            await runtime.send(msg)

            # Wait for response before showing next prompt.
            # (The answer is already printed by the progress callback.)
            await orch.wait_for_response(timeout=120.0)

    finally:
        # 11. Shutdown  §6.1 step 11
        print("Shutting down...")
        await orch.shutdown()
        await runtime.shutdown()


# ---------------------------------------------------------------------------
# python -m llend
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Set default log level to WARNING so the CLI output is clean.
    # Progress events are printed by the callback, not the logger.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
