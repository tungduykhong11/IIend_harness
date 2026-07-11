"""OrchestratorAgent — the central "brain" of the llend harness.

The Orchestrator is the only long-lived agent that spawns other agents.
It receives human messages, classifies them, routes to Executor pipelines
or the Responder, adjudicates Reviewer verdicts, relays interrupts, gates
tool requests, and manages the session from start to completion.

Spec references
===============
- **§2.1** → Role & mindset table (Orchestrator vs Executor/Reviewer/Responder)
- **§2.2** → Lifecycle diagram (INIT → RUNNING → … → DEAD)
- **§2.3** → Orchestrator vs other agents — only agent that spawns others
- **§3** → Message classification and routing table (§3.4)
- **§4** → Task execution loop (Executor → Reviewer → adjudicate)
- **§5** → Execution plan consumption, skill name extraction (§5.2)
- **§8** → Interrupt propagation — relay to human or auto-resolve (§8.1–§8.2)
- **§9** → Responder tool approval gate (§9.1)
- **§11** → Session lifecycle — start (§11.1), during (§11.2), complete (§11.3)
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from llend.llm.client import LLMClient
from llend.orchestrator.adjudicator import AdjudicationAction, AdjudicationResult, adjudicate
from llend.orchestrator.classifier import (
    ClassificationResult,
    MessageCategory,
    classify_message,
)
from llend.orchestrator.config import OrchestratorConfig
from llend.orchestrator.gate import GateDecision, ToolApprovalGate
from llend.orchestrator.progress import ProgressEvent, ProgressReporter, format_plan_progress
from llend.orchestrator.recovery import (
    get_recovery_action,
    handle_executor_crash,
    handle_reviewer_crash,
    should_skip_downstream,
    with_llm_retry,
)
from llend.orchestrator.session import SessionManager, SessionState
from llend.orchestrator.summarizer import (
    summarize_task_result,
    synthesize_session,
)
from llend.orchestrator.wiring import (
    coerce_to_expected_type,
    wire_upstream_output,
)
from llend.registry.models import Skill
from llend.registry.pipeline import ExecutionPlan, SkillPipeline, TaskSpec
from llend.responder.context import TaskResultSummary
from llend.responder.memory import UserProfile
from llend.responder.persona import Persona
from llend.runtime.base import AgentRuntime
from llend.runtime.lifecycle import AgentState, AgentType
from llend.runtime.message import (
    AgentErrorCode,
    Artifact,
    Message,
    MsgType,
    ReviewIssue,
    TaskStatus,
    Verdict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill extraction prompt  —  §5.2
# ---------------------------------------------------------------------------

SKILL_EXTRACTION_PROMPT = """You are a task parser. Given a user request and a list of available skills,
extract the target skill and its parameters.

**IMPORTANT:** The params you extract are forwarded to ALL skills in the
dependency chain, not just the target skill.  Include ALL information the
user provided — especially the search query (what product to analyze),
platform (ebay/amazon/...), and any other details.  Use common-sense
parameter names like "query", "platform", "target_item", "max_items".

Available skills:
{skill_listings}

User request: "{message}"

Respond with JSON:
{{
  "skill_name": "...",
  "params": {{...}},
  "confidence": 0.0-1.0
}}"""


# ---------------------------------------------------------------------------
# Reviewer adversarial prompt  —  §4.5 (verbatim from spec)
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM_PROMPT = """You are a REVIEWER. Your job is to find flaws in the Executor's output.

Task: {task_spec}
Expected output schema: {output_schema}
Enforcement level: {enforcement}

Executor's output: {executor_output}
Executor's own concerns: {concerns}

Verify:
1. Does the output match the expected schema? (if schema exists)
2. Are the numbers/data consistent and reasonable?
3. Are there logical errors or unsupported claims?
4. Did the Executor address ALL requirements in the task spec?
5. [If concerns from Executor]: Verify each concern — confirm or dismiss.

For each issue found, provide:
- severity: "critical" | "important" | "minor"
- field: which part of the output
- message: what's wrong

Return your verdict:
- "pass": no issues found
- "pass_with_warnings": minor issues only, output is still usable
- "fail": critical or important issues make the output unreliable"""


# ---------------------------------------------------------------------------
# OrchestratorAgent  —  §2, §5, §8
# ---------------------------------------------------------------------------


class OrchestratorAgent:
    """Central coordinating agent.  §2.1.

    The Orchestrator is the "brain" — the only long-lived agent that:
    - Receives the user's request
    - Decides whether it's a task or a conversational question
    - Builds an execution plan via SkillPipeline
    - Spawns Executor → Reviewer loops per task
    - Adjudicates verdicts (pass → next, fail → re-do)
    - Summarizes results for the Responder
    - Relays interrupts to the human
    - Gates Responder's tool requests
    - Manages the session from start to completion
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        registry: Any,  # SkillRegistry — avoid circular import
        llm_client: LLMClient,
        *,
        config: OrchestratorConfig | None = None,
        pipeline: SkillPipeline | None = None,
        profile_path: Path | None = None,
        on_progress: "Callable[[ProgressEvent], Awaitable[None]] | None" = None,
    ) -> None:
        import asyncio

        self._runtime = runtime
        self._registry = registry
        self._llm = llm_client
        self._config = config or OrchestratorConfig()
        self._pipeline = pipeline or SkillPipeline(registry)
        self._profile_path = profile_path

        # Session management  §11
        self._session_mgr = SessionManager(
            output_dir=self._config.output_dir,
            profile_path=profile_path,
        )
        self._user_profile: UserProfile | None = None

        # Responder state
        self._responder_id: str | None = None

        # Tool gate  §9
        self._tool_gate = ToolApprovalGate(
            auto_approve_timeout_ms=self._config.tool_auto_approve_timeout_ms,
            max_requests_per_turn=self._config.max_tool_requests_per_turn,
        )

        # Progress reporting  §12
        self._progress = ProgressReporter(on_event=on_progress)

        # Interrupt machinery  §8
        self._pending_interrupts: dict[str, asyncio.Future[str]] = {}

        # Internal state
        self._instance_id: str = ""
        self._session_id: UUID | None = None
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        self._processing_task: asyncio.Task[None] | None = None
        self._closed = False

        # Active task tracking for error recovery
        self._active_executors: dict[str, str] = {}  # task_id → instance_id
        self._active_reviewers: dict[str, str] = {}  # task_id → instance_id
        self._retry_counts: dict[str, int] = {}  # task_id → retry count

        # Response feedback to CLI  (CLI response loop fix)
        self._response_ready: asyncio.Event | None = None
        self._last_response: str = ""

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def session_state(self) -> SessionState:
        return self._session_mgr.state

    # ------------------------------------------------------------------
    # response feedback  (CLI response loop fix)
    # ------------------------------------------------------------------

    async def wait_for_response(self, timeout: float = 300.0) -> str:
        """Block until the next response is ready. Returns the response text.

        Called by the CLI REPL after sending a ``user.message``.  The event
        is signalled by ``_process_message`` when a ``RESPOND_REPLY`` arrives
        or when a task plan completes.
        """
        self._response_ready = asyncio.Event()
        try:
            await asyncio.wait_for(self._response_ready.wait(), timeout=timeout)
            return self._last_response
        except asyncio.TimeoutError:
            return ""
        finally:
            self._response_ready = None
        return self._session_mgr.state

    # ------------------------------------------------------------------
    # lifecycle  §2.2, §11.1
    # ------------------------------------------------------------------

    async def start(self, session_goal: str = "") -> str:
        """Spawn the Orchestrator and begin the session.  §11.1.

        1. Spawn the Orchestrator via the runtime (gets instance_id)
        2. Load UserProfile
        3. Spawn Responder (if enabled)
        4. Register message handler
        5. Begin main processing loop
        """
        # Spawn self through runtime
        self._instance_id = await self._runtime.spawn(
            AgentType.ORCHESTRATOR.value,
            context={"session_goal": session_goal},
        )
        self._session_id = getattr(self._runtime, "session_id", uuid4())

        # §11.1 step 1: Load UserProfile
        self._user_profile = self._session_mgr.start()
        self._session_mgr.state.session_goal = session_goal

        # §11.1 step 2: Spawn Responder
        if self._config.responder_enabled:
            await self._spawn_responder()

        # §11.1 step 3: session.start removed — the runtime is not a routable
        # agent and session goal is already recorded by _session_mgr.start().
        # See: _resolve_recipient() in asyncio_runtime.py drops messages to "runtime".

        # §11.1 step 4: Register handler
        await self._runtime.register_handler(self._instance_id, self._message_handler)

        # §11.1 step 5: Begin main loop
        self._processing_task = asyncio.create_task(
            self._main_loop(),
            name=f"orchestrator-loop-{self._instance_id}",
        )

        logger.info(
            "OrchestratorAgent started instance_id=%s session=%s goal=%r",
            self._instance_id,
            self._session_id,
            session_goal,
        )
        return self._instance_id

    async def shutdown(self) -> None:
        """Complete the session and clean up.  §11.3."""
        if self._closed:
            return
        self._closed = True

        # §11.3: Complete sequence
        try:
            # Cancel any running tasks (with grace period)
            for task_id, instance_id in list(self._active_executors.items()):
                await self._runtime.kill(instance_id)
                self._active_executors.pop(task_id, None)

            for task_id, instance_id in list(self._active_reviewers.items()):
                await self._runtime.kill(instance_id)
                self._active_reviewers.pop(task_id, None)

            # Generate final synthesis
            synthesis = await synthesize_session(
                session_goal=self._session_mgr.state.session_goal,
                completed_tasks=self._session_mgr.state.completed_tasks,
                conversation_history=self._session_mgr.state.conversation_history,
                llm_client=self._llm,
                skipped_tasks=self._session_mgr.state.skipped_tasks,
                artifacts=self._session_mgr.state.artifact_paths,
            )

            # Save artifacts, update profile
            self._session_mgr.complete(synthesis, updated_profile=self._user_profile)

            # Kill Responder
            if self._responder_id is not None:
                await self._runtime.kill(self._responder_id)
                self._responder_id = None

            # Send session.complete to Runtime
            await self._send_message(
                recipient="runtime",
                msg_type=MsgType.SESSION_COMPLETE,
                payload={"synthesis": synthesis},
            )
        except Exception:
            logger.exception("Error during shutdown sequence")

        # Cancel main loop
        if self._processing_task is not None:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

        logger.info("OrchestratorAgent shutdown instance_id=%s", self._instance_id)

    # ------------------------------------------------------------------
    # message handler  (called by runtime via register_handler)
    # ------------------------------------------------------------------

    async def _message_handler(self, message: Message) -> None:
        """Fire-and-forget callback.  Pushes into serialized inbox."""
        try:
            self._inbox.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("Orchestrator inbox full — dropping message id=%s", message.id)

    # ------------------------------------------------------------------
    # main loop  §11.2
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        """Wait for messages, classify, route.  §11.2.

        Loop:
          ├── Wait for human message
          ├── Classify (§3)
          ├── Route to Executor pipeline OR Responder
          ├── Handle interrupts if raised
          └── Accumulate results in SessionState
        """
        while not self._closed:
            try:
                msg = await self._inbox.get()
            except asyncio.CancelledError:
                break

            try:
                await self._process_message(msg)
            except Exception:
                logger.exception("Error processing message id=%s", msg.id)

    async def _handle_agent_response(self, msg: Message, agent_role: str) -> None:
        """Re-queue TASK_RESULT / TASK_VERDICT for _await_response().

        These messages are consumed by _await_response() inside
        _execute_task_loop(), NOT by _main_loop.  If _main_loop wins the
        inbox race, we must put the message back so _await_response doesn't
        starve and time out.
        """
        self._inbox.put_nowait(msg)

    async def _process_message(self, msg: Message) -> None:
        """Route a single message based on its type and classification.  §3.4."""
        msg_type = msg.msg_type

        if msg_type == MsgType.SESSION_START:
            # Session initiation — record session goal
            goal = msg.payload.get("goal") or msg.payload.get("session_goal") or msg.payload.get("text", "")
            if goal:
                self._session_mgr.state.session_goal = goal
            logger.info("Session started: %s", goal[:80] if goal else "(no goal)")

        elif msg_type == MsgType.USER_MESSAGE:
            # Human message during session — classify and route  §3
            await self._handle_human_message(msg)

        elif msg_type == MsgType.TASK_RESULT:
            # Executor finished — consumed by _execute_task_loop → _await_response
            await self._handle_agent_response(msg, "executor")

        elif msg_type == MsgType.TASK_VERDICT:
            # Reviewer finished — consumed by _execute_task_loop → _await_response
            await self._handle_agent_response(msg, "reviewer")

        elif msg_type == MsgType.INTERRUPT_RAISE:
            await self._handle_interrupt(msg)

        elif msg_type == MsgType.RESPOND_REQUEST_TOOL:
            await self._handle_tool_request(msg)

        elif msg_type == MsgType.AGENT_ERROR:
            await self._handle_agent_error(msg)

        elif msg_type == MsgType.RESPOND_REPLY:
            # Responder's answer — forward to progress channel + signal waiting CLI.
            # Handles both streaming chunks (§8.1) and non-streaming replies (§8.2).
            done = msg.payload.get("done", False)
            is_final = done or "final_answer" in msg.payload

            if is_final:
                answer = msg.payload.get("final_answer", msg.payload.get("answer", ""))
                if answer:
                    await self._progress.emit(ProgressEvent(level="info", message=answer))
                self._session_mgr.state.add_conversation_turn("responder", answer)
                self._last_response = answer
                if self._response_ready is not None:
                    self._response_ready.set()
            # Streaming chunks are accumulated silently — only the final
            # answer is emitted to the progress channel.

        elif msg_type == MsgType.AGENT_HEARTBEAT:
            logger.debug("heartbeat from %s", msg.sender_instance)

        else:
            logger.debug("orchestrator ignoring msg_type=%s", msg_type.value)

    # ------------------------------------------------------------------
    # §3 — Human message handling
    # ------------------------------------------------------------------

    async def _handle_human_message(self, msg: Message) -> None:
        """Classify a human message and route accordingly.  §3.2, §3.4."""
        text = msg.payload.get("text", msg.payload.get("message", ""))
        if not text:
            logger.warning("Empty human message — ignoring")
            return

        # §3.2: Classify
        classification = await classify_message(text, self._llm)
        logger.info(
            "Classified: %s → %s (confidence=%.2f)",
            text[:80],
            classification.category.value,
            classification.confidence,
        )

        # §3.4: Route (see classifier.ROUTING_TABLE for category → MsgType mapping)
        if classification.category == MessageCategory.TASK:
            await self._handle_task(msg, text)
        elif classification.category == MessageCategory.CONVERSATIONAL:
            await self._handle_conversational(msg, text)
        elif classification.category == MessageCategory.SESSION_END:
            await self._handle_session_end(msg)
        elif classification.category == MessageCategory.CONTROL:
            await self._handle_control(msg, text)

    # ------------------------------------------------------------------
    # §5 — Task handling
    # ------------------------------------------------------------------

    async def _handle_task(self, msg: Message, text: str) -> None:
        """Extract skill, build plan, execute.  §5.1, §5.2."""
        # §5.2: Extract skill name + params
        skill_name, params = await self._extract_skill_name(text)

        if skill_name is None:
            # No matching skill — tell user what's available
            available = self._registry.list_skills()
            skill_list = ", ".join(
                name for names in available.values() for name in [n.name for n in names]
            )
            await self._send_message(
                recipient="orchestrator",  # Will be shown to human
                msg_type=MsgType.RESPOND_QUERY,
                payload={
                    "question": text,
                    "error": f"No matching skill. Available: {skill_list or 'none'}",
                },
                parent_id=msg.id,
            )
            return

        # §5.1: Build plan
        try:
            plan = self._pipeline.build_plan(skill_name, params)
        except Exception as exc:
            logger.exception("Failed to build plan for %s", skill_name)
            await self._progress.error(f"Failed to build plan: {exc}")
            return

        self._session_mgr.set_plan(plan)
        await self._progress.plan_start(plan)

        # Reset tool gate for this turn
        self._tool_gate.reset_turn()

        # §5.3 / §5.4: Execute tasks (sequential by default, parallel if enabled)
        if self._config.allow_parallel:
            await self._execute_plan_parallel(plan)
        else:
            await self._execute_plan_sequential(plan)

        # Signal waiting CLI that a response is ready
        completed = self._session_mgr.state.completed_tasks
        if completed:
            self._last_response = completed[-1].summary
        if self._response_ready is not None:
            self._response_ready.set()

    async def _execute_plan_sequential(self, plan: ExecutionPlan) -> None:
        """Execute tasks one-by-one in order.  §5.3."""
        upstream_outputs: dict[str, Any] = {}
        completed: set[str] = set()

        for task_spec in plan.skills:
            # Wire upstream outputs into task params  §6
            if task_spec.input_from:
                for upstream_name in task_spec.input_from:
                    if upstream_name in upstream_outputs:
                        wired = wire_upstream_output(
                            upstream_outputs[upstream_name],
                            upstream_name,
                            task_spec,
                        )
                        task_spec.task_spec.update(wired)
                        # Type coercion for downstream consumption  §6.4
                        for key, value in wired.items():
                            if isinstance(value, list):
                                task_spec.task_spec[key] = coerce_to_expected_type(
                                    value, "list[dict]"
                                )

            task_id = uuid4()
            result = await self._execute_task_loop(task_spec, task_id, len(plan.skills))

            if result is not None:
                upstream_outputs[task_spec.skill_name] = result
                completed.add(task_spec.skill_name)
            else:
                # Task failed
                self._session_mgr.state.add_warning(
                    f"Task {task_spec.skill_name} could not be completed."
                )
                # Graceful degradation  §10.3 — skip downstream dependents
                affected = should_skip_downstream(task_spec.skill_name, plan)
                for downstream_name in affected:
                    self._session_mgr.state.add_warning(
                        f"Task {downstream_name} skipped — depends on failed task {task_spec.skill_name}."
                    )
                if task_spec.skill_name in self._session_mgr.state.skipped_tasks:
                    continue
                # Mandatory failure → abort
                skill = self._registry.get(task_spec.skill_name)
                if skill is not None and skill.enforcement == "mandatory":
                    await self._progress.error(
                        f"Mandatory task {task_spec.skill_name} failed — aborting session."
                    )
                    break

        # §5.3: Report if no tasks completed successfully
        if not completed:
            await self._progress.error(
                "Không thể hoàn thành yêu cầu. Có thể do: "
                "trang web chặn bot, cấu trúc HTML không được hỗ trợ, "
                "hoặc không tìm thấy dữ liệu sản phẩm phù hợp."
            )

    async def _execute_plan_parallel(self, plan: ExecutionPlan) -> None:
        """Execute tasks with parallel batches where possible.  §5.4.

        Gated by ``config.allow_parallel`` — only called when True.
        Collects consecutive parallelizable tasks into batches and runs
        each batch via ``asyncio.gather``.
        """
        results: list[Any] = []
        i = 0
        while i < len(plan.skills):
            # Collect parallelizable batch  §5.4
            batch = [plan.skills[i]]
            while i + 1 < len(plan.skills) and plan.skills[i + 1].parallelizable:
                batch.append(plan.skills[i + 1])
                i += 1
            i += 1

            # Execute batch concurrently  §5.4
            batch_tasks = [
                self._execute_task_loop(ts, uuid4(), len(plan.skills))
                for ts in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            for ts, result in zip(batch, batch_results):
                if result is not None:
                    await self._progress.task_complete(
                        ts, uuid4(), len(plan.skills),
                        f"Completed {ts.skill_name}.",
                    )
                else:
                    self._session_mgr.state.add_warning(
                        f"Task {ts.skill_name} could not be completed."
                    )
                    affected = should_skip_downstream(ts.skill_name, plan)
                    for downstream_name in affected:
                        self._session_mgr.state.add_warning(
                            f"Task {downstream_name} skipped — depends on "
                            f"failed task {ts.skill_name}."
                        )
            results.extend(batch_results)

    # ------------------------------------------------------------------
    # §4 — Task execution loop (Executor → Reviewer → adjudicate)
    # ------------------------------------------------------------------

    async def _execute_task_loop(
        self, task_spec: TaskSpec, task_id: UUID, total: int
    ) -> Any | None:
        """Run the Executor → Reviewer loop for one task.  §4.1, §4.2."""
        # Use .get() (cached resolved skill) rather than .resolve() to avoid
        # redundant validation — skills are pre-resolved during plan building.  §4.2
        skill = self._registry.get(task_spec.skill_name)
        if skill is None:
            await self._progress.error(f"Skill {task_spec.skill_name!r} not found in registry")
            return None

        max_retries = self._config.get_max_retries(skill.enforcement)
        retry_count = 0
        output = None

        while retry_count <= max_retries:
            # --- Step 1-3: Spawn Executor, dispatch, wait for result  §4.2 ---
            await self._progress.task_start(task_spec, task_id, total)

            executor_id = await self._runtime.spawn(
                AgentType.EXECUTOR.value,
                context={
                    "skill_name": task_spec.skill_name,
                    "task_spec": task_spec.task_spec,
                    "skill_context": {
                        "skill_md": skill.skill_md,
                        "action_bindings": {
                            name: binding.model_dump()
                            for name, binding in skill.action_bindings.items()
                        },
                        "output_schema": skill.output_schema,
                        "enforcement": skill.enforcement,
                        "handler": skill.handler,  # custom handler instance (in-process)
                    },
                },
            )
            self._active_executors[str(task_id)] = executor_id

            # Send task.dispatch  §4.2 step 3 — includes skill_context §2.2 (Spec 005 fix)
            dispatch_msg = Message(
                session_id=self._session_id or uuid4(),
                sender=AgentType.ORCHESTRATOR.value,
                sender_instance=self._instance_id,
                recipient=AgentType.EXECUTOR.value,
                recipient_instance=executor_id,
                msg_type=MsgType.TASK_DISPATCH,
                payload={
                    "task_id": str(task_id),
                    "skill_name": task_spec.skill_name,
                    "task_spec": task_spec.task_spec,
                    "skill_context": {
                        "skill_md": skill.skill_md,
                        "allowed_actions": list(skill.action_bindings.keys()),
                        "action_bindings": {
                            name: binding.model_dump()
                            for name, binding in skill.action_bindings.items()
                        },
                        "output_schema": skill.output_schema,
                        "enforcement": skill.enforcement,
                        "handler": skill.handler,  # custom handler instance (in-process)
                    },
                },
            )
            await self._runtime.send(dispatch_msg)

            # §4.2 step 4: Wait for task.result (with timeout)
            result_msg = await self._await_response(
                executor_id, MsgType.TASK_RESULT, timeout=self._config.task_timeout_default
            )
            self._active_executors.pop(str(task_id), None)

            if result_msg is None:
                # Timeout → kill Executor, retry
                await self._runtime.kill(executor_id)
                retry_count += 1
                if retry_count > max_retries:
                    break
                continue

            output = result_msg.payload.get("output", {})
            concerns = result_msg.payload.get("concerns")

            # --- Step 5: Validate output schema  §4.2 ---
            schema_issues: list[str] = []
            if skill.output_schema is not None and output:
                schema_issues = self._validate_output(output, skill)

            if schema_issues and skill.enforcement == "mandatory":
                retry_count += 1
                if retry_count <= max_retries:
                    # Retry with schema feedback
                    task_spec.task_spec["schema_errors"] = schema_issues
                    continue

            # --- Step 6-8: Reviewer cycle  §4.2 ---
            reviewer_id = await self._runtime.spawn(
                AgentType.REVIEWER.value,
                context={"task_spec": task_spec.task_spec},
            )
            self._active_reviewers[str(task_id)] = reviewer_id

            # Build adversarially-framed review prompt  §4.5
            import json as _json
            reviewer_prompt = REVIEWER_SYSTEM_PROMPT.format(
                task_spec=_json.dumps(task_spec.task_spec, ensure_ascii=False),
                output_schema=_json.dumps(skill.output_schema) if skill.output_schema else "(none)",
                enforcement=skill.enforcement,
                executor_output=_json.dumps(output, ensure_ascii=False, default=str)[:4000],
                concerns=_json.dumps(concerns) if concerns else "(none)",
            )
            review_msg = Message(
                session_id=self._session_id or uuid4(),
                sender=AgentType.ORCHESTRATOR.value,
                sender_instance=self._instance_id,
                recipient=AgentType.REVIEWER.value,
                recipient_instance=reviewer_id,
                msg_type=MsgType.TASK_REVIEW,
                payload={
                    "task_id": str(task_id),
                    "task_spec": task_spec.task_spec,
                    "executor_output": output,
                    "system_prompt": reviewer_prompt,
                    "concerns": concerns,
                    "schema_validation_issues": schema_issues,
                },
            )
            await self._runtime.send(review_msg)

            # Wait for task.verdict
            verdict_msg = await self._await_response(
                reviewer_id, MsgType.TASK_VERDICT, timeout=self._config.review_timeout_default
            )
            self._active_reviewers.pop(str(task_id), None)

            if verdict_msg is None:
                # Reviewer timeout → auto-pass with warning
                await self._runtime.kill(reviewer_id)
                verdict = Verdict.PASS_WITH_WARNINGS
                issues: list[ReviewIssue] = []
                self._session_mgr.state.add_warning(
                    f"Reviewer timed out for {task_spec.skill_name} — auto-passing."
                )
            else:
                verdict = Verdict(verdict_msg.payload.get("verdict", "fail"))
                raw_issues = verdict_msg.payload.get("issues", [])
                issues = [
                    ReviewIssue(**i) if isinstance(i, dict) else i for i in raw_issues
                ]

            # §4.3: Adjudicate
            result = adjudicate(
                verdict,
                retry_count,
                skill.enforcement,
                reviewer_issues=issues,
                max_retries_override={
                    "mandatory": self._config.max_retries_mandatory,
                    "strict": self._config.max_retries_strict,
                    "suggested": self._config.max_retries_suggested,
                },
            )

            if result.action == AdjudicationAction.NEXT_TASK:
                # Summarize and complete
                if output is not None:
                    summary = await summarize_task_result(
                        task_result_payload={"task_id": str(task_id), "output": output},
                        skill=skill,
                        task_spec=task_spec,
                        llm_client=self._llm,
                    )
                    self._session_mgr.state.add_task_result(summary)
                    await self._progress.task_complete(
                        task_spec, task_id, total, summary.summary,
                    )
                for warning in result.warnings:
                    self._session_mgr.state.add_warning(warning)
                    await self._progress.task_warning(task_spec, warning)
                return output

            elif result.action == AdjudicationAction.RETRY:
                retry_count += 1
                if result.improved_task_spec:
                    task_spec.task_spec.update(result.improved_task_spec)
                await self._progress.task_warning(
                    task_spec, f"Retrying ({retry_count}/{max_retries})..."
                )
                continue

            elif result.action == AdjudicationAction.SKIP_TASK:
                self._session_mgr.state.add_warning(
                    f"Task {task_spec.skill_name} skipped after {retry_count} retries."
                )
                await self._progress.task_warning(task_spec, "Skipping task.")
                return None

            elif result.action == AdjudicationAction.ABORT_SESSION:
                await self._progress.error(
                    f"Critical task {task_spec.skill_name} failed after {retry_count} attempts."
                )
                return None

            # Kill reviewer after use
            await self._runtime.kill(reviewer_id)

        # Exhausted all retries
        return None

    # ------------------------------------------------------------------
    # §3.4 — Conversational handling
    # ------------------------------------------------------------------

    async def _handle_conversational(self, msg: Message, text: str) -> None:
        """Route a conversational message to the Responder.  §3.4."""
        if self._responder_id is None:
            # No Responder — handle inline
            logger.warning("No Responder spawned — cannot handle conversational message")
            return

        # Build session context for Responder  §5.1 (Spec 003)
        session_context = {
            "session_goal": self._session_mgr.state.session_goal,
            "task_results": [
                ts.model_dump() for ts in self._session_mgr.state.completed_tasks[-5:]
            ],
            "active_task": (
                self._session_mgr.state.active_task.model_dump()
                if self._session_mgr.state.active_task
                else None
            ),
        }

        query_msg = Message(
            session_id=self._session_id or uuid4(),
            sender=AgentType.ORCHESTRATOR.value,
            sender_instance=self._instance_id,
            recipient=AgentType.RESPONDER.value,
            recipient_instance=self._responder_id,
            msg_type=MsgType.RESPOND_QUERY,
            payload={
                "question": text,
                "session_context": session_context,
            },
            parent_id=msg.id,
        )
        await self._runtime.send(query_msg)

        # Record conversation turn  §7.3
        self._session_mgr.state.add_conversation_turn("user", text)

    # ------------------------------------------------------------------
    # §3.4 — Session end
    # ------------------------------------------------------------------

    async def _handle_session_end(self, msg: Message) -> None:
        """Handle a session-end message.  §3.4."""
        logger.info("Session end requested by human")
        await self.shutdown()

    # ------------------------------------------------------------------
    # §3.4 — Control messages
    # ------------------------------------------------------------------

    async def _handle_control(self, msg: Message, text: str) -> None:
        """Handle a control message (cancel, pause, status).  §3.1."""
        text_lower = text.lower()

        if "cancel" in text_lower or "dừng" in text_lower:
            # Cancel active task
            for task_id, instance_id in list(self._active_executors.items()):
                await self._runtime.kill(instance_id)
                self._active_executors.pop(task_id, None)
            await self._progress.emit(ProgressEvent(
                level="info", message="Active task cancelled.",
            ))

        elif "status" in text_lower or "đang làm gì" in text_lower:
            # §12.3: In-progress status
            if self._session_mgr.state.plan is not None:
                completed_names = {ts.skill_name for ts in self._session_mgr.state.completed_tasks}
                active_name = (
                    self._session_mgr.state.active_task.skill_name
                    if self._session_mgr.state.active_task
                    else None
                )
                status_text = format_plan_progress(
                    self._session_mgr.state.plan, completed_names, active=active_name
                )
                await self._progress.emit(ProgressEvent(
                    level="info", message=status_text,
                ))

        elif "pause" in text_lower or "tạm dừng" in text_lower:
            await self._progress.emit(ProgressEvent(
                level="warning", message="Pause requested but not yet implemented (v1).",
            ))

    # ------------------------------------------------------------------
    # §8 — Interrupt propagation
    # ------------------------------------------------------------------

    async def _handle_interrupt(self, msg: Message) -> None:
        """Relay or auto-resolve an interrupt from an agent.  §8.1, §8.2."""
        payload = msg.payload
        prompt = payload.get("message", payload.get("prompt", ""))
        options: list[str] = payload.get("options", [])

        # §8.3: If human interrupts Responder mid-stream, kill the in-progress
        # generation but keep the Responder instance alive.
        if msg.sender == AgentType.RESPONDER.value and payload.get("streaming"):
            logger.info("Interrupting Responder mid-stream — killing generation, keeping instance alive")
            # Send a cancellation signal to stop the current generation
            cancel_msg = Message(
                session_id=self._session_id or uuid4(),
                sender=AgentType.ORCHESTRATOR.value,
                sender_instance=self._instance_id,
                recipient=AgentType.RESPONDER.value,
                recipient_instance=msg.sender_instance,
                msg_type=MsgType.RESPOND_REPLY,
                payload={
                    "streaming_cancelled": True,
                    "message": "Generation cancelled by user interrupt.",
                },
                parent_id=msg.id,
            )
            await self._runtime.send(cancel_msg)
            return

        # §8.2: Auto-response heuristics
        if len(options) == 1:
            # Only one option → auto-select
            await self._resolve_interrupt(
                msg.sender_instance, options[0], "auto: single option"
            )
            return

        if self._is_trivial_decision(payload):
            await self._resolve_interrupt(
                msg.sender_instance, options[0], "auto: trivial"
            )
            return

        # Relay to human  §8.1
        logger.info("Interrupt relayed to human: %s", prompt[:120])
        # In v0, we log the interrupt and use the runtime's interrupt mechanism
        await self._runtime.interrupt(msg.sender_instance, prompt, options)

    @staticmethod
    def _is_trivial_decision(payload: dict[str, Any]) -> bool:
        """Check whether this interrupt can be auto-resolved.  §8.2."""
        prompt = payload.get("message", payload.get("prompt", ""))
        lower = prompt.lower()
        trivial_patterns = [
            "continue with default",
            "retry after error",
            "proceed with",
        ]
        if any(p in lower for p in trivial_patterns):
            return True

        # Cost/time estimate within budget → auto-approve  §8.2
        cost_estimate = payload.get("cost_estimate")
        budget_limit = payload.get("budget_limit")
        if cost_estimate is not None and budget_limit is not None:
            try:
                if float(cost_estimate) <= float(budget_limit):
                    return True
            except (ValueError, TypeError):
                pass

        return False

    async def _resolve_interrupt(
        self, target_instance: str, decision: str, source: str
    ) -> None:
        """Send an interrupt response back to the agent.  §8.1."""
        logger.info("Auto-resolving interrupt for %s: %s (%s)", target_instance, decision, source)
        response = Message(
            session_id=self._session_id or uuid4(),
            sender=AgentType.ORCHESTRATOR.value,
            sender_instance=self._instance_id,
            recipient="executor",  # or reviewer/responder — resolved by runtime
            recipient_instance=target_instance,
            msg_type=MsgType.INTERRUPT_RESPONSE,
            payload={"decision": decision, "auto": True},
        )
        await self._runtime.send(response)

    # ------------------------------------------------------------------
    # §9 — Tool request handling
    # ------------------------------------------------------------------

    async def _handle_tool_request(self, msg: Message) -> None:
        """Evaluate and approve/deny a Responder tool request.  §9.1."""
        payload = msg.payload
        suggested_skill = payload.get("suggested_skill", "")
        tool_params = payload.get("tool_params", {})
        request_id = str(msg.id)

        decision = self._tool_gate.evaluate(suggested_skill, tool_params, self._registry)

        if decision.approved:
            if decision.cached_result is not None:
                # Return cached result immediately  §9.1 step 3
                await self._send_tool_result(request_id, decision.cached_result, msg)
            else:
                # Auto-approved — dispatch Executor silently  §9.1
                await self._dispatch_tool_executor(request_id, suggested_skill, tool_params, msg)
        elif decision.needs_human:
            # Escalate to human via interrupt  §9.1
            logger.info("Tool request needs human approval: %s", decision.reason)
            skill = self._registry.get(suggested_skill)
            skill_desc = skill.description if skill else suggested_skill
            # Estimate cost from action binding timeout
            cost_estimate = "unknown"
            if skill is not None:
                binding = skill.action_bindings.get(suggested_skill) if skill.action_bindings else None
                if binding is not None and binding.timeout_ms:
                    cost_estimate = f"~{binding.timeout_ms / 1000:.0f}s processing time"
            prompt = (
                f"Responder wants to {skill_desc}. "
                f"Estimated cost: {cost_estimate}.\n"
                f"[A] Allow  [B] Deny  [C] Allow with limits (specify)"
            )
            await self._runtime.interrupt(
                self._instance_id, prompt, ["A: Allow", "B: Deny", "C: Allow with limits (specify)"]
            )
        else:
            # Denied
            await self._send_tool_result(
                request_id,
                {"error": decision.reason},
                msg,
            )

    async def _dispatch_tool_executor(
        self,
        request_id: str,
        skill_name: str,
        tool_params: dict[str, Any],
        request_msg: Message,
    ) -> None:
        """Dispatch an Executor for a Responder tool request.  §9.1."""
        task_id = uuid4()
        executor_id = await self._runtime.spawn(
            AgentType.EXECUTOR.value,
            context={"skill_name": skill_name, "tool_params": tool_params},
        )

        # Resolve skill to build skill_context for the dispatch  §9.1
        skill = self._registry.get(skill_name)
        skill_context_payload: dict[str, Any] = {}
        if skill is not None:
            skill_context_payload = {
                "skill_md": skill.skill_md,
                "allowed_actions": list(skill.action_bindings.keys()),
                "action_bindings": {
                    name: binding.model_dump()
                    for name, binding in skill.action_bindings.items()
                },
                "output_schema": skill.output_schema,
                "enforcement": skill.enforcement,
            }

        dispatch = Message(
            session_id=self._session_id or uuid4(),
            sender=AgentType.ORCHESTRATOR.value,
            sender_instance=self._instance_id,
            recipient=AgentType.EXECUTOR.value,
            recipient_instance=executor_id,
            msg_type=MsgType.TASK_DISPATCH,
            payload={
                "task_id": str(task_id),
                "skill_name": skill_name,
                "task_spec": tool_params,
                "skill_context": skill_context_payload,
            },
        )
        await self._runtime.send(dispatch)

        # Wait for result
        result_msg = await self._await_response(
            executor_id, MsgType.TASK_RESULT, timeout=60.0
        )

        if result_msg is not None:
            result_data = result_msg.payload.get("output", {})
            # Cache for future requests  §9.1 step 3
            self._tool_gate.cache_result(skill_name, tool_params, result_data)
            await self._send_tool_result(request_id, result_data, request_msg)
        else:
            await self._runtime.kill(executor_id)
            await self._send_tool_result(
                request_id,
                {"error": f"Tool '{skill_name}' timed out."},
                request_msg,
            )

    async def _send_tool_result(
        self, request_id: str, result: dict[str, Any], request_msg: Message
    ) -> None:
        """Send ``respond.tool_result`` back to the Responder.  §6.2 (Spec 003)."""
        tool_result = Message(
            session_id=self._session_id or uuid4(),
            sender=AgentType.ORCHESTRATOR.value,
            sender_instance=self._instance_id,
            recipient=AgentType.RESPONDER.value,
            recipient_instance=self._responder_id,
            msg_type=MsgType.RESPOND_TOOL_RESULT,
            payload={
                "request_id": request_id,
                "result": result,
            },
            parent_id=request_msg.id,
        )
        await self._runtime.send(tool_result)

    # ------------------------------------------------------------------
    # §10 — Error handling
    # ------------------------------------------------------------------

    async def _handle_agent_error(self, msg: Message) -> None:
        """Handle an ``agent.error`` message.  §10.1."""
        error_code_raw = msg.payload.get("error_code", "unknown")
        try:
            error_code = AgentErrorCode(error_code_raw)
        except ValueError:
            error_code = AgentErrorCode.UNKNOWN

        detail = msg.payload.get("detail", "")
        logger.warning(
            "agent.error from %s: code=%s detail=%s",
            msg.sender_instance,
            error_code.value,
            detail,
        )

        # Look up recovery action  §10.1
        recovery = get_recovery_action(error_code)
        logger.info(
            "Recovery for %s: action=%s max_attempts=%d reason=%s",
            error_code.value,
            recovery.action,
            recovery.max_attempts,
            recovery.reason,
        )

        # Forward to progress
        await self._progress.error(f"[{error_code.value}] {detail}")

    # ------------------------------------------------------------------
    # §5.2 — Skill name extraction
    # ------------------------------------------------------------------

    async def _extract_skill_name(self, text: str) -> tuple[str | None, dict[str, Any]]:
        """Extract target skill and parameters from a user request.  §5.2.

        Returns ``(skill_name, params)`` or ``(None, {})`` if no match.
        """
        # Build available skills listing
        available = self._registry.list_skills()
        skill_lines: list[str] = []
        for category, metas in available.items():
            for meta in metas:
                resolved = self._registry.get(meta.name)
                skill_lines.append(
                    f"- {meta.name}: {meta.description} "
                    f"(inputs: {', '.join(meta.inputs) if meta.inputs else 'none'})"
                )

        skill_listings = "\n".join(skill_lines) if skill_lines else "(no skills available)"

        if not skill_lines:
            return None, {}

        prompt = SKILL_EXTRACTION_PROMPT.format(
            skill_listings=skill_listings,
            message=text,
        )

        try:
            import json
            raw = await self._llm.generate(
                messages=[{"role": "user", "content": prompt}],
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            data = json.loads(raw)
            skill_name = data.get("skill_name")
            params = data.get("params", {})
            confidence = data.get("confidence", 0.0)

            # Require minimum confidence to avoid LLM hallucinated skill names.
            # Threshold of 0.3 is a conservative floor — well below typical
            # well-matched extractions (0.7+) but above pure guesses.  §5.2
            if skill_name and confidence > 0.3:
                return skill_name, params
            return None, {}
        except Exception:
            logger.exception("Skill extraction failed")
            return None, {}

    # ------------------------------------------------------------------
    # Responder management
    # ------------------------------------------------------------------

    async def _spawn_responder(self) -> None:
        """Spawn the Responder agent.  §11.1."""
        persona = (
            self._user_profile.persona_preference
            if self._user_profile
            else Persona.AUTO
        )

        # Import here to avoid circular dependency
        from llend.responder.agent import ResponderAgent

        self._responder_id = await self._runtime.spawn(
            AgentType.RESPONDER.value,
            context={"persona": persona.value},
        )
        logger.info("Responder spawned: %s", self._responder_id)

        # The Responder manages its own lifecycle via start()/shutdown().
        # We just track the instance_id for routing.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _await_response(
        self,
        target_instance: str,
        expected_type: MsgType,
        timeout: float = 300.0,
    ) -> Message | None:
        """Wait for a message of *expected_type* from *target_instance*.  §4.2."""
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning(
                    "Timeout waiting for %s from %s", expected_type.value, target_instance
                )
                return None

            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None

            if msg.msg_type == expected_type and msg.sender_instance == target_instance:
                return msg

            # Re-queue messages that don't match (but not if they're critical)
            if msg.msg_type in (MsgType.AGENT_ERROR, MsgType.INTERRUPT_RAISE):
                # Process these immediately
                await self._process_message(msg)
            else:
                self._inbox.put_nowait(msg)

    async def _send_message(
        self,
        recipient: str,
        msg_type: MsgType,
        payload: dict[str, Any],
        parent_id: UUID | None = None,
    ) -> None:
        """Build and send a message through the runtime."""
        msg = Message(
            session_id=self._session_id or uuid4(),
            sender=AgentType.ORCHESTRATOR.value,
            sender_instance=self._instance_id,
            recipient=recipient,
            msg_type=msg_type,
            payload=payload,
            parent_id=parent_id,
        )
        await self._runtime.send(msg)

    def _validate_output(self, output: Any, skill: Skill) -> list[str]:
        """Validate Executor output against the skill's output schema.  §4.2 step 5.

        Spec describes ``SkillOutputModel.model_validate(executor_output)`` with
        three enforcement-dependent retry paths (mandatory: max 3, strict: max 2
        then pass to Reviewer, suggested: pass as-is).  In v0 we do basic
        structural type-checking only; full Pydantic model validation with
        JSON Schema is deferred to v1.
        """
        if skill.output_schema is None:
            return []
        issues: list[str] = []
        if isinstance(output, dict) and "type" in skill.output_schema:
            expected = skill.output_schema["type"]
            if expected == "object" and not isinstance(output, dict):
                issues.append(f"Expected object, got {type(output).__name__}")
        return issues
