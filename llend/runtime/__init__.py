"""Runtime core: message protocol, agent lifecycle, checkpoint system, and
LangGraph-based agent execution backend.  Spec 001.

File layout matches Spec 001 §5:
- ``base.py``              → AgentRuntime ABC (§3.2)
- ``langgraph_runtime.py`` → v0 primary backend (§3.1)
- ``asyncio_runtime.py``   → Lightweight alternative (§3.1)
- ``message.py``           → Message envelope + enums (§2.1, §2.2, §2.2.1)
- ``lifecycle.py``         → Agent states + transitions (§3.3)
- ``checkpoint.py``        → Checkpoint model + disk persistence (§3.4)
- ``notifications.py``     → Human notification channels (§3.4 step 3)
"""

from llend.runtime.asyncio_runtime import AsyncioRuntime
from llend.runtime.base import AgentRuntime
from llend.runtime.checkpoint import (
    Checkpoint,
    InterruptTimeoutError,
    get_default_base_dir,
    set_default_base_dir,
)
from llend.runtime.langgraph_runtime import LangGraphRuntime
from llend.runtime.lifecycle import AgentInstance, AgentState, AgentType
from llend.runtime.message import (
    AgentErrorCode,
    Artifact,
    Message,
    MsgType,
    ReviewIssue,
    TaskStatus,
    Verdict,
)
from llend.runtime.notifications import (
    ConsoleNotificationChannel,
    MultiChannel,
    NotificationChannel,
)

__all__ = [
    # §3.2 — ABC
    "AgentRuntime",
    # §3.1 — concrete backends
    "AsyncioRuntime",
    "LangGraphRuntime",
    # §2.1 — message envelope
    "Message",
    # §2.2 / §2.2.1 — enums
    "MsgType",
    "TaskStatus",
    "Verdict",
    "AgentErrorCode",
    "ReviewIssue",
    "Artifact",
    # §3.3 — lifecycle
    "AgentState",
    "AgentType",
    "AgentInstance",
    # §3.4 — checkpoint
    "Checkpoint",
    "InterruptTimeoutError",
    "get_default_base_dir",
    "set_default_base_dir",
    # §3.4 step 3 — notifications
    "NotificationChannel",
    "ConsoleNotificationChannel",
    "MultiChannel",
]
