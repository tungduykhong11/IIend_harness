"""Responder agent for conversational Q&A.  Spec 003.

Re-exports all public types from the responder subpackage.
"""

from llend.responder.agent import ResponderAgent
from llend.responder.context import (
    ConversationTurn,
    SessionContext,
    TaskResultSummary,
)
from llend.responder.memory import UserProfile
from llend.responder.persona import Persona
from llend.responder.stream import (
    make_error_reply,
    make_final_reply,
    make_reply_chunk,
    reassemble_chunks,
)

# Resolve forward reference: SessionContext.user_profile references UserProfile
# which lives in memory.py.  §5.1, §9.1.
SessionContext.model_rebuild()

__all__ = [
    "ResponderAgent",
    "SessionContext",
    "ConversationTurn",
    "TaskResultSummary",
    "UserProfile",
    "Persona",
    "make_reply_chunk",
    "make_final_reply",
    "make_error_reply",
    "reassemble_chunks",
]
