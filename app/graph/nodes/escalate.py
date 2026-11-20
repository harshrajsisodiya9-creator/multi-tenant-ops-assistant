from __future__ import annotations

import logging

from app.graph.schemas import PolicyDecision
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def escalate_node(state: AgentState) -> dict:
    decision = PolicyDecision(**state["policy_decision"])
    logger.info("Escalated: %s", decision.reason_code)
    return {
        "response": (
            "I'm connecting you with a human agent who can help with this. "
            "Your request has been flagged for review."
        )
    }
