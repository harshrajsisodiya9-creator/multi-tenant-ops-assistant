from __future__ import annotations

import logging

from app.graph.schemas import PolicyDecision
from app.graph.state import AgentState

logger = logging.getLogger(__name__)


def block_node(state: AgentState) -> dict:
    decision = PolicyDecision(**state["policy_decision"])
    logger.info("Blocked: %s", decision.reason_code)
    return {
        "response": (
            "I'm sorry, I'm not able to help with that request. "
            "Please contact our support team for further assistance."
        )
    }
