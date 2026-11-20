from __future__ import annotations

from app.graph.schemas import PolicyDecision
from app.graph.state import AgentState

CLARIFY_MESSAGES = {
    "LOW_CONFIDENCE": "Could you rephrase that? I want to make sure I understand your request correctly.",
    "VERIFICATION_REQUIRED": "Can you share your order number and the email address associated with your account?",
    "IDENTITY_MISMATCH": "That order doesn't match the email on file. Can you double-check the details?",
}


def clarify_node(state: AgentState) -> dict:
    decision = PolicyDecision(**state["policy_decision"])
    message = CLARIFY_MESSAGES.get(
        decision.reason_code,
        "Could you provide more details so I can assist you?",
    )
    return {"response": message}
