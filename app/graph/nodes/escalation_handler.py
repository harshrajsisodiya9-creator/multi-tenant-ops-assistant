from __future__ import annotations

from langgraph.types import interrupt

from app.graph.state import AgentState


def escalation_handler_node(state: AgentState) -> dict:
    intent_results: list[dict] = list(state.get("intent_results") or [])

    pending_escalations = []
    for i, result in enumerate(intent_results):
        if result.get("status") == "awaiting_human":
            pending_escalations.append(
                {
                    "index": i,
                    "intent": result["intent"],
                    "reason_code": result.get("reason_code", ""),
                    "domain_result": result.get("domain_result", ""),
                    "entities": result.get("entities", {}),
                }
            )

    if not pending_escalations:
        return {}

    human_decisions = interrupt(pending_escalations)

    decisions = human_decisions if isinstance(human_decisions, list) else human_decisions.get("decisions", [])

    for decision in decisions:
        idx = decision.get("index")
        action = decision.get("action", "deny")

        if idx is None or idx >= len(intent_results):
            continue

        result = intent_results[idx]

        if action == "approve":
            result["status"] = "fulfilled"
            note = decision.get("note", "")
            original = result.get("domain_result", "")
            result["domain_result"] = f"{original} Approved by support team." + (f" Note: {note}" if note else "")
            result["_newly_resolved"] = True

        elif action == "deny":
            result["status"] = "blocked"
            result["reason_code"] = "HUMAN_DENIED"
            result["domain_result"] = decision.get("reason", "This request has been reviewed and cannot be processed.")
            result["_newly_resolved"] = True

        elif action == "override":
            result["status"] = "fulfilled"
            result["domain_result"] = decision.get("custom_response", "")
            result["_newly_resolved"] = True

    return {
        "intent_results": intent_results,
        "response_phase": "follow_up",
    }
