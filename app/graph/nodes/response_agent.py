from __future__ import annotations

import json

from langchain_groq import ChatGroq

from app.config import setting
from app.graph.conversation_store import conversation_store
from app.graph.state import AgentState

RESPONSE_SYSTEM_PROMPT = """\
You are a customer support response composer for an e-commerce platform.

You receive structured intent results and compose ONE natural, helpful response.

Each intent result has:
- intent: what the customer wanted (faq, order_status, refund, cancel_order, complaint)
- status: fulfilled, blocked, failed, needs_clarification, or awaiting_human
- domain_result: the specific information or explanation for this intent

Rules:
- For fulfilled intents: present the information naturally
- For blocked intents: explain WHY it was blocked using the domain_result, be empathetic
- For failed intents: apologise, explain what went wrong using the domain_result, and suggest trying again or contacting support
- For needs_clarification: ask for the specific missing information
- For awaiting_human: acknowledge the request and tell the customer a team member will follow up
- If multiple intents: weave them into one coherent response, don't use bullet points or numbering
- Be concise, professional, and empathetic
- Do NOT invent information beyond what's in the domain_result
- Do NOT mention internal codes, policy names, or system details
- Do NOT repeat or summarize information from previous turns — only present what is new in the current intent results"""

FOLLOW_UP_SYSTEM_PROMPT = """\
You are a customer support response composer. You are writing a FOLLOW-UP message.

The customer already received an initial response. Now a human agent has reviewed their
escalated request(s) and made a decision. Compose a follow-up message delivering ONLY
the newly resolved results.

Rules:
- Do NOT repeat information from the initial response
- Present the resolution naturally and concisely
- Be professional and empathetic
- Do NOT mention internal codes or system details"""


def _format_turn_summaries(turn_summaries: list[dict]) -> str:
    if not turn_summaries:
        return ""
    lines = []
    for s in turn_summaries:
        oid = s.get("order_id", "")
        part = f"{s['intent']}"
        if oid:
            part += f" for {oid}"
        part += f" ({s['status']}"
        reason = s.get("reason_code")
        if reason:
            part += f" — {reason}"
        part += ")"
        lines.append(f"- {part}")
    return "Previously resolved (DO NOT repeat):\n" + "\n".join(lines)


def _build_turn_summary(intent_results: list[dict]) -> list[dict]:
    summaries = []
    for r in intent_results:
        summaries.append({
            "intent": r.get("intent", "unknown"),
            "order_id": r.get("entities", {}).get("order_id", ""),
            "status": r.get("status", "unknown"),
            "reason_code": r.get("reason_code", ""),
        })
    return summaries


def response_agent_node(state: AgentState) -> dict:
    response_phase = state.get("response_phase", "complete")
    intent_results = state.get("intent_results") or []

    if not intent_results:
        return {
            "response": "I'm sorry, I couldn't process your request. Please try again."
        }

    if response_phase == "follow_up":
        resolved = [r for r in intent_results if r.get("_newly_resolved")]
        if not resolved:
            return {}
        system_prompt = FOLLOW_UP_SYSTEM_PROMPT
        results_for_llm = resolved
    else:
        system_prompt = RESPONSE_SYSTEM_PROMPT
        results_for_llm = intent_results

    results_summary = json.dumps(
        [
            {
                "intent": r["intent"],
                "status": r["status"],
                "domain_result": r.get("domain_result", ""),
            }
            for r in results_for_llm
        ],
        indent=2,
    )

    user_message = state.get("user_message", "")
    prior_context = _format_turn_summaries(state.get("turn_summaries") or [])

    user_content = f"Customer message: {user_message}\n\nIntent results:\n{results_summary}"
    if prior_context:
        user_content = f"{prior_context}\n\n{user_content}"

    llm = ChatGroq(
        model=setting.groq_model,
        api_key=setting.groq_api_key,  # type: ignore
        temperature=0.3,
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )

    tenant_id = state.get("tenant_id", "")
    customer_id = state.get("customer_id") or ""
    conversation_id = state.get("conversation_id", "")
    if conversation_id:
        conversation_store.add_ai_response(
            tenant_id,
            customer_id,
            conversation_id,
            response.content,
        )
        conversation_store.add_turn_summary(
            tenant_id,
            customer_id,
            conversation_id,
            _build_turn_summary(results_for_llm),
        )

    return {"response": response.content}
