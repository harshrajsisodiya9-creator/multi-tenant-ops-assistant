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
- status: fulfilled, blocked, needs_clarification, or awaiting_human
- domain_result: the specific information or explanation for this intent

Rules:
- For fulfilled intents: present the information naturally
- For blocked intents: explain WHY it was blocked using the domain_result, be empathetic
- For needs_clarification: ask for the specific missing information
- For awaiting_human: acknowledge the request and tell the customer a team member will follow up
- If multiple intents: weave them into one coherent response, don't use bullet points or numbering
- Be concise, professional, and empathetic
- Do NOT invent information beyond what's in the domain_result
- Do NOT mention internal codes, policy names, or system details"""

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


def response_agent_node(state: AgentState) -> dict:
    response_phase = state.get("response_phase", "complete")
    intent_results = state.get("intent_results") or []

    if not intent_results:
        return {"response": "I'm sorry, I couldn't process your request. Please try again."}

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

    context_parts: list[str] = []
    history = state.get("conversation_history") or []
    if history:
        lines = [
            f"{'Customer' if m['role'] == 'human' else 'Assistant'}: {m['content']}"
            for m in history
        ]
        context_parts.append("Previous conversation:\n" + "\n".join(lines))

    orders = state.get("orders_context") or []
    if orders:
        order_strs = [
            f"  {o.get('order_id', '?')} — {o.get('product', '?')} ({o.get('status', '?')})"
            for o in orders
        ]
        context_parts.append("Known orders:\n" + "\n".join(order_strs))

    context_prefix = "\n\n".join(context_parts)
    user_content = f"Customer message: {user_message}\n\nIntent results:\n{results_summary}"
    if context_prefix:
        user_content = f"{context_prefix}\n\n{user_content}"

    llm = ChatGroq(
        model=setting.groq_model,
        api_key=setting.groq_api_key,
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
            tenant_id, customer_id, conversation_id, response.content,
        )

    return {"response": response.content}
