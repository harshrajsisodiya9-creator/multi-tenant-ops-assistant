from __future__ import annotations

import json

from langchain_groq import ChatGroq

from app.config import setting
from app.graph.agent.tools import TOOL_DESCRIPTIONS, TOOL_REGISTRY

MAX_ITERATIONS = 8

REACT_SYSTEM_PROMPT = """\
You are a tool-calling agent for e-commerce customer support. You have access to the following tools:

{tools}

To use a tool, respond with EXACTLY this JSON format on a single line:
{{"tool": "<tool_name>", "args": {{"param": "value"}}}}

When you have enough information to give a final answer, respond with:
{{"final_answer": "<your answer>"}}

Rules:
- Always call get_order_details first to verify the order exists.
- For refunds: check return eligibility before initiating.
- For cancellations: the cancel_order tool checks status internally.
- Never skip eligibility checks.
- The tenant_id is provided automatically — do not include it in args.
- Respect the constraints given to you (refund_type, condition, etc.).
- Be concise in your final answer.

Constraints for this request:
{constraints}"""


def _format_tools() -> str:
    lines = []
    for t in TOOL_DESCRIPTIONS:
        lines.append(f"- {t['name']}: {t['description']} (params: {', '.join(t['parameters'])})")
    return "\n".join(lines)


def _parse_agent_response(text: str) -> dict | None:
    text = text.strip()
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return None


def _action_check_in_loop(
    tool_name: str, tool_args: dict, constraints: dict
) -> str | None:
    if tool_name == "initiate_refund":
        max_amount = constraints.get("max_amount")
        if max_amount is not None and tool_args.get("amount", 0) > max_amount:
            return f"Refund amount {tool_args.get('amount')} exceeds maximum allowed {max_amount}."

        required_type = constraints.get("refund_type")
        if required_type and tool_args.get("refund_type") != required_type:
            return f"Refund type must be '{required_type}', got '{tool_args.get('refund_type')}'."

    if tool_name == "cancel_order":
        if constraints.get("cancel_blocked"):
            return "Cancellation is not allowed per policy constraints."

    return None


# Next phase converts this to a subgraph with reason → act → reflect nodes
# and per-step checkpointing for mid-run resume.
def run_react_agent(
    tenant_id: str,
    order_id: str,
    user_message: str,
    intent: str,
    constraints: dict | None = None,
    *,
    llm: ChatGroq | None = None,
) -> dict:
    constraints = constraints or {}

    system_prompt = REACT_SYSTEM_PROMPT.format(
        tools=_format_tools(),
        constraints=json.dumps(constraints) if constraints else "None",
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Order: {order_id}\nRequest: {user_message}"},
    ]

    if llm is None:
        llm = ChatGroq(
            model=setting.groq_model,
            api_key=setting.groq_api_key,
            temperature=0,
        )

    tool_calls: list[dict] = []

    for iteration in range(MAX_ITERATIONS):
        response = llm.invoke(messages)
        agent_text = response.content

        parsed = _parse_agent_response(agent_text)

        if parsed is None:
            return {
                "result": agent_text,
                "tool_calls": tool_calls,
                "iterations": iteration + 1,
            }

        if "final_answer" in parsed:
            return {
                "result": parsed["final_answer"],
                "tool_calls": tool_calls,
                "iterations": iteration + 1,
            }

        if "tool" in parsed:
            tool_name = parsed["tool"]
            tool_args = parsed.get("args", {})

            if tool_name not in TOOL_REGISTRY:
                messages.append({"role": "assistant", "content": agent_text})
                messages.append({"role": "user", "content": f"Error: Unknown tool '{tool_name}'."})
                continue

            violation = _action_check_in_loop(tool_name, tool_args, constraints)
            if violation:
                messages.append({"role": "assistant", "content": agent_text})
                messages.append({"role": "user", "content": f"BLOCKED: {violation}"})
                tool_calls.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "blocked": True,
                    "reason": violation,
                })
                continue

            tool_fn = TOOL_REGISTRY[tool_name]
            tool_result = tool_fn(tenant_id, **tool_args)

            tool_calls.append({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result,
            })

            messages.append({"role": "assistant", "content": agent_text})
            messages.append({"role": "user", "content": f"Tool result:\n{json.dumps(tool_result, indent=2)}"})

    return {
        "result": "Agent reached maximum iterations without a final answer.",
        "tool_calls": tool_calls,
        "iterations": MAX_ITERATIONS,
    }
