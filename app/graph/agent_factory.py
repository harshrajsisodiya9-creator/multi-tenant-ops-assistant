from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import setting

if TYPE_CHECKING:
    from typing import Callable


def get_react_agent() -> Callable:
    if setting.agent_mode == "prod":
        from app.graph.agent_prod.react_agent import run_react_agent
    else:
        from app.graph.agent.react_agent import run_react_agent
    return run_react_agent


def get_tool_registry() -> dict[str, Callable]:
    if setting.agent_mode == "prod":
        from app.graph.agent_prod.tools import TOOL_REGISTRY
    else:
        from app.graph.agent.tools import TOOL_REGISTRY
    return TOOL_REGISTRY


def get_tool_descriptions() -> list[dict]:
    if setting.agent_mode == "prod":
        from app.graph.agent_prod.tools import TOOL_DESCRIPTIONS
    else:
        from app.graph.agent.tools import TOOL_DESCRIPTIONS
    return TOOL_DESCRIPTIONS
