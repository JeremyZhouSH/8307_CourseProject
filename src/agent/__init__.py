"""代理层编排、规划与状态容器。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.controller import AgentController
    from src.agent.dialogue import DialogueManager
    from src.agent.memory import AgentMemoryStore, MemoryRecord
    from src.agent.planner import AgentAction, AgentPlanner
    from src.agent.reviewer import AgentReviewer, ReviewDecision
    from src.agent.state import PipelineState
    from src.agent.tools import AgentTools

__all__ = [
    "AgentAction",
    "AgentController",
    "AgentMemoryStore",
    "AgentPlanner",
    "AgentReviewer",
    "AgentTools",
    "DialogueManager",
    "MemoryRecord",
    "PipelineState",
    "ReviewDecision",
]


# 函数作用：内部辅助逻辑，服务当前类/模块主流程。
def __getattr__(name: str) -> Any:
    if name == "AgentAction":
        from src.agent.planner import AgentAction
        return AgentAction
    if name == "AgentController":
        from src.agent.controller import AgentController
        return AgentController
    if name == "AgentMemoryStore":
        from src.agent.memory import AgentMemoryStore
        return AgentMemoryStore
    if name == "AgentPlanner":
        from src.agent.planner import AgentPlanner
        return AgentPlanner
    if name == "AgentReviewer":
        from src.agent.reviewer import AgentReviewer
        return AgentReviewer
    if name == "AgentTools":
        from src.agent.tools import AgentTools
        return AgentTools
    if name == "DialogueManager":
        from src.agent.dialogue import DialogueManager
        return DialogueManager
    if name == "MemoryRecord":
        from src.agent.memory import MemoryRecord
        return MemoryRecord
    if name == "PipelineState":
        from src.agent.state import PipelineState
        return PipelineState
    if name == "ReviewDecision":
        from src.agent.reviewer import ReviewDecision
        return ReviewDecision
    raise AttributeError(f"module 'src.agent' has no attribute {name!r}")
