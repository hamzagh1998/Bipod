from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class BrainRequest(BaseModel):
    user_input: str
    conversation_id: str
    user_id: int
    model_id: Optional[str] = None
    reasoning_mode: Optional[str] = None
    imagine_model: Optional[str] = None
    attachments: Optional[List[dict]] = None


class RoutingDecision(BaseModel):
    mode: Literal["chat", "tools", "handoff"] = "chat"
    reason: str
    intent: Optional[str] = None
    allowed_tools: List[str] = Field(default_factory=list)
    use_semantic_fallback: bool = False


class ContextBundle(BaseModel):
    system_prompt: str
    recent_messages: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    memory_context: str = ""
    thread_has_images: bool = False
    current_image_paths: List[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    tool: str
    status: Literal["ok", "error"]
    payload: Dict[str, Any] = Field(default_factory=dict)
    user_safe_summary: str


class OrchestrationResult(BaseModel):
    final_answer: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results_summary: List[str] = Field(default_factory=list)
    generated_images: List[str] = Field(default_factory=list)
    executed_tools: List[str] = Field(default_factory=list)


class BrainResponse(BaseModel):
    response: str
    routing: RoutingDecision
    tool_results: List[ToolResult] = Field(default_factory=list)
