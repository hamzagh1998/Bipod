import asyncio
from types import SimpleNamespace

import app.services.brain_service as brain_module
from app.services.brain.contracts import ContextBundle, RoutingDecision
from app.services.brain_service import BrainService


def test_extract_local_file_path_handles_spaces_and_parentheses():
    brain = BrainService()
    user_input = (
        "read and summarize with great detail the file at "
        "/home/hamza/Documents/pdf/The Tibetan Book of the Dead First Complete "
        "Translation (Penguin Classics Deluxe Edition).pdf"
    )

    assert brain._extract_local_file_path(user_input) == (
        "/home/hamza/Documents/pdf/The Tibetan Book of the Dead First Complete "
        "Translation (Penguin Classics Deluxe Edition).pdf"
    )


def test_should_handoff_local_file_read_requires_explicit_read_like_request():
    brain = BrainService()
    path = "/home/hamza/Documents/pdf/book.pdf"

    assert brain._should_handoff_local_file_read(
        f"read and summarize {path}",
        "file_operation",
    ) is True
    assert brain._should_handoff_local_file_read(
        f"this path exists {path}",
        "file_operation",
    ) is False


def test_think_uses_local_file_handoff_for_explicit_path_requests(monkeypatch):
    brain = BrainService()
    seen = {}
    request_path = "/home/hamza/Documents/pdf/book.pdf"

    async def fake_add_message(*args, **kwargs):
        return SimpleNamespace(id=42)

    async def fake_get_messages(*args, **kwargs):
        return []

    async def fake_build(*args, **kwargs):
        return ContextBundle(system_prompt="sys", recent_messages=[])

    async def fake_route(*args, **kwargs):
        return RoutingDecision(
            mode="tools",
            reason="explicit_file_request",
            intent="file_operation",
            allowed_tools=["read_file"],
        )

    async def fake_complete_with_file_read(*, file_path, **kwargs):
        seen["file_path"] = file_path
        return "Detailed summary from middleware"

    async def fake_store_assistant_turn(*args, **kwargs):
        seen["stored"] = True

    async def fail_tool_orchestration(*args, **kwargs):
        raise AssertionError("Tool orchestration should be bypassed for direct local file handoff.")

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(brain_module.memory_service, "add_message", fake_add_message)
    monkeypatch.setattr(brain_module.memory_service, "get_messages", fake_get_messages)
    monkeypatch.setattr(brain.context_builder, "build", fake_build)
    monkeypatch.setattr(brain.router, "route", fake_route)
    monkeypatch.setattr(brain, "_complete_with_file_read", fake_complete_with_file_read)
    monkeypatch.setattr(brain, "_store_assistant_turn", fake_store_assistant_turn)
    monkeypatch.setattr(brain.tool_orchestrator, "run", fail_tool_orchestration)
    monkeypatch.setattr(brain_module.httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(
        brain.think(
            user_input=f"read and summarize with great detail the file at {request_path}",
            conversation_id="conv-1",
            user_id=7,
        )
    )

    assert result == "Detailed summary from middleware"
    assert seen["file_path"] == request_path
    assert seen["stored"] is True
