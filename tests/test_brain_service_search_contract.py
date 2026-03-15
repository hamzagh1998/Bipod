import asyncio
from types import SimpleNamespace

import app.services.brain_service as brain_module
from app.services.brain.contracts import ContextBundle, OrchestrationResult, RoutingDecision
from app.services.brain_service import BrainService


def test_think_enforces_web_search_contract_when_routed_query_skips_search(monkeypatch):
    brain = BrainService()
    seen = {"search_calls": 0}

    async def fake_add_message(*args, **kwargs):
        return SimpleNamespace(id=42)

    async def fake_get_messages(*args, **kwargs):
        return []

    async def fake_build(*args, **kwargs):
        return ContextBundle(system_prompt="sys", recent_messages=[])

    async def fake_route(*args, **kwargs):
        return RoutingDecision(
            mode="tools",
            reason="explicit_or_current_web_request",
            intent="web_search",
            allowed_tools=["web_search", "fetch_web_page"],
        )

    async def fake_run(*args, **kwargs):
        return OrchestrationResult(
            final_answer="As of my knowledge cutoff, the answer is Lloyd Austin.",
            messages=[],
            tool_results_summary=[],
            generated_images=[],
            executed_tools=[],
        )

    async def fake_complete_with_web_search(*, search_query, **kwargs):
        seen["search_calls"] += 1
        seen["query"] = search_query
        return "The current Secretary of Defense is from web search."

    async def fake_store_assistant_turn(*args, **kwargs):
        seen["stored"] = True

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
    monkeypatch.setattr(brain.tool_orchestrator, "run", fake_run)
    monkeypatch.setattr(brain, "_complete_with_web_search", fake_complete_with_web_search)
    monkeypatch.setattr(brain, "_store_assistant_turn", fake_store_assistant_turn)
    monkeypatch.setattr(brain_module.httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(
        brain.think(
            user_input="who is the current secretary of defense of the united states",
            conversation_id="conv-1",
            user_id=7,
        )
    )

    assert result == "The current Secretary of Defense is from web search."
    assert seen["search_calls"] == 1
    assert seen["query"] == "who is the current secretary of defense of the united states"
    assert seen["stored"] is True


def test_think_does_not_repeat_search_when_web_lookup_already_executed(monkeypatch):
    brain = BrainService()
    seen = {"search_calls": 0}

    async def fake_add_message(*args, **kwargs):
        return SimpleNamespace(id=42)

    async def fake_get_messages(*args, **kwargs):
        return []

    async def fake_build(*args, **kwargs):
        return ContextBundle(system_prompt="sys", recent_messages=[])

    async def fake_route(*args, **kwargs):
        return RoutingDecision(
            mode="tools",
            reason="explicit_or_current_web_request",
            intent="web_search",
            allowed_tools=["web_search", "fetch_web_page"],
        )

    async def fake_run(*args, **kwargs):
        return OrchestrationResult(
            final_answer="Pete Hegseth is the current Secretary of Defense.",
            messages=[],
            tool_results_summary=["Search results for 'current secretary of defense'"],
            generated_images=[],
            executed_tools=["web_search"],
        )

    async def fake_complete_with_web_search(*args, **kwargs):
        seen["search_calls"] += 1
        return "should not happen"

    async def fake_store_assistant_turn(*args, **kwargs):
        seen["stored"] = True

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
    monkeypatch.setattr(brain.tool_orchestrator, "run", fake_run)
    monkeypatch.setattr(brain, "_complete_with_web_search", fake_complete_with_web_search)
    monkeypatch.setattr(brain, "_store_assistant_turn", fake_store_assistant_turn)
    monkeypatch.setattr(brain_module.httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(
        brain.think(
            user_input="who is the current secretary of defense of the united states",
            conversation_id="conv-1",
            user_id=7,
        )
    )

    assert result == "Pete Hegseth is the current Secretary of Defense."
    assert seen["search_calls"] == 0
    assert seen["stored"] is True


def test_resolve_requested_model_rejects_light_model_on_non_arm64(monkeypatch):
    brain = BrainService()

    monkeypatch.setattr(brain_module.settings, "HARDWARE_TARGET", "amd64")
    monkeypatch.setattr(brain, "active_model", brain_module.settings.MEDIUM_MODEL)

    resolved = brain._resolve_requested_model(brain_module.settings.LIGHT_MODEL)

    assert resolved == brain_module.settings.MEDIUM_MODEL


def test_resolve_requested_model_allows_light_model_on_arm64(monkeypatch):
    brain = BrainService()

    monkeypatch.setattr(brain_module.settings, "HARDWARE_TARGET", "arm64")

    resolved = brain._resolve_requested_model(brain_module.settings.LIGHT_MODEL)

    assert resolved == brain_module.settings.LIGHT_MODEL
