import asyncio

from app.services.brain.contracts import RoutingDecision
from app.services.brain.router_service import RouterService


def test_route_identity_query_bypasses_tools(monkeypatch):
    router = RouterService()
    monkeypatch.setattr(router, "_route_with_semantic_fallback", _async_return(None))

    decision = asyncio.run(router.route("who are you?"))

    assert decision.mode == "chat"
    assert decision.reason == "conversational_query"
    assert decision.allowed_tools == []


def test_route_writing_help_bypasses_tools(monkeypatch):
    router = RouterService()
    monkeypatch.setattr(router, "_route_with_semantic_fallback", _async_return(None))

    decision = asyncio.run(router.route("gimme an essay subject and later score my response"))

    assert decision.mode == "chat"
    assert decision.reason == "writing_help"


def test_route_time_sensitive_fact_uses_web_search(monkeypatch):
    router = RouterService()
    monkeypatch.setattr(router, "_route_with_semantic_fallback", _async_return(None))

    decision = asyncio.run(router.route("who is the current president of france"))

    assert decision.mode == "tools"
    assert decision.intent == "web_search"
    assert "web_search" in decision.allowed_tools


def test_route_current_secretary_of_defense_uses_web_search(monkeypatch):
    router = RouterService()
    monkeypatch.setattr(router, "_route_with_semantic_fallback", _async_return(None))

    decision = asyncio.run(router.route("who is the current secretary of defense of the united states"))

    assert decision.mode == "tools"
    assert decision.intent == "web_search"
    assert "web_search" in decision.allowed_tools


def test_route_time_query_uses_system_info(monkeypatch):
    router = RouterService()
    monkeypatch.setattr(router, "_route_with_semantic_fallback", _async_return(None))

    decision = asyncio.run(router.route("gimme the time in utc"))

    assert decision.mode == "tools"
    assert decision.intent == "system_info"
    assert decision.allowed_tools == ["get_system_info"]


def test_route_image_request_uses_generate_image(monkeypatch):
    router = RouterService()
    monkeypatch.setattr(router, "_route_with_semantic_fallback", _async_return(None))

    decision = asyncio.run(router.route("generate an image of a red car"))

    assert decision.mode == "tools"
    assert decision.intent == "image_generation"
    assert decision.allowed_tools == ["generate_image"]


def test_route_malformed_image_request_still_uses_generate_image(monkeypatch):
    router = RouterService()
    monkeypatch.setattr(router, "_route_with_semantic_fallback", _async_return(None))

    decision = asyncio.run(router.route("generate an of spaceship orbiting earth"))

    assert decision.mode == "tools"
    assert decision.intent == "image_generation"
    assert decision.allowed_tools == ["generate_image"]


def test_semantic_fallback_can_route_when_enabled(monkeypatch):
    router = RouterService()
    monkeypatch.setattr(
        router,
        "_route_with_semantic_fallback",
        _async_return(
            RoutingDecision(
                mode="tools",
                reason="semantic_fallback",
                intent="troubleshooting",
                allowed_tools=["read_file", "search_files"],
                use_semantic_fallback=True,
            )
        ),
    )

    decision = asyncio.run(router.route("this keeps crashing and I need help"))

    assert decision.mode == "tools"
    assert decision.use_semantic_fallback is True
    assert decision.intent == "troubleshooting"


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner
