import asyncio
from types import SimpleNamespace

from app.services.brain.contracts import ContextBundle, OrchestrationResult, RoutingDecision
from app.services.brain_service import BrainService
import app.services.brain_service as brain_module


def test_think_enforces_imagine_call_when_model_skips_generate_image(monkeypatch):
    brain = BrainService()
    generate_calls = []
    stored_messages = []

    async def fake_add_message(conversation_id, role, content, attachments=None):
        assert role == "user"
        return SimpleNamespace(id=1)

    async def fake_get_messages(conversation_id, user_id):
        return []

    async def fake_build(**kwargs):
        return ContextBundle(system_prompt="system", recent_messages=[])

    async def fake_route(user_input):
        return RoutingDecision(
            mode="tools",
            reason="explicit_image_request",
            intent="image_generation",
            allowed_tools=["generate_image"],
        )

    def fake_filter_tools(all_tools, decision):
        return [tool for tool in all_tools if tool["function"]["name"] == "generate_image"]

    async def fake_tool_run(**kwargs):
        return OrchestrationResult(
            final_answer="The image has been created and saved to /app/data/generated/fake.jpg",
            messages=[],
            tool_results_summary=[],
            generated_images=[],
            executed_tools=[],
            image_generation_result="",
        )

    async def fake_generate(prompt, model_type="sdxl-lightning", image_path=None):
        generate_calls.append((prompt, model_type, image_path))
        return "Image generated successfully! Saved to: /app/data/generated/real.jpg\n\n![Generated Image](/generated/real.jpg)"

    async def fake_store(conversation_id, user_id, user_message_id, user_input, assistant_response):
        stored_messages.append(assistant_response)

    monkeypatch.setattr(brain_module.memory_service, "add_message", fake_add_message)
    monkeypatch.setattr(brain_module.memory_service, "get_messages", fake_get_messages)
    monkeypatch.setattr(brain.context_builder, "build", fake_build)
    monkeypatch.setattr(brain.router, "route", fake_route)
    monkeypatch.setattr(brain.router, "filter_tools", fake_filter_tools)
    monkeypatch.setattr(brain.tool_orchestrator, "run", fake_tool_run)
    monkeypatch.setattr(brain, "_generate_image_request", fake_generate)
    monkeypatch.setattr(brain, "_store_assistant_turn", fake_store)

    result = asyncio.run(
        brain.think(
            user_input="generate an image of a spaceship cockpit orbiting the earth",
            conversation_id="conv-1",
            user_id=7,
            imagine_model="sdxl-lightning",
        )
    )

    assert generate_calls == [
        (
            "generate an image of a spaceship cockpit orbiting the earth",
            "sdxl-lightning",
            None,
        )
    ]
    assert result == stored_messages[-1]
    assert "![Generated Image](/generated/real.jpg)" in result
    assert "/app/data/generated/fake.jpg" not in result


def test_think_prefers_actual_generate_image_tool_result_over_model_narration(monkeypatch):
    brain = BrainService()
    stored_messages = []

    async def fake_add_message(conversation_id, role, content, attachments=None):
        return SimpleNamespace(id=1)

    async def fake_get_messages(conversation_id, user_id):
        return []

    async def fake_build(**kwargs):
        return ContextBundle(system_prompt="system", recent_messages=[])

    async def fake_route(user_input):
        return RoutingDecision(
            mode="tools",
            reason="explicit_image_request",
            intent="image_generation",
            allowed_tools=["generate_image"],
        )

    def fake_filter_tools(all_tools, decision):
        return [tool for tool in all_tools if tool["function"]["name"] == "generate_image"]

    async def fake_tool_run(**kwargs):
        return OrchestrationResult(
            final_answer="The cockpit render is done and saved to /app/data/generated/hallucinated.jpg",
            messages=[],
            tool_results_summary=[],
            generated_images=["![Generated Image](/generated/real.jpg)"],
            executed_tools=["generate_image"],
            image_generation_result="Image generated successfully! Saved to: /app/data/generated/real.jpg\n\n![Generated Image](/generated/real.jpg)",
        )

    async def fake_generate(*args, **kwargs):
        raise AssertionError("_generate_image_request should not be called when the tool already succeeded")

    async def fake_store(conversation_id, user_id, user_message_id, user_input, assistant_response):
        stored_messages.append(assistant_response)

    monkeypatch.setattr(brain_module.memory_service, "add_message", fake_add_message)
    monkeypatch.setattr(brain_module.memory_service, "get_messages", fake_get_messages)
    monkeypatch.setattr(brain.context_builder, "build", fake_build)
    monkeypatch.setattr(brain.router, "route", fake_route)
    monkeypatch.setattr(brain.router, "filter_tools", fake_filter_tools)
    monkeypatch.setattr(brain.tool_orchestrator, "run", fake_tool_run)
    monkeypatch.setattr(brain, "_generate_image_request", fake_generate)
    monkeypatch.setattr(brain, "_store_assistant_turn", fake_store)

    result = asyncio.run(
        brain.think(
            user_input="generate an image of a spaceship cockpit orbiting the earth",
            conversation_id="conv-1",
            user_id=7,
            imagine_model="sdxl-lightning",
        )
    )

    assert result == stored_messages[-1]
    assert "![Generated Image](/generated/real.jpg)" in result
    assert "hallucinated.jpg" not in result
