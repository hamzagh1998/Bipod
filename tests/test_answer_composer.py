from app.services.brain.answer_composer import AnswerComposer
from app.services.brain.contracts import OrchestrationResult


def test_compose_merges_generated_images_and_sanitizes_internal_state():
    composer = AnswerComposer()
    orchestration = OrchestrationResult(
        final_answer="ignored",
        generated_images=["![Generated Image](/generated/test.jpg)"],
        tool_results_summary=[],
    )

    result = composer.compose(
        "<|thought|>Here you go.\n\n[[BIPOD_WEB_SEARCH: current france president]]\nhttp://localhost:3333/secret",
        orchestration,
    )

    assert "<|thought|>" not in result
    assert "BIPOD_WEB_SEARCH" not in result
    assert "localhost:3333" not in result
    assert "Here you go." in result
    assert "![Generated Image](/generated/test.jpg)" in result


def test_compose_falls_back_to_tool_results_when_answer_is_empty():
    composer = AnswerComposer()
    orchestration = OrchestrationResult(
        final_answer="",
        generated_images=[],
        tool_results_summary=["Tool result 1", "Tool result 2"],
    )

    result = composer.compose("", orchestration)

    assert result == "Tool result 1\n\nTool result 2"


def test_compose_returns_default_prompt_when_everything_is_empty():
    composer = AnswerComposer()
    orchestration = OrchestrationResult(final_answer="", generated_images=[], tool_results_summary=[])

    result = composer.compose("", orchestration)

    assert "Could you rephrase your request?" in result
