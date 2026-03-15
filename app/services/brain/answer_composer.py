import re

from app.core.logger import get_logger
from app.services.brain.contracts import OrchestrationResult

logger = get_logger("bipod.services.brain.answer")


class AnswerComposer:
    """Final user-facing answer cleanup and fallback selection."""

    INTERNAL_MARKERS = (
        "<|python_tag|>",
        "<|action_tag|>",
        "<|thought|>",
    )

    def compose(self, final_answer: str, orchestration: OrchestrationResult) -> str:
        answer = self._sanitize(final_answer)

        for img_tag in orchestration.generated_images:
            if img_tag not in answer:
                answer = f"{answer}\n\n{img_tag}".strip() if answer else img_tag

        if answer.strip():
            return answer.strip()

        if orchestration.tool_results_summary:
            logger.info("Composer using tool results fallback because the model returned an empty answer.")
            fallback = "\n\n".join(orchestration.tool_results_summary)
            return self._sanitize(fallback).strip()

        return "I'm listening, but I didn't quite get that. Could you rephrase your request?"

    def _sanitize(self, text: str) -> str:
        if not text:
            return ""

        sanitized = text
        for marker in self.INTERNAL_MARKERS:
            sanitized = sanitized.replace(marker, "")

        sanitized = re.sub(r"\[\[\s*BIPOD_WEB_SEARCH\s*:\s*.+?\s*\]\]", "", sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r"https?://(?:imagine|ollama|localhost):\d+\S*", "", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()


answer_composer = AnswerComposer()
