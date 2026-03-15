import re
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.logger import get_logger
from app.services.brain.contracts import RoutingDecision
from app.services.intent_router import intent_router

logger = get_logger("bipod.services.brain.router")


class RouterService:
    """Rule-first request router with semantic fallback for ambiguous prompts."""

    TOOL_MAP: Dict[str, List[str]] = {
        "web_search": ["web_search", "fetch_web_page"],
        "fetch_web_page": ["fetch_web_page", "web_search"],
        "image_generation": ["generate_image"],
        "system_info": ["get_system_info"],
        "file_operation": ["read_file", "save_file", "search_files", "move_file", "delete_file", "organize_files"],
        "coding": ["read_file", "save_file", "search_files", "move_file"],
        "vision_analysis": ["analyze_image_file"],
        "troubleshooting": ["read_file", "search_files", "get_system_info", "fetch_web_page"],
    }

    async def route(self, user_input: str) -> RoutingDecision:
        normalized = self._normalize(user_input)
        if not normalized:
            return RoutingDecision(mode="chat", reason="empty_input")

        url_match = re.search(r"https?://\S+", normalized)
        if url_match:
            return RoutingDecision(
                mode="tools",
                reason="url_detected",
                intent="fetch_web_page",
                allowed_tools=self.TOOL_MAP["fetch_web_page"],
            )

        chat_reason = self._match_chat_bypass(normalized)
        if chat_reason:
            return RoutingDecision(mode="chat", reason=chat_reason)

        explicit_tool_decision = self._match_explicit_tool_route(normalized)
        if explicit_tool_decision:
            return explicit_tool_decision

        if settings.ROUTER_USE_SEMANTIC_FALLBACK:
            semantic_decision = await self._route_with_semantic_fallback(user_input)
            if semantic_decision:
                return semantic_decision

        return RoutingDecision(mode="chat", reason="default_chat")

    def filter_tools(self, all_tools: List[Dict], decision: RoutingDecision) -> List[Dict]:
        if decision.mode != "tools" or not decision.allowed_tools:
            return []
        allowed = set(decision.allowed_tools)
        return [tool for tool in all_tools if tool["function"]["name"] in allowed]

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _match_chat_bypass(self, normalized: str) -> Optional[str]:
        conversational_phrases = (
            "who are you",
            "who r u",
            "what are you",
            "introduce yourself",
            "tell me about yourself",
            "hello",
            "hey",
            "good morning",
            "good evening",
            "how are you",
            "what can you do",
            "is this model good",
            "what model are you using",
            "which model are you using",
            "current model",
            "active model",
        )
        if any(phrase in normalized for phrase in conversational_phrases):
            return "conversational_query"

        writing_help_markers = (
            "essay",
            "paragraph",
            "feedback",
            "score my",
            "give me a score",
            "gimme a score",
            "grade this",
            "rewrite this",
            "improve this writing",
            "check grammar",
            "summarize this text",
            "subject for an essay",
        )
        if any(marker in normalized for marker in writing_help_markers):
            return "writing_help"

        return None

    def _match_explicit_tool_route(self, normalized: str) -> Optional[RoutingDecision]:
        if self._is_image_generation_request(normalized):
            return RoutingDecision(
                mode="tools",
                reason="explicit_image_request",
                intent="image_generation",
                allowed_tools=self.TOOL_MAP["image_generation"],
            )

        if self._is_system_info_request(normalized):
            return RoutingDecision(
                mode="tools",
                reason="explicit_system_info_request",
                intent="system_info",
                allowed_tools=self.TOOL_MAP["system_info"],
            )

        if self._is_explicit_web_search_request(normalized) or self._is_time_sensitive_fact_query(normalized):
            return RoutingDecision(
                mode="tools",
                reason="explicit_or_current_web_request",
                intent="web_search",
                allowed_tools=self.TOOL_MAP["web_search"],
            )

        if self._is_file_request(normalized):
            return RoutingDecision(
                mode="tools",
                reason="explicit_file_request",
                intent="file_operation",
                allowed_tools=self.TOOL_MAP["file_operation"],
            )

        if self._is_vision_request(normalized):
            return RoutingDecision(
                mode="tools",
                reason="explicit_vision_request",
                intent="vision_analysis",
                allowed_tools=self.TOOL_MAP["vision_analysis"],
            )

        return None

    async def _route_with_semantic_fallback(self, user_input: str) -> Optional[RoutingDecision]:
        semantic_result = await intent_router.classify_with_scores(
            user_input,
            threshold=settings.ROUTER_SEMANTIC_THRESHOLD,
            margin=settings.ROUTER_MARGIN_THRESHOLD,
        )
        if not semantic_result:
            return None

        intent, score, margin = semantic_result
        allowed_tools = self.TOOL_MAP.get(intent, [])
        if not allowed_tools:
            return None

        logger.info(
            "Semantic fallback routed '%s' with score %.2f and margin %.2f",
            intent,
            score,
            margin,
        )
        return RoutingDecision(
            mode="tools",
            reason="semantic_fallback",
            intent=intent,
            allowed_tools=allowed_tools,
            use_semantic_fallback=True,
        )

    def _is_image_generation_request(self, normalized: str) -> bool:
        direct_patterns = (
            "generate an image",
            "generate image",
            "create an image",
            "create a picture",
            "create a photo",
            "make a picture",
            "make an image",
            "draw ",
            "paint ",
            "illustrate ",
            "render ",
            "sketch ",
            "upscale this image",
            "edit this image",
        )
        if any(pattern in normalized for pattern in direct_patterns):
            return True

        visual_nouns = (
            "image",
            "picture",
            "photo",
            "portrait",
            "wallpaper",
            "logo",
            "poster",
            "banner",
            "drawing",
            "painting",
            "illustration",
            "sketch",
            "avatar",
            "icon",
        )
        visual_verbs = ("generate", "create", "make", "design", "edit", "upscale")
        return any(verb in normalized for verb in visual_verbs) and any(noun in normalized for noun in visual_nouns)

    def _is_system_info_request(self, normalized: str) -> bool:
        keywords = (
            "what time",
            "current time",
            "time in utc",
            "utc time",
            "gpu status",
            "cpu usage",
            "system info",
            "what os",
            "memory usage",
            "hardware",
            "motherboard",
            "chipset",
            "pcie",
        )
        return any(keyword in normalized for keyword in keywords)

    def _is_explicit_web_search_request(self, normalized: str) -> bool:
        phrases = (
            "search for",
            "search the web",
            "look it up",
            "browse the web",
            "check online",
            "google",
            "use the internet",
            "find online",
            "search online",
        )
        return any(phrase in normalized for phrase in phrases)

    def _is_time_sensitive_fact_query(self, normalized: str) -> bool:
        freshness_terms = ("current", "latest", "today", "now", "recent", "this week")
        topic_terms = (
            "president",
            "prime minister",
            "secretary",
            "secretary of defense",
            "defense secretary",
            "minister",
            "defense minister",
            "attorney general",
            "foreign minister",
            "chancellor",
            "speaker",
            "leader",
            "office holder",
            "ceo",
            "price",
            "stock",
            "bitcoin",
            "weather",
            "news",
            "score",
            "winner",
            "exchange rate",
        )
        has_freshness = any(term in normalized for term in freshness_terms)
        if not has_freshness:
            return False

        if any(topic in normalized for topic in topic_terms):
            return True

        # Generalize current office-holder lookups like
        # "who is the current secretary of defense of the united states".
        office_holder_patterns = (
            r"\bwho(?:'s| is)\s+(?:the\s+)?(?:current|latest|present)\s+[\w\s-]+?\s+of\s+[\w\s.-]+\b",
            r"\b(?:current|latest|present)\s+[\w\s-]+?\s+of\s+[\w\s.-]+\b",
        )
        return any(re.search(pattern, normalized) for pattern in office_holder_patterns)

    def _is_file_request(self, normalized: str) -> bool:
        phrases = (
            "read file",
            "open file",
            "find file",
            "search files",
            "save file",
            "write file",
            "rename file",
            "delete file",
            "move file",
            "organize files",
        )
        has_path = bool(re.search(r"(/[\w./-]+)|([\w.-]+\.(py|js|ts|md|txt|json|pdf|csv|png|jpg))", normalized))
        return any(phrase in normalized for phrase in phrases) or has_path

    def _is_vision_request(self, normalized: str) -> bool:
        phrases = (
            "what is in this image",
            "describe this image",
            "analyze this image",
            "look at this image",
            "what do you see",
            "describe the screenshot",
        )
        return any(phrase in normalized for phrase in phrases)


router_service = RouterService()
