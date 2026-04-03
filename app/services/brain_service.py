import json
import os
import httpx
import re
import asyncio
import datetime
import base64
import uuid
import html
import warnings
from typing import Any, Awaitable, Callable, Dict, List, Optional

try:
    from ddgs import DDGS  # type: ignore
except ImportError:
    from duckduckgo_search import DDGS

from app.core.config import settings
from app.core.logger import get_logger
from app.services.brain.answer_composer import answer_composer
from app.services.brain.context_builder import ContextBuilder
from app.services.brain.router_service import router_service
from app.services.brain.tool_orchestrator import ToolOrchestrator
from app.services.memory_service import memory_service
from app.services.vector_service import vector_service
from app.services.file_service import file_service

logger = get_logger("bipod.brain")

ProgressCallback = Callable[[str, Dict[str, Any]], Awaitable[None]]


class BrainService:
    """The central intelligence service of Bipod with tool-calling capabilities."""

    FILE_HANDOFF_EXTENSIONS = (
        "pdf",
        "txt",
        "md",
        "rtf",
        "docx",
        "csv",
        "json",
        "xml",
        "html",
        "css",
        "js",
        "ts",
        "tsx",
        "jsx",
        "py",
        "java",
        "c",
        "cpp",
        "rs",
        "go",
        "sh",
        "yaml",
        "yml",
    )
    WEB_SEARCH_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "latest",
        "me",
        "my",
        "news",
        "now",
        "of",
        "on",
        "please",
        "recent",
        "show",
        "tell",
        "the",
        "this",
        "today",
        "up",
        "update",
        "what",
        "when",
        "where",
        "who",
        "why",
    }
    WEB_SEARCH_QUERY_REPLACEMENTS = (
        (r"\bwho's\b", "who is"),
        (r"\bwhat's\b", "what is"),
        (r"\bdefence\b", "defense"),
        (r"\bminister(?:y)? of war\b", "secretary of defense"),
        (r"\bsecretary of war\b", "secretary of defense"),
        (r"\bunited state of america\b", "united states"),
        (r"\bunited state\b", "united states"),
        (r"\bu\.s\.a\.\b", "united states"),
        (r"\bu\.s\.a\b", "united states"),
        (r"\bu\.s\.\b", "united states"),
        (r"\bamerica\b", "united states"),
    )

    OLLAMA_CONNECT_TIMEOUT_SEC = 10.0
    OLLAMA_WRITE_TIMEOUT_SEC = 30.0
    OLLAMA_POOL_TIMEOUT_SEC = 30.0

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.active_model = settings.ACTIVE_MODEL

        # System Prompt following Bipod Philosophy
        self.system_prompt = (
            "You are Bipod, an AI agent running entirely on the user's local machine. "
            "You prioritize privacy — no data ever leaves this device. "
            "You are helpful, concise, and intelligent. "
            "You can have natural conversations on any topic. "
            "You also have filesystem tools available. You can search, read, create, or update TEXT files. "
            "Arbitrary shell command execution is disabled for security reasons. "
            "You can ANALYZE and DESCRIBE existing images at specific paths using the vision model. "
            "You have a specialized tool to get real-time SYSTEM INFORMATION (CPU, GPU, OS, Time). "
            "### CORE DIRECTIVES:\n"
            "1. ALWAYS PRIORITIZE the current user message. Execute exactly what the user asks.\n"
            "2. If a user asks to **draw**, **create**, **make**, or **generate** an image, you **MUST** use the `generate_image` tool.\n"
            "3. Use `search_files` to find filenames. Use `read_file` to see content. Use `save_file` to persist progress.\n"
            "4. DO NOT explain why you are using a tool unless it's a complex multi-step process.\n"
            "5. Use specialized tools (save_file, generate_image, search_files, read_file) for all tasks.\n"
            "6. EACH CONVERSATION IS INDEPENDENT. Do not mix up content from different sessions.\n"
            "7. For image generation, the `model_type` MUST be one of: 'sdxl-lightning' (fast, default), 'juggernaut-xl' (high quality SDXL), 'flux-schnell' (photoreal - requires high VRAM), 'stable-diffusion' (balanced), or 'dalle-mini' (low resource). DO NOT use any other words for `model_type`.\n"
            "8. ALWAYS provide the actual file path returned by the tool when confirming a task (e.g. 'Image saved to /app/data/...'). If an image was generated, you MUST include the markdown preview (e.g. ![Generated Image](/generated/filename.jpg)) EXACTLY as returned by the tool so the user can see it in the chat.\n"
            "[SYSTEM: TOOLS]\n1. `read_file`: Reads text/PDF from the host. \n2. `save_file`: Writes text to the host. \n3. `move_file`: Moves/renames files/dirs on the host. Supports wildcards (e.g. *.pdf). \n4. `delete_file`: Deletes files/dirs on the host. Supports wildcards. \n5. `search_files`: Finds files by pattern. \n6. `get_system_info`: Returns CPU/GPU usage, model info, and current time. \n7. `web_search`: Searches the internet. \n8. `fetch_web_page`: Reads a URL.\n9. `organize_files`: Automatically sorts files in a directory into folders by their extension (e.g. 'pdf/', 'docx/').\n\n"
            "- THE USER CANNOT SEE YOUR TOOL CALLS. If you narrate them, you are talking to yourself and confusing the user.\n"
            "- Just do the work. Once the tool finishes, you can give a final summary.\n"
            "9. If the user asks for the CURRENT TIME, CPU usage, GPU status, or OS info, you MUST use the `get_system_info` tool. DO NOT guess or claim you lack access. \n"
            "10. You have INTERNET ACCESS via `web_search` and `fetch_web_page`. Use them to find current information, news, or to summarize specific web pages. \n"
            "11. **ULTRA-CRITICAL - THE TRUTH DIRECTIVE**: You MUST acknowledge that your internal training data is OUTDATED. "
            "For current events (like 'Who is the Prime Minister?' or 'Bitcoin Price'), you MUST SEARCH THE WEB. "
            "SEARCH RESULTS ARE THE ABSOLUTE SOURCE OF TRUTH. Use 'latest', 'today', or 'current' in your search queries to get fresh snippets.\n"
            "The information you provide MUST come from the tool output. "
            "NEVER mention internal tool names or installation issues to the user. Just execute the tool. \n"
            "12. **SILENT TOOL CALLING**: When tools are available, use the built-in tool-calling interface. Do NOT print JSON tool calls or internal notes to the user.\n"
            "13. If you need current or external factual information and do NOT already have search results, respond ONLY with "
            "`[[BIPOD_WEB_SEARCH: concise search query]]`. Do not answer the user until the search results are provided.\n"
            "### FORMATTING RULES:\n"
            "1. ALWAYS wrap code snippets in triple backticks (```) and specify the programming language (e.g. ```python) for proper syntax highlighting.\n"
            "2. DO NOT just write the code as plain text.\n"
            "Keep your internal thoughts concise and focus only on the logic of the task."
        )

        self.router = router_service
        self.context_builder = ContextBuilder(self.base_url, self._ollama_options)

        # Define tools for Ollama
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Searches the host filesystem for files using a glob pattern (e.g., 'projects/*.py').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The glob pattern to search for.",
                            },
                            "root": {
                                "type": "string",
                                "description": "Optional directory to start search from (default: /).",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Reads the content of a file from the host filesystem.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The absolute or relative path to the file on the host.",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Optional character limit when reading large files.",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_file",
                    "description": "Creates or overwrites a TEXT file on the host filesystem at the specified path. ONLY use this when the user explicitly asks to create, save, or write a file. This tool can only write TEXT content — it CANNOT create binary files like images, audio, or video.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The absolute path on the host filesystem where the file should be saved (e.g. '/home/user/notes.txt').",
                            },
                            "content": {
                                "type": "string",
                                "description": "The exact text content to write into the file.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_image_file",
                    "description": "Reads an EXISTING image file from the host filesystem and describes its content using the vision model (Moondream). This tool ONLY analyzes and describes existing images — it does NOT generate, create, or produce new images.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The absolute path to an EXISTING image file on the host.",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Optional specific question or prompt about the image.",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generates a new premium-quality image based on a text prompt. The tool automatically handles high-resolution descriptors. Returns the path to the generated image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Description of the image. For best results, Bipod will expand this with quality tags like 'cinematic lighting, 8k, highly detailed, masterpiece'.",
                            },
                            "image_path": {
                                "type": "string",
                                "description": "Optional. Path to an existing image file to use for variations.",
                            },
                            "model_type": {
                                "type": "string",
                                "enum": [
                                    "sdxl-lightning",
                                    "juggernaut-xl",
                                    "flux-schnell",
                                    "stable-diffusion",
                                    "dalle-mini",
                                ],
                                "description": "Model to use. Default 'sdxl-lightning'. 'juggernaut-xl' offers higher quality SDXL output.",
                            },
                        },
                        "required": ["prompt"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "move_file",
                    "description": "Moves or renames a file or directory on the host filesystem.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "src": {
                                "type": "string",
                                "description": "The current absolute path to the file/directory.",
                            },
                            "dest": {
                                "type": "string",
                                "description": "The new absolute path or destination directory.",
                            },
                        },
                        "required": ["src", "dest"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Permanently deletes a file or directory from the host filesystem.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The absolute path to the file/directory to delete.",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "organize_files",
                    "description": "Automatically sorts files in a specified directory into subfolders based on their file extensions (e.g., 'pdf/', 'docx/').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "The absolute path to the directory whose files should be organized.",
                            },
                        },
                        "required": ["directory"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_system_command",
                    "description": "Legacy tool retained for compatibility. Arbitrary shell execution is disabled for security reasons.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The full shell command to execute. Prefix host paths with /host.",
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_system_info",
                    "description": "Returns current system information including CPU usage, GPU status, OS details, and current time.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Searches the internet for the given query using DuckDuckGo. Returns a list of relevant search results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up on the internet.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_web_page",
                    "description": "Retrieves the content of a specific web page/URL and returns its text content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the web page to fetch.",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
        ]
        self.tool_orchestrator = ToolOrchestrator(
            base_url=self.base_url,
            tools=self.tools,
            options_provider=self._ollama_options,
            check_hallucinated_tools=self._check_for_hallucinated_tools,
            run_web_search=self._run_web_search,
            vision_request=self._vision_request,
            generate_image_request=self._generate_image_request,
            map_reduce_summarize=self._map_reduce_summarize,
        )

    def _ollama_options(self) -> Dict[str, float | int]:
        """Shared generation settings to reduce repetition and bound context usage."""
        return {
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "temperature": settings.OLLAMA_TEMPERATURE,
            "repeat_penalty": settings.OLLAMA_REPEAT_PENALTY,
        }

    def _ollama_request_timeout(self, include_tools: bool) -> httpx.Timeout:
        """Use a longer read timeout for chat generation, especially with tool turns."""
        read_timeout = (
            float(settings.OLLAMA_TOOL_CHAT_TIMEOUT_SEC)
            if include_tools
            else float(settings.OLLAMA_CHAT_TIMEOUT_SEC)
        )
        return httpx.Timeout(
            connect=self.OLLAMA_CONNECT_TIMEOUT_SEC,
            read=read_timeout,
            write=self.OLLAMA_WRITE_TIMEOUT_SEC,
            pool=self.OLLAMA_POOL_TIMEOUT_SEC,
        )

    def _is_image_generation_request(self, user_input: str) -> bool:
        """Detect explicit requests to create or transform visuals.

        We avoid generic verbs like "make" or "create" on their own because they
        produce false positives for normal chat requests such as "what can this model do?"
        """
        normalized = re.sub(r"\s+", " ", user_input.lower()).strip()
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
        return any(verb in normalized for verb in visual_verbs) and any(
            noun in normalized for noun in visual_nouns
        )

    def _extract_web_search_signal(
        self, content: str, user_input: str
    ) -> Optional[str]:
        """Detect explicit model handoff requests for web search.

        The exact sentinel is preferred, but we also recover a few common
        low-discipline variants from smaller local models and fall back to the
        original user query if the model clearly asked for a lookup.
        """
        if not content:
            return None

        exact = re.search(
            r"\[\[\s*BIPOD_WEB_SEARCH\s*:\s*(.+?)\s*\]\]",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if exact:
            query = re.sub(r"\s+", " ", exact.group(1)).strip()
            return query or user_input

        lowered = content.lower()
        fallback_phrases = (
            "i need to search",
            "i need to look that up",
            "i should search the web",
            "i should look it up",
            "i don't have current information",
            "i do not have current information",
            "i need current information",
            "i need up-to-date information",
            "knowledge cutoff",
            "recommend checking official",
            "check official sources",
            "recent news updates",
            "there may have been a change",
            "may have been a change in leadership",
        )
        if any(phrase in lowered for phrase in fallback_phrases):
            return user_input

        return None

    def _extract_explicit_web_search_signal(
        self, content: str, user_input: str
    ) -> Optional[str]:
        """Read only the explicit model-to-middleware search sentinel."""
        if not content:
            return None

        exact = re.search(
            r"\[\[\s*BIPOD_WEB_SEARCH\s*:\s*(.+?)\s*\]\]",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if not exact:
            return None

        query = re.sub(r"\s+", " ", exact.group(1)).strip()
        return query or user_input

    def _orchestration_has_web_lookup(self, orchestration) -> bool:
        """Return True once the middleware/model exchange already executed web retrieval."""
        return any(
            tool in {"web_search", "fetch_web_page"}
            for tool in orchestration.executed_tools
        )

    def _should_enforce_web_search_contract(
        self, intent: Optional[str], orchestration
    ) -> bool:
        """Current-fact routes must not return a direct answer without web retrieval."""
        return intent == "web_search" and not self._orchestration_has_web_lookup(
            orchestration
        )

    def _normalize_web_search_query(self, query: str) -> str:
        """Canonicalize user phrasing into a more search-engine-friendly query."""
        normalized = re.sub(r"\s+", " ", query.lower()).strip()
        normalized = re.sub(r"[“”]", '"', normalized)
        normalized = re.sub(r"[‘’]", "'", normalized)

        for pattern, replacement in self.WEB_SEARCH_QUERY_REPLACEMENTS:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        normalized = re.sub(r"[?!.]+$", "", normalized).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _extract_search_terms(self, query: str) -> List[str]:
        normalized = self._normalize_web_search_query(query)
        terms = []
        for term in re.findall(r"[a-z0-9]+", normalized):
            if len(term) <= 1 or term in self.WEB_SEARCH_STOPWORDS:
                continue
            if term not in terms:
                terms.append(term)
        return terms

    def _build_web_search_candidates(self, query: str) -> List[str]:
        """Generate a few progressively more canonical search queries for current-fact lookups."""
        normalized = self._normalize_web_search_query(query)
        candidates: List[str] = []
        preferred_domains = self._preferred_official_domains_for_query(normalized)

        def _add(candidate: str) -> None:
            cleaned = re.sub(r"\s+", " ", candidate).strip(" ?!.,")
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

        _add(query)
        _add(normalized)

        role_lookup = self._extract_current_role_lookup(normalized)
        if role_lookup:
            role, entity = role_lookup
            if preferred_domains:
                primary_domain = preferred_domains[0]
                _add(f'site:{primary_domain} "{role}" "{entity}"')
                _add(f'site:{primary_domain} "{role}"')
            _add(f"current {role} {entity} official")
            _add(f"{entity} {role} official")
            _add(f'site:gov "{role}" "{entity}"')
            _add(f'"{role}" "{entity}" official')

        compact_terms = self._extract_search_terms(normalized)
        if compact_terms:
            _add(" ".join(compact_terms))

        office_titles = (
            "president",
            "prime minister",
            "secretary",
            "minister",
            "ceo",
            "mayor",
            "governor",
            "defense",
        )
        if any(title in normalized for title in office_titles):
            _add(f"{normalized} official")

        return candidates[:5]

    def _extract_current_role_lookup(self, query: str) -> Optional[tuple[str, str]]:
        """Extract the role and governing entity from current office-holder questions."""
        normalized = self._normalize_web_search_query(query)
        prefixes = (
            "who is the current ",
            "who is current ",
            "who is the present ",
            "who is present ",
            "who is the latest ",
            "who is latest ",
            "who is the ",
            "who is ",
            "current ",
        )

        remainder = normalized
        for prefix in prefixes:
            if normalized.startswith(prefix):
                remainder = normalized[len(prefix) :]
                break

        role, separator, entity = remainder.rpartition(" of ")
        if separator:
            role = re.sub(r"\s+", " ", role).strip(" ?!.,")
            entity = re.sub(r"\s+", " ", entity).strip(" ?!.,")
            if role and entity:
                return role, entity

        return None

    async def _expand_web_search_candidates_with_model(
        self, query: str, base_candidates: List[str]
    ) -> List[str]:
        """Ask the local model to propose a few better search queries from user intent."""
        prompt = (
            "Rewrite the user's request into up to 4 concise web search queries.\n"
            "Infer the user's intent rather than copying the wording.\n"
            "Normalize contractions, spelling issues, and outdated or incorrect titles when the modern equivalent is obvious.\n"
            "For current office-holder or current-fact questions, prefer queries likely to surface authoritative sources.\n"
            "Return only a JSON array of strings and nothing else."
        )

        try:
            async with httpx.AsyncClient(
                timeout=self._ollama_request_timeout(include_tools=False)
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": settings.ACTIVE_MODEL,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": (
                                    f"User request: {query}\n"
                                    f"Seed queries: {json.dumps(base_candidates)}"
                                ),
                            },
                        ],
                        "stream": False,
                        "options": self._ollama_options(),
                    },
                )
                response.raise_for_status()
        except Exception as exc:
            logger.warning(
                f"Search query planning failed, using heuristic candidates: {exc}"
            )
            return base_candidates

        content = response.json().get("message", {}).get("content", "").strip()
        parsed = self._parse_search_query_candidates(content)
        if not parsed:
            return base_candidates

        merged: List[str] = []
        for candidate in [*parsed, *base_candidates]:
            cleaned = re.sub(r"\s+", " ", candidate).strip(" ?!.,")
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged[:6]

    def _parse_search_query_candidates(self, content: str) -> List[str]:
        """Parse a JSON array or line-based fallback into search query strings."""
        if not content:
            return []

        json_block = re.search(r"\[[\s\S]*\]", content)
        if json_block:
            try:
                data = json.loads(json_block.group(0))
                if isinstance(data, list):
                    return [str(item).strip() for item in data if str(item).strip()]
            except Exception:
                pass

        lines = []
        for line in content.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                lines.append(cleaned)
        return lines[:6]

    def _is_time_sensitive_query(self, query: str) -> bool:
        freshness_terms = {
            "current",
            "latest",
            "news",
            "today",
            "now",
            "recent",
            "update",
            "updated",
            "price",
            "value",
            "cost",
            "score",
            "winner",
            "result",
            "finance",
            "stock",
            "who is",
        }
        lowered = query.lower()
        return any(term in lowered for term in freshness_terms)

    def _extract_result_date(self, result: Dict) -> Optional[datetime.datetime]:
        """Best-effort extraction of a result date from snippet text or URL."""
        text_candidates = [
            str(result.get("title", "")),
            str(result.get("body", "")),
            str(result.get("href", "")),
        ]
        now = datetime.datetime.now(datetime.timezone.utc)

        relative_patterns = (
            (r"\b(\d+)\s+minutes?\s+ago\b", "minutes"),
            (r"\b(\d+)\s+hours?\s+ago\b", "hours"),
            (r"\b(\d+)\s+days?\s+ago\b", "days"),
            (r"\b(\d+)\s+weeks?\s+ago\b", "weeks"),
            (r"\b(\d+)\s+months?\s+ago\b", "months"),
            (r"\b(\d+)\s+years?\s+ago\b", "years"),
        )
        absolute_patterns = (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%b %d, %Y",
            "%B %d, %Y",
            "%d %b %Y",
            "%d %B %Y",
        )

        for candidate in text_candidates:
            lowered = candidate.lower()
            if "today" in lowered:
                return now
            if "yesterday" in lowered:
                return now - datetime.timedelta(days=1)

            for pattern, unit in relative_patterns:
                match = re.search(pattern, lowered)
                if not match:
                    continue

                quantity = int(match.group(1))
                if unit == "minutes":
                    return now - datetime.timedelta(minutes=quantity)
                if unit == "hours":
                    return now - datetime.timedelta(hours=quantity)
                if unit == "days":
                    return now - datetime.timedelta(days=quantity)
                if unit == "weeks":
                    return now - datetime.timedelta(weeks=quantity)
                if unit == "months":
                    return now - datetime.timedelta(days=quantity * 30)
                if unit == "years":
                    return now - datetime.timedelta(days=quantity * 365)

            normalized_candidate = re.sub(
                r"([A-Za-z]+)\.(\s+\d{1,2},\s+\d{4})", r"\1\2", candidate
            )
            for date_match in re.finditer(
                r"\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
                normalized_candidate,
                re.IGNORECASE,
            ):
                raw_date = (
                    date_match.group(0).replace("Sept ", "Sep ").replace("Sept.", "Sep")
                )
                for pattern in absolute_patterns:
                    try:
                        return datetime.datetime.strptime(raw_date, pattern).replace(
                            tzinfo=datetime.timezone.utc
                        )
                    except ValueError:
                        continue

        return None

    def _freshness_score(self, query: str, result: Dict) -> int:
        """Reward newer results for time-sensitive queries, but don't dominate relevance."""
        if not self._is_time_sensitive_query(query):
            return 0

        extracted_date = self._extract_result_date(result)
        if not extracted_date:
            return 0

        now = datetime.datetime.now(datetime.timezone.utc)
        age = now - extracted_date.astimezone(datetime.timezone.utc)
        age_days = max(age.total_seconds() / 86400, 0)

        if age_days <= 1:
            return 8
        if age_days <= 7:
            return 6
        if age_days <= 30:
            return 4
        if age_days <= 180:
            return 2
        if age_days <= 365:
            return 1
        return -3

    def _score_search_result(self, query: str, result: Dict) -> int:
        """Simple relevance scoring to suppress obviously off-topic search results."""
        normalized_query = self._normalize_web_search_query(query)
        terms = self._extract_search_terms(query)
        title = str(result.get("title", "")).lower()
        body = str(result.get("body", "")).lower()
        href = str(result.get("href", "")).lower()
        combined = f"{title} {body} {href}"

        score = 0
        for term in terms:
            if term in title:
                score += 4
            elif term in body:
                score += 2
            elif term in href:
                score += 1

        official_domains = (
            ".gov",
            ".mil",
            "whitehouse.gov",
            "defense.gov",
            "state.gov",
            "senate.gov",
            "house.gov",
        )
        if any(domain in href for domain in official_domains):
            score += 6

        if title and all(term not in combined for term in terms[: min(2, len(terms))]):
            score -= 5

        if "united states" in normalized_query and not any(
            marker in combined
            for marker in (
                "united states",
                "u.s.",
                "u.s ",
                "america",
                "american",
                "washington",
            )
        ):
            score -= 8

        score += self._freshness_score(query, result)

        return score

    def _preferred_official_domains_for_query(self, query: str) -> tuple[str, ...]:
        normalized = self._normalize_web_search_query(query)

        if "united states" in normalized and any(
            term in normalized
            for term in ("secretary of defense", "defense secretary", "defense")
        ):
            return (
                "defense.gov",
                "whitehouse.gov",
                "congress.gov",
                "senate.gov",
                "house.gov",
            )

        if "united states" in normalized and "president" in normalized:
            return (
                "whitehouse.gov",
                "congress.gov",
                "senate.gov",
                "house.gov",
            )

        return ()

    def _select_search_results(
        self,
        original_query: str,
        candidates: List[str],
        candidate_results: Dict[str, List[Dict]],
    ) -> tuple[str, List[Dict]]:
        """Pick the best candidate query and rank its results by relevance."""
        best_query = original_query
        best_results: List[Dict] = []
        best_score = -(10**9)

        for candidate in candidates:
            raw_results = candidate_results.get(candidate, [])
            ranked_with_scores = sorted(
                (
                    (self._score_search_result(candidate, item), item)
                    for item in raw_results
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            ranked = [item for score, item in ranked_with_scores if score > 0]
            if not ranked:
                ranked = [item for _, item in ranked_with_scores[:1]]
            top_score = ranked_with_scores[0][0] if ranked_with_scores else -(10**9)
            if top_score > best_score:
                best_query = candidate
                best_results = ranked
                best_score = top_score

        return best_query, best_results[:8]

    def _filter_grounding_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """For current office-holder queries, keep only authoritative official results when available."""
        if not results:
            return []

        role_lookup = self._extract_current_role_lookup(query)
        preferred_domains = self._preferred_official_domains_for_query(query)
        if role_lookup and preferred_domains:
            official_results = [
                result
                for result in results
                if any(
                    domain in str(result.get("href", "")).lower()
                    for domain in preferred_domains
                )
            ]
            if official_results:
                return official_results[:4]

        return results[:8]

    def _extract_relevant_text_window(
        self, text: str, query: str, max_chars: int = 1600
    ) -> str:
        """Extract a compact passage centered around the most relevant search terms."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""

        terms = self._extract_search_terms(query)[:6]
        best_index = -1
        for term in terms:
            idx = cleaned.lower().find(term.lower())
            if idx != -1 and (best_index == -1 or idx < best_index):
                best_index = idx

        if best_index == -1:
            return cleaned[:max_chars]

        half_window = max_chars // 2
        start = max(best_index - half_window, 0)
        end = min(start + max_chars, len(cleaned))
        excerpt = cleaned[start:end]
        if start > 0:
            excerpt = f"...{excerpt}"
        if end < len(cleaned):
            excerpt = f"{excerpt}..."
        return excerpt

    async def _fetch_search_result_excerpt(self, url: str, query: str) -> Optional[str]:
        """Fetch a result page and extract a short relevant passage for the model."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning(f"Failed to fetch search result page '{url}': {exc}")
            return None

        html_text = resp.text
        html_text = re.sub(
            r"<(script|style|header|footer|nav|noscript).*?>.*?</\1>",
            " ",
            html_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        html_text = re.sub(r"<[^>]+>", " ", html_text)
        html_text = html.unescape(html_text)
        excerpt = self._extract_relevant_text_window(html_text, query)
        return excerpt or None

    async def _build_search_result_enrichment(
        self, query: str, results: List[Dict]
    ) -> str:
        """Fetch a couple of authoritative pages so the model sees live page text, not only snippets."""
        if not results:
            return ""

        preferred_domains = self._preferred_official_domains_for_query(query)
        official_candidates: List[Dict] = []
        fallback_candidates: List[Dict] = []
        for result in results:
            href = str(result.get("href", "")).lower()
            if preferred_domains:
                is_official = any(domain in href for domain in preferred_domains)
            else:
                is_official = any(
                    domain in href
                    for domain in (
                        ".gov",
                        ".mil",
                        "defense.gov",
                        "whitehouse.gov",
                        "state.gov",
                    )
                )

            if is_official:
                official_candidates.append(result)
            elif self._score_search_result(query, result) > 0:
                fallback_candidates.append(result)

        selected = official_candidates[:2]
        if not preferred_domains and len(selected) < 2:
            selected.extend(fallback_candidates[: 2 - len(selected)])

        excerpts: List[str] = []
        for result in selected:
            href = str(result.get("href", "")).strip()
            if not href:
                continue

            excerpt = await self._fetch_search_result_excerpt(href, query)
            if not excerpt:
                continue

            title = str(result.get("title", href)).strip() or href
            excerpts.append(f"- {title}\n  {href}\n  {excerpt}")

        if not excerpts:
            return ""

        return "Fetched page excerpts:\n" + "\n\n".join(excerpts)

    async def _run_web_search(self, query: str) -> str:
        """Execute a web search and return formatted results for the model."""
        candidates = await self._expand_web_search_candidates_with_model(
            query,
            self._build_web_search_candidates(query),
        )

        def _do_search():
            time_sensitive_keywords = {
                "current",
                "latest",
                "news",
                "today",
                "now",
                "recent",
                "update",
                "price",
                "value",
                "cost",
                "score",
                "winner",
                "result",
                "finance",
                "stock",
            }
            is_fresh = any(k in query.lower() for k in time_sensitive_keywords)
            time_limit = "w" if is_fresh else None
            candidate_results: Dict[str, List[Dict]] = {}

            def _search_with_preferred_backends(
                ddgs: DDGS, candidate: str
            ) -> List[Dict]:
                search_attempts = [
                    (
                        "_text_html",
                        {"region": "us-en", "timelimit": time_limit, "max_results": 8},
                    ),
                    (
                        "_text_lite",
                        {"region": "us-en", "timelimit": time_limit, "max_results": 8},
                    ),
                    (
                        "_text_bing",
                        {"region": "us-en", "timelimit": time_limit, "max_results": 8},
                    ),
                ]

                for method_name, kwargs in search_attempts:
                    method = getattr(ddgs, method_name, None)
                    if not callable(method):
                        continue
                    try:
                        results = list(method(candidate, **kwargs))
                    except Exception as exc:
                        logger.warning(
                            f"Search backend {method_name} failed for '{candidate}': {exc}"
                        )
                        continue
                    if results:
                        return results
                return []

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with DDGS(timeout=20) as ddgs:
                    for candidate in candidates[:3]:
                        candidate_results[candidate] = _search_with_preferred_backends(
                            ddgs, candidate
                        )

            best_query, best_results = self._select_search_results(
                query, candidates, candidate_results
            )
            return best_query, best_results

        try:
            effective_query, results = await asyncio.to_thread(_do_search)
            if not results:
                return f"No direct web results found for '{query}'."

            results = self._filter_grounding_results(query, results)
            enrichment = await self._build_search_result_enrichment(query, results)

            res_str = f"Search results for '{query}'"
            if effective_query.strip().lower() != query.strip().lower():
                res_str += f" (best query: '{effective_query}')"
            res_str += ":\n\n"
            for i, r in enumerate(results):
                res_str += (
                    f"{i+1}. **{r['title']}**\n   {r['href']}\n   {r['body']}\n\n"
                )
            if enrichment:
                res_str += f"{enrichment}\n"
            return res_str
        except Exception as e:
            logger.error(f"Search tool failed: {e}")
            return "Search failed due to a temporary search backend issue."

    async def _complete_with_web_search(
        self,
        client: httpx.AsyncClient,
        target_model: str,
        messages: List[Dict],
        user_input: str,
        search_query: str,
    ) -> str:
        """Run web search in middleware, then ask the model to answer from results."""
        search_results = await self._run_web_search(search_query)
        role_lookup = self._extract_current_role_lookup(
            user_input
        ) or self._extract_current_role_lookup(search_query)
        role_guidance = ""
        if role_lookup:
            role, entity = role_lookup
            role_guidance = (
                f"\nThis is a current office-holder query about the {role} of {entity}. "
                "Return one grounded present-tense answer. "
                "If the results indicate a modern official title, use that title even if the user used an outdated alias."
            )

        handoff_messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "system",
                "content": (
                    "[SYSTEM: WEB SEARCH HANDOFF]\n"
                    "The middleware already executed a web search for you. "
                    "Answer the user's request using ONLY the provided web results and fetched page excerpts below. "
                    "Treat those results as the only source of truth for current facts. "
                    "Ignore any prior assistant statements, latent memory, or world knowledge that is not explicitly supported by the provided results. "
                    "Do not merge different officeholders, dates, or titles unless the provided results explicitly describe a transition. "
                    "Prefer fetched page excerpts from official domains over snippets when they conflict. "
                    "If the provided results do not establish an answer, say the results are insufficient briefly."
                    f"{role_guidance}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original user request: {user_input}\n\n"
                    f"Web search query: {search_query}\n\n"
                    "Use only the information below.\n\n"
                    f"{search_results}"
                ),
            },
        ]

        response = await client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": target_model,
                "messages": handoff_messages,
                "stream": False,
                "options": self._ollama_options(),
            },
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()

    def _resolve_requested_model(self, model_id: Optional[str]) -> str:
        """Apply server-side policy for user-selected chat models."""
        configured_model = model_id if model_id else self.active_model

        # The 1B model is only appropriate as an edge fallback on arm64 devices.
        # On desktop-class installs it produces poor tool use and high hallucination rates,
        # so ignore manual/UI requests and fall back to the normal active model.
        if (
            configured_model == settings.LIGHT_MODEL
            and settings.HARDWARE_TARGET != "arm64"
        ):
            logger.warning(
                "Requested light chat model '%s' on %s hardware; falling back to '%s'.",
                configured_model,
                settings.HARDWARE_TARGET,
                self.active_model,
            )
            return self.active_model

        return configured_model

    def _extract_local_file_path(self, user_input: str) -> Optional[str]:
        """Extract an explicit local file path from user input, including paths with spaces."""
        if not user_input:
            return None

        ext_pattern = "|".join(re.escape(ext) for ext in self.FILE_HANDOFF_EXTENSIONS)
        quoted_match = re.search(
            rf'["\'](?P<path>/[^"\']+?\.(?:{ext_pattern}))["\']',
            user_input,
            re.IGNORECASE,
        )
        if quoted_match:
            return quoted_match.group("path").strip()

        unquoted_match = re.search(
            rf"(?P<path>/[^\n]+?\.(?:{ext_pattern}))(?=(?:\s|$|[.,;:!?]))",
            user_input,
            re.IGNORECASE,
        )
        if unquoted_match:
            return unquoted_match.group("path").strip()

        return None

    def _should_handoff_local_file_read(
        self, user_input: str, intent: Optional[str]
    ) -> bool:
        """Detect explicit local file requests that should bypass model tool indecision."""
        if intent != "file_operation":
            return False

        normalized = re.sub(r"\s+", " ", user_input.lower()).strip()
        path = self._extract_local_file_path(user_input)
        if not path:
            return False

        request_markers = (
            "read ",
            "open ",
            "summarize",
            "summarise",
            "analyze",
            "analyse",
            "explain",
            "review",
            "extract",
            "what is in",
            "what's in",
        )
        return any(marker in normalized for marker in request_markers)

    def _preferred_file_read_limit(self, user_input: str, path: str) -> int:
        """Tune file extraction size so detailed summaries get enough context without overloading the prompt."""
        normalized = re.sub(r"\s+", " ", user_input.lower()).strip()
        limit = settings.MAX_ATTACHMENT_TEXT_CHARS

        if any(
            term in normalized
            for term in (
                "summarize",
                "summarise",
                "explain",
                "analyze",
                "analyse",
                "review",
            )
        ):
            limit = 18000

        if any(
            term in normalized
            for term in ("great detail", "detailed", "deep", "comprehensive")
        ):
            limit = 22000

        if path.lower().endswith(".pdf"):
            limit = max(limit, 20000)

        return min(limit, 24000)

    async def _complete_with_file_read(
        self,
        client: httpx.AsyncClient,
        target_model: str,
        messages: List[Dict],
        user_input: str,
        file_path: str,
    ) -> str:
        """Read a requested local file in middleware, then have the model answer from extracted content."""
        char_limit = self._preferred_file_read_limit(user_input, file_path)
        file_text = await file_service.read_host_file(file_path, max_chars=char_limit)

        if not file_text:
            return (
                f"I couldn't read `{file_path}` because it wasn't found or was empty."
            )

        lowered = file_text.lower()
        if lowered.startswith("error reading file:") or lowered.startswith(
            "failed to extract text from pdf:"
        ):
            return file_text

        handoff_messages = messages + [
            {
                "role": "system",
                "content": (
                    "[SYSTEM: LOCAL FILE HANDOFF]\n"
                    "The middleware already read the user's local file for you. "
                    "Answer the user's original request directly using the extracted file content below. "
                    "Do not mention internal routing, middleware, or tool calls. "
                    f"You are seeing up to {char_limit} characters of extracted content."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original user request: {user_input}\n\n"
                    f"File path: {file_path}\n\n"
                    "Extracted file content:\n\n"
                    f"{file_text}"
                ),
            },
        ]

        response = await client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": target_model,
                "messages": handoff_messages,
                "stream": False,
                "options": self._ollama_options(),
            },
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()

    async def think(
        self,
        user_input: str,
        conversation_id: str,
        user_id: int,
        model_id: Optional[str] = None,
        reasoning_mode: Optional[str] = None,
        imagine_model: Optional[str] = None,
        attachments: Optional[List[dict]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """Processes user input, handles tool calls, and returns a response.

        Key design: Each conversation is isolated. Vector memories from OTHER
        conversations are only used as very light background context, never as
        instructions to act upon.
        """
        await self._emit_progress(
            progress_callback,
            "status",
            label="Saving your message",
            detail="Persisting the user turn before building context.",
        )
        # 1. Save current user message to DB first so it's part of context
        user_msg = await memory_service.add_message(
            conversation_id, "user", user_input, attachments=attachments
        )

        # 2. Retrieve updated context
        await self._emit_progress(
            progress_callback,
            "status",
            label="Loading conversation context",
            detail="Collecting the recent messages and memory state.",
        )
        history = await memory_service.get_messages(conversation_id, user_id)

        # Determine reasoning instructions
        mode_instruction = ""
        if reasoning_mode == "deep":
            mode_instruction = (
                "\n\n[SYSTEM: DEEP THINK MODE]\n"
                "You are in Deep Think mode. Provide the absolute best match, use 100% of your reasoning ability, "
                "and give comprehensive details about the subject. Explore all angles."
            )
        elif reasoning_mode == "precise":
            mode_instruction = (
                "\n\n[SYSTEM: PRECISE MODE]\n"
                "You are in Precise mode. Provide a short, concise answer with 100% precision. "
                "Do not waffle. Be direct."
            )

        # Build the base system prompt before adding recovered conversation context.
        now = datetime.datetime.now()
        time_context = f"\n\n[SYSTEM: REAL-TIME CONTEXT]\n- Current Date/Time: {now.strftime('%A, %B %d, %Y %I:%M %p')}\n- Location: Host Machine (Bipod Space)"
        current_system_prompt = self.system_prompt + time_context + mode_instruction

        # 3. Build bounded conversation context
        await self._emit_progress(
            progress_callback,
            "status",
            label="Preparing context",
            detail="Compressing older turns and assembling the prompt.",
        )
        context_bundle = await self.context_builder.build(
            history=history,
            user_input=user_input,
            user_id=user_id,
            conversation_id=conversation_id,
            attachments=attachments,
            base_system_prompt=current_system_prompt,
        )
        current_system_prompt = context_bundle.system_prompt

        configured_model = self._resolve_requested_model(model_id)
        target_model = configured_model
        active_imagine_model = imagine_model or settings.ACTIVE_IMAGINE_MODEL

        # Determine if this is a generation request (to avoid switching to vision model which can't call tools)
        lower_input = user_input.lower()
        is_generation_request = self._is_image_generation_request(user_input)

        # Only force vision model for specialized analysis, not for generation/search
        if context_bundle.thread_has_images and not is_generation_request:
            # If the user is asking to "see", "describe" or "analyze", we NEED the vision model
            vision_trigger = {"describe", "see", "what", "analyze", "explain", "look"}
            if any(v in lower_input for v in vision_trigger):
                target_model = settings.VISION_MODEL
                logger.info(
                    "Vision task detected — switching brain to specialized eyes."
                )
                await self._emit_progress(
                    progress_callback,
                    "status",
                    label="Switching to the vision model",
                    detail="The request looks like image analysis rather than plain chat.",
                )

        if self._is_model_status_query(user_input):
            await self._emit_progress(
                progress_callback,
                "status",
                label="Checking active models",
                detail="Returning the currently selected chat and image models.",
            )
            ai_message = self._build_model_status_response(
                configured_model=configured_model,
                effective_model=target_model,
                imagine_model=active_imagine_model,
            )
            await self._emit_progress(
                progress_callback,
                "status",
                label="Saving the reply",
                detail="Persisting the assistant response.",
            )
            await self._store_assistant_turn(
                conversation_id, user_id, user_msg.id, user_input, ai_message
            )
            return ai_message

        if imagine_model:
            current_system_prompt += f"\n[USER PREFERENCE]: When generating images, you MUST use the '{imagine_model}' model via the `generate_image` tool."

        # Strengthen the core image generation directive right before the history
        if is_generation_request:
            current_system_prompt += (
                "\n\n[SYSTEM DIRECTIVE: PHOTO-REALISTIC IMAGE GENERATION]\n"
                "You are producing a HIGHLY REALISTIC photograph. You MUST expand the user's prompt into a technical description. "
                "Include camera specs (e.g., 'f/1.8, 85mm, ISO 100') and detailed lighting descriptions. "
                "ALWAYS include these quality tokens: 'raw photo, masterpiece, 8k uhd, photorealistic, cinematic lighting, "
                "highly detailed, soft lighting, sharp focus'. "
                "DO NOT use digital-art words like '3d render' or 'illustration'. "
                "Just call the 'generate_image' tool with your expanded, technical photographic prompt."
            )

        # Inject context about uploaded images (for Img2Img)
        if context_bundle.current_image_paths:
            img_instructions = (
                "\n\n[SYSTEM: IMAGE ATTACHED]\n"
                "The user has attached the following image(s). You can use 'generate_image' with 'image_path' to modify them:\n"
            )
            for path in context_bundle.current_image_paths:
                img_instructions += f"- {path}\n"
            current_system_prompt += img_instructions

        messages = [
            {"role": "system", "content": current_system_prompt}
        ] + context_bundle.recent_messages

        # 5. Request routing
        await self._emit_progress(
            progress_callback,
            "status",
            label="Routing the request",
            detail="Deciding between direct chat, tools, and handoffs.",
        )
        routing_decision = await self.router.route(user_input)
        intent = routing_decision.intent
        filtered_tools = self.router.filter_tools(self.tools, routing_decision)
        include_tools = len(filtered_tools) > 0
        file_handoff_path = (
            self._extract_local_file_path(user_input)
            if self._should_handoff_local_file_read(user_input, intent)
            else None
        )

        if include_tools:
            logger.info(
                "Routing decision: mode='%s' reason='%s' intent='%s' tools=%s",
                routing_decision.mode,
                routing_decision.reason,
                intent,
                [t["function"]["name"] for t in filtered_tools],
            )
        else:
            logger.info(
                "Routing decision: mode='%s' reason='%s' — processing as pure chat.",
                routing_decision.mode,
                routing_decision.reason,
            )

        try:
            request_timeout = self._ollama_request_timeout(include_tools)
            logger.info(
                "Starting Ollama chat request with read timeout %.1fs (tools=%s, model=%s)",
                request_timeout.read,
                include_tools,
                target_model,
            )
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                if file_handoff_path:
                    logger.info(
                        f"Using local file handoff for explicit path request: {file_handoff_path}"
                    )
                    await self._emit_progress(
                        progress_callback,
                        "status",
                        label="Reading the requested file",
                        detail=file_handoff_path,
                    )
                    ai_message = await self._complete_with_file_read(
                        client=client,
                        target_model=target_model,
                        messages=messages,
                        user_input=user_input,
                        file_path=file_handoff_path,
                    )
                    await self._emit_progress(
                        progress_callback,
                        "status",
                        label="Saving the reply",
                        detail="Persisting the assistant response.",
                    )
                    await self._store_assistant_turn(
                        conversation_id, user_id, user_msg.id, user_input, ai_message
                    )
                    return ai_message

                orchestration = await self.tool_orchestrator.run(
                    client=client,
                    target_model=target_model,
                    messages=messages,
                    filtered_tools=filtered_tools,
                    include_tools=include_tools,
                    intent=intent,
                    user_input=user_input,
                    imagine_model=imagine_model,
                    configured_model=configured_model,
                    active_imagine_model=active_imagine_model,
                    progress_callback=progress_callback,
                )

                if intent == "image_generation":
                    ai_message = await self._resolve_image_generation_response(
                        orchestration=orchestration,
                        user_input=user_input,
                        imagine_model=active_imagine_model,
                        current_image_paths=context_bundle.current_image_paths,
                        progress_callback=progress_callback,
                    )
                    await self._emit_progress(
                        progress_callback,
                        "status",
                        label="Saving the reply",
                        detail="Persisting the assistant response.",
                    )
                    await self._store_assistant_turn(
                        conversation_id, user_id, user_msg.id, user_input, ai_message
                    )
                    return ai_message

                final_answer = orchestration.final_answer
                search_handoff_query = self._extract_explicit_web_search_signal(
                    final_answer, user_input
                )
                if (
                    not search_handoff_query
                    and self._should_enforce_web_search_contract(intent, orchestration)
                ):
                    logger.info(
                        "Enforcing web search contract because the routed request returned without search results."
                    )
                    search_handoff_query = user_input
                elif not search_handoff_query:
                    search_handoff_query = self._extract_web_search_signal(
                        final_answer, user_input
                    )
                if search_handoff_query:
                    logger.info(
                        f"Model requested middleware web search handoff for query: {search_handoff_query}"
                    )
                    await self._emit_progress(
                        progress_callback,
                        "status",
                        label="Searching for current information",
                        detail=search_handoff_query,
                    )
                    final_answer = await self._complete_with_web_search(
                        client=client,
                        target_model=target_model,
                        messages=orchestration.messages,
                        user_input=user_input,
                        search_query=search_handoff_query,
                    )

                await self._emit_progress(
                    progress_callback,
                    "status",
                    label="Composing the final answer",
                    detail="Cleaning up tool output and assembling the reply.",
                )
                ai_message = answer_composer.compose(final_answer, orchestration)

                # Store AI message to DB
                await self._emit_progress(
                    progress_callback,
                    "status",
                    label="Saving the reply",
                    detail="Persisting the assistant response.",
                )
                await self._store_assistant_turn(
                    conversation_id, user_id, user_msg.id, user_input, ai_message
                )
                return ai_message

        except httpx.ReadTimeout:
            read_timeout = (
                settings.OLLAMA_TOOL_CHAT_TIMEOUT_SEC
                if include_tools
                else settings.OLLAMA_CHAT_TIMEOUT_SEC
            )
            logger.error(
                "Brain failure: Ollama timed out after %ss while handling model '%s' (tools=%s).",
                read_timeout,
                target_model,
                include_tools,
            )
            return (
                f"My thoughts are currently fragmented: the local model backend did not respond within {read_timeout} seconds. "
                "This usually means the model is still warming up, overloaded, or taking too long to produce the first token."
            )
        except httpx.RequestError as e:
            logger.error(f"Brain failure (Request Error): {type(e).__name__}: {e}")
            return (
                "My thoughts are currently fragmented: I could not reach the local model backend. "
                "Please check whether Ollama is still running and healthy."
            )
        except httpx.HTTPStatusError as e:
            response = e.response
            if response.status_code == 404:
                return f"I seem to be missing the required model '{target_model}'. Please install it by running:\n\n`docker exec -it bipod_ollama ollama pull {target_model}`"
            logger.error(f"Brain failure (HTTP Status Error): {e}")
            return f"My thoughts are currently fragmented: local model backend returned HTTP {response.status_code} {response.reason_phrase}."
        except Exception as e:
            logger.error(f"Brain failure ({type(e).__name__}): {e}")
            return f"My thoughts are currently fragmented: {str(e)}"

    async def _emit_progress(
        self,
        progress_callback: Optional[ProgressCallback],
        event: str,
        **payload: Any,
    ) -> None:
        if progress_callback is None:
            return
        await progress_callback(event, payload)

    async def _resolve_image_generation_response(
        self,
        orchestration,
        user_input: str,
        imagine_model: str,
        current_image_paths: List[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        if orchestration.image_generation_result:
            logger.info(
                "Using direct generate_image tool result for the final image reply."
            )
            return orchestration.image_generation_result.strip()

        logger.warning(
            "Image generation request completed without executing generate_image; enforcing direct imagine call."
        )
        await self._emit_progress(
            progress_callback,
            "tool_call",
            label="Generating an image",
            detail="The model skipped the image tool, so Bipod is invoking the imagine service directly.",
            tool_name="generate_image",
        )
        image_path = current_image_paths[0] if current_image_paths else None
        result = await self._generate_image_request(
            user_input,
            imagine_model,
            image_path=image_path,
        )
        return result.strip()

    async def clear_memory(self, conversation_id: str):
        await memory_service.clear_conversation(conversation_id)
        await vector_service.delete_conversation_memories(conversation_id)

    async def _store_assistant_turn(
        self,
        conversation_id: str,
        user_id: int,
        user_message_id: int,
        user_input: str,
        assistant_response: str,
    ) -> None:
        """Persist the assistant reply and the user turn for long-term memory."""
        await memory_service.add_message(
            conversation_id, "assistant", assistant_response
        )
        await vector_service.add_memory(
            user_input, user_id, user_message_id, conversation_id
        )

    def _is_model_status_query(self, user_input: str) -> bool:
        """Detect questions asking which model Bipod is currently using."""
        normalized = re.sub(r"\s+", " ", user_input.lower()).strip()
        direct_phrases = (
            "what model",
            "which model",
            "current model",
            "active model",
            "what llm",
            "which llm",
            "brain model",
            "chat model",
        )
        status_verbs = ("using", "running", "powered", "active", "current")
        return any(phrase in normalized for phrase in direct_phrases) or (
            "model" in normalized and any(verb in normalized for verb in status_verbs)
        )

    def _build_model_status_response(
        self,
        configured_model: str,
        effective_model: str,
        imagine_model: str,
    ) -> str:
        """Build a direct, accurate answer for model-identity questions."""
        if effective_model != configured_model:
            return (
                f"For this request, I'm using `{effective_model}`. "
                f"Your selected general chat model is `{configured_model}`. "
                f"Image generation uses `{imagine_model}`, and image analysis uses `{settings.VISION_MODEL}`."
            )

        return (
            f"I'm currently using `{configured_model}` for this chat. "
            f"Image generation uses `{imagine_model}`, and image analysis uses `{settings.VISION_MODEL}`."
        )

    async def _unload_ollama(self):
        """Tells Ollama to unload all models from VRAM to make room for Image Gen."""
        try:
            logger.info("Requesting Ollama to free all VRAM...")
            async with httpx.AsyncClient(timeout=10.0) as u_client:
                # 1. Query Ollama to see what is actually loaded
                try:
                    ps_resp = await u_client.get(f"{self.base_url}/api/ps")
                    if ps_resp.status_code == 200:
                        running = ps_resp.json().get("models", [])
                        for model in running:
                            name = model.get("name")
                            logger.info(f"Unloading model '{name}'...")
                            await u_client.post(
                                f"{self.base_url}/api/generate",
                                json={"model": name, "keep_alive": 0},
                            )
                        if not running:
                            logger.info("Ollama reports no models currently in VRAM.")
                    else:
                        raise Exception(f"PS endpoint returned {ps_resp.status_code}")
                except Exception as ps_e:
                    logger.warning(
                        f"Could not query running models ({ps_e}), falling back to default list."
                    )
                    # Fallback: Unload the usual suspects
                    for m in [
                        self.active_model,
                        settings.VISION_MODEL,
                        settings.EMBEDDING_MODEL,
                    ]:
                        await u_client.post(
                            f"{self.base_url}/api/generate",
                            json={"model": m, "keep_alive": 0},
                        )

                # Small delay to allow Ollama process to actually release handles
                await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to unload Ollama models: {e}")

    async def _vision_request(self, image_b64: str, prompt: str) -> str:
        """Handles vision requests using a specialized Moondream or Llama Vision model."""
        try:
            async with httpx.AsyncClient(timeout=90.0) as v_client:
                payload = {
                    "model": settings.VISION_MODEL,
                    "messages": [
                        {"role": "user", "content": prompt, "images": [image_b64]}
                    ],
                    "stream": False,
                }
                response = await v_client.post(
                    f"{self.base_url}/api/chat", json=payload
                )
                response.raise_for_status()
                return (
                    response.json()
                    .get("message", {})
                    .get(
                        "content",
                        "I saw the image but couldn't think of anything to say.",
                    )
                )
        except Exception as e:
            logger.error(f"Vision tool failure: {e}")
            return f"Error analyzing image: {str(e)}"

    async def _generate_image_request(
        self,
        prompt: str,
        model_type: str = "sdxl-lightning",
        image_path: Optional[str] = None,
    ) -> str:
        """Internal helper to call the Imagine service."""
        try:
            # 1. First, tell Ollama to get out of the GPU
            await self._unload_ollama()

            logger.info(
                f"Requesting image generation: '{prompt}' via {model_type} (Img2Img: {bool(image_path)})"
            )
            # Increase timeout to 10 minutes: first-run model download is ~4GB
            async with httpx.AsyncClient(timeout=600.0) as client:
                payload = {
                    "prompt": prompt,
                    "model_type": model_type,
                    "steps": 40,  # Upgraded quality
                }

                # If image_path is provided, read and encode it for Img2Img
                if image_path:
                    try:
                        with open(image_path, "rb") as f:
                            encoded_img = base64.b64encode(f.read()).decode("utf-8")
                        payload["image"] = encoded_img
                        logger.info(
                            f"Attaching image from {image_path} for generation."
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to read image for Img2Img: {e}. Falling back to Text2Img."
                        )

                response = await client.post(
                    f"{settings.IMAGINE_API_URL}/generate", json=payload
                )
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "success":
                    # Decode and save to file
                    img_data = base64.b64decode(data["image_base64"])
                    filename = f"generated_{uuid.uuid4().hex[:8]}.jpg"

                    # Save to Bipod's generated directory
                    # We use file_service logic manually here to access the path
                    filepath = os.path.join(settings.GENERATED_DIR, filename)
                    with open(filepath, "wb") as f:
                        f.write(img_data)

                    logger.info(f"Generated image saved to {filepath}")
                    return f"Image generated successfully! Saved to: {filepath}\n\n![Generated Image](/generated/{filename})"
                else:
                    return f"Generation failed: {data}"

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response else "unknown"
            logger.error(f"Image generation HTTP failure: {e}")
            return (
                f"Image generation failed because the local image service returned HTTP {status_code}. "
                "It may still be starting up or downloading model weights."
            )
        except httpx.ReadTimeout:
            logger.error("Image generation timed out.")
            return "Generation timed out. Bipod is likely downloading the model weights for the first time (approx 4GB). Please wait a few minutes and try again — the download will continue in the background."
        except Exception as e:
            logger.error(f"Image generation failure: {e}")
            return (
                "Failed to generate image because the local image service is unavailable or still starting. "
                "If this is the first run, it may still be downloading model weights."
            )

    async def improve_studio_prompts(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_type: Optional[str] = None,
    ) -> dict[str, str]:
        system_prompt = (
            "You improve prompts for local image generation UIs.\n"
            "Return only valid JSON with keys 'prompt' and 'negative_prompt'.\n"
            "Preserve the user's core subject and intent.\n"
            "Make the positive prompt clearer, denser, and better structured for image generation.\n"
            "Make the negative prompt concise and practical, focused on artifact prevention.\n"
            "Do not add explanations, markdown, or code fences.\n"
            "Do not make the prompts excessively long."
        )

        user_prompt = (
            f"Model type: {model_type or 'unspecified'}\n"
            f"Current prompt: {prompt}\n"
            f"Current negative prompt: {negative_prompt or '(empty)'}"
        )

        async with httpx.AsyncClient(
            timeout=self._ollama_request_timeout(include_tools=False)
        ) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": settings.ACTIVE_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": self._ollama_options(),
                },
            )
            response.raise_for_status()

        content = response.json().get("message", {}).get("content", "").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}

        improved_prompt = str(parsed.get("prompt") or prompt).strip()
        improved_negative = str(
            parsed.get("negative_prompt") or negative_prompt or ""
        ).strip()
        return {
            "prompt": improved_prompt,
            "negative_prompt": improved_negative,
        }

    async def _map_reduce_summarize(self, text: str) -> str:
        """Summarizes large text chunks using a Map-Reduce approach."""
        chunk_size = 25000
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        if len(chunks) == 1:
            # Just a simple summary for one chunk
            return await self._summarize_chunk(chunks[0])

        logger.info(f"Summarizing {len(chunks)} chunks via Map-Reduce...")
        summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Mapping chunk {i+1}/{len(chunks)}...")
            summary = await self._summarize_chunk(chunk)
            summaries.append(summary)

        final_text = "\n\n".join(summaries)
        return await self._summarize_chunk(final_text, is_final=True)

    async def _summarize_chunk(self, text: str, is_final: bool = False) -> str:
        """Calls the LLM to summarize a specific chunk of text."""
        try:
            prompt = (
                "Summarize the following text. Focus on technical details, structure, and key logic flows. "
                "Keep it concise but detailed enough for a developer to understand the core functionality. "
                "Maintain specific variable names or function signatures if they are important."
            )
            if is_final:
                prompt = "Synthesize the following summaries into a final, coherent overview. Preserve all technical specifics."

            async with httpx.AsyncClient(
                timeout=self._ollama_request_timeout(include_tools=False)
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": settings.ACTIVE_MODEL,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": text},
                        ],
                        "stream": False,
                        "options": self._ollama_options(),
                    },
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:1000] + "... [Summary failed]"

    def _check_for_hallucinated_tools(
        self, content: str, allowed_tools: set[str] | None = None
    ) -> tuple[List[Dict], str]:
        """Detects tool calls that the model output as plain text JSON instead of real tool_calls.
        Only strips blocks that are successfully identified as internal tool calls.
        """
        cleaned_content = content
        all_tool_names = {t["function"]["name"] for t in self.tools}
        valid_tools = allowed_tools if allowed_tools else all_tool_names

        tool_calls = []
        strip_ranges = []  # Track where internal content was found to strip it later

        # 1. Strip specific internal model tags that should NEVER be seen
        internal_markers = ["<|python_tag|>", "<|action_tag|>", "<|thought|>"]
        for marker in internal_markers:
            for m in re.finditer(re.escape(marker), cleaned_content):
                strip_ranges.append((m.start(), m.end()))

        # 2. Extract JSON blocks using brace-counting
        start_idx = -1
        brace_count = 0

        for i, char in enumerate(cleaned_content):
            if char == "{":
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    potential_json = cleaned_content[start_idx : i + 1]

                    try:
                        data = json.loads(potential_json)
                        fn_name = data.get("name") or data.get("function", {}).get(
                            "name"
                        )
                        args = (
                            data.get("parameters")
                            or data.get("arguments")
                            or data.get("function", {}).get("arguments")
                        )

                        # Alias 'cmd' to 'command' for shell tool
                        if fn_name == "shell" and args and "cmd" in args:
                            args["command"] = args.pop("cmd")

                        # Strip if it LOOKS like a tool call (has both 'name' AND 'arguments'/'parameters')
                        # This catches real tools AND completely hallucinated ones (e.g. 'generate_story')
                        # but avoids stripping random JSON data that just has a 'name' key
                        is_tool_shaped = fn_name and (
                            args is not None
                            or "arguments" in data
                            or "parameters" in data
                        )
                        if is_tool_shaped:
                            # Also try to strip surrounding backticks if present
                            current_start = start_idx
                            current_end = i + 1

                            pre = cleaned_content[
                                max(0, current_start - 10) : current_start
                            ]
                            post = cleaned_content[
                                current_end : min(
                                    len(cleaned_content), current_end + 11
                                )
                            ]

                            if "```json" in pre:
                                current_start = cleaned_content.rfind(
                                    "```json", 0, current_start
                                )
                            elif "```" in pre:
                                current_start = cleaned_content.rfind(
                                    "```", 0, current_start
                                )

                            if "```" in post:
                                # Find the closing backticks after the JSON block
                                closing_backticks_idx = cleaned_content.find(
                                    "```", current_end
                                )
                                if closing_backticks_idx != -1:
                                    current_end = (
                                        closing_backticks_idx + 3
                                    )  # Include the '```'

                            # Aggressively strip surrounding characters that common models leak (stray braces, backticks, newlines)
                            while current_start > 0 and cleaned_content[
                                current_start - 1
                            ] in [" ", "\n", "\r", "}", "]", "`", ":", ","]:
                                current_start -= 1
                            while current_end < len(
                                cleaned_content
                            ) and cleaned_content[current_end] in [
                                " ",
                                "\n",
                                "\r",
                                "{",
                                "[",
                                "`",
                                ":",
                                ",",
                            ]:
                                current_end += 1

                            strip_ranges.append((current_start, current_end))

                            if fn_name in valid_tools:
                                tool_calls.append(
                                    {
                                        "id": f"call_{os.urandom(4).hex()}",
                                        "type": "function",
                                        "function": {
                                            "name": fn_name,
                                            "arguments": args,
                                        },
                                    }
                                )
                            elif fn_name in all_tool_names:
                                logger.warning(
                                    f"Blocked hallucinated call to '{fn_name}' — not in allowed tool set."
                                )
                            else:
                                logger.warning(
                                    f"Stripped completely fake tool call: '{fn_name}' — tool does not exist."
                                )
                    except:
                        # 2b. Attempt to recover malformed JSON for known tools (LLMs often output broken JSON)
                        # Regex to find "name": "something" pattern
                        name_match = re.search(r'"name"\s*:\s*"(\w+)"', potential_json)
                        if name_match:
                            found_name = name_match.group(1)

                            # SAFETY: Only strip if it looks like a tool call (has args key) OR is a known tool
                            # This prevents stripping `{"name": "John"}` in normal code
                            has_args_key = (
                                "arguments" in potential_json
                                or "parameters" in potential_json
                            )
                            is_known_tool = found_name in all_tool_names

                            if has_args_key or is_known_tool:
                                # Determine range to strip (same logic as above)
                                current_start = start_idx
                                current_end = i + 1
                                # Expand to surrounding backticks
                                pre = cleaned_content[
                                    max(0, current_start - 10) : current_start
                                ]
                                if "```json" in pre:
                                    current_start = cleaned_content.rfind(
                                        "```json", 0, current_start
                                    )
                                elif "```" in pre:
                                    current_start = cleaned_content.rfind(
                                        "```", 0, current_start
                                    )

                                post = cleaned_content[
                                    current_end : min(
                                        len(cleaned_content), current_end + 11
                                    )
                                ]
                                if "```" in post:
                                    cb_idx = cleaned_content.find("```", current_end)
                                    if cb_idx != -1:
                                        current_end = cb_idx + 3

                                # Aggressively strip surrounding characters that common models leak (stray braces, backticks, newlines)
                                while current_start > 0 and cleaned_content[
                                    current_start - 1
                                ] in [" ", "\n", "\r", "}", "]", "`", ":", ","]:
                                    current_start -= 1
                                while current_end < len(
                                    cleaned_content
                                ) and cleaned_content[current_end] in [
                                    " ",
                                    "\n",
                                    "\r",
                                    "{",
                                    "[",
                                    "`",
                                    ":",
                                    ",",
                                ]:
                                    current_end += 1

                                # ALWAYS strip the broken tool-like JSON
                                strip_ranges.append((current_start, current_end))

                                # RECOVERY: specialized fix for No-Arg tools like get_system_info
                                found_name_valid = found_name in valid_tools
                                if found_name == "get_system_info" and found_name_valid:
                                    logger.info(
                                        "Recovered malformed JSON for get_system_info"
                                    )
                                    tool_calls.append(
                                        {
                                            "id": f"call_{os.urandom(4).hex()}",
                                            "type": "function",
                                            "function": {
                                                "name": "get_system_info",
                                                "arguments": {},
                                            },
                                        }
                                    )
                                elif found_name_valid:
                                    # RECOVERY: specialized fix for shell commands with broken JSON
                                    if found_name == "execute_system_command":
                                        cmd_match = re.search(
                                            r'["\'](?:command|cmd)["\']\s*:\s*["\'](.*?)["\']',
                                            potential_json,
                                        )
                                        if cmd_match:
                                            logger.info(
                                                "Recovered malformed JSON for execute_system_command"
                                            )
                                            tool_calls.append(
                                                {
                                                    "id": f"call_{os.urandom(4).hex()}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "execute_system_command",
                                                        "arguments": {
                                                            "command": cmd_match.group(
                                                                1
                                                            )
                                                        },
                                                    },
                                                }
                                            )
                                            continue

                                    logger.warning(
                                        f"Detected malformed JSON for '{found_name}' — cannot execute safely."
                                    )
                                else:
                                    logger.warning(
                                        f"Stripped hallucinated/malformed tool: '{found_name}'"
                                    )

                    start_idx = -1

        # 3. Extract Function-like calls: tool_name("arg")
        func_pattern = r'(\w+)\((?:arguments=)?(\{.*?\}|"(.*?)")\)'
        for match in re.finditer(func_pattern, cleaned_content):
            fname = match.group(1)
            raw_args = match.group(2)
            str_arg = match.group(3)

            # Only strip if it matches a tool name
            if fname in all_tool_names:
                strip_ranges.append((match.start(), match.end()))

                if fname in valid_tools:
                    final_args = {}
                    if raw_args.startswith("{"):
                        try:
                            final_args = json.loads(raw_args)
                        except:
                            continue
                    elif str_arg:
                        if fname == "execute_system_command":
                            final_args = {"command": str_arg}
                        elif fname == "read_file":
                            final_args = {"path": str_arg}
                        elif fname == "search_files":
                            final_args = {"pattern": str_arg}

                    tool_calls.append(
                        {
                            "id": f"call_{os.urandom(4).hex()}",
                            "type": "function",
                            "function": {"name": fname, "arguments": final_args},
                        }
                    )

        # 4. Strip ONLY the identified internal blocks
        final_text = cleaned_content
        # Sort and merge overlapping ranges to avoid double-stripping issues
        if not strip_ranges:
            return tool_calls, final_text.strip()

        strip_ranges.sort(key=lambda x: x[0])
        merged = []
        if strip_ranges:
            curr_start, curr_end = strip_ranges[0]
            for next_start, next_end in strip_ranges[1:]:
                if next_start < curr_end:
                    curr_end = max(curr_end, next_end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))

        for start, end in reversed(merged):
            final_text = final_text[:start] + final_text[end:]

        return tool_calls, final_text.strip()


brain_service = BrainService()
