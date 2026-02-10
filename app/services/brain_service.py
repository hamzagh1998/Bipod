import json
import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logger import get_logger
from app.services.file_service import file_service

from app.services.memory_service import memory_service

logger = get_logger("bipod.brain")

class BrainService:
    """The central intelligence service of Bipod with tool-calling capabilities."""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.active_model = settings.ACTIVE_MODEL
        
        # System Prompt following Bipod Philosophy
        self.system_prompt = (
            "You are Bipod, a weightless AI companion running entirely on the user's local machine. "
            "You prioritize privacy — no data ever leaves this device. "
            "You are helpful, concise, and intelligent. "
            "You can have natural conversations on any topic. "
            "You also have filesystem tools available, but you must ONLY use them when the user "
            "EXPLICITLY asks you to find, search, open, or read a file. "
            "For normal conversation, questions, or explanations, just respond directly without using any tools. "
            "Never search for files or use tools unless the user's message clearly requests file operations."
        )

        # Keywords that trigger tool inclusion
        self.FILE_KEYWORDS = {
            "file", "files", "find", "search", "read", "open",
            "look", "list", "directory", "folder", "path",
            "document", "documents", "locate", "show me",
        }

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
                            "pattern": {"type": "string", "description": "The glob pattern to search for."},
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
                            "path": {"type": "string", "description": "The absolute or relative path to the file on the host."},
                        },
                        "required": ["path"],
                    },
                },
            }
        ]

    async def think(
        self, 
        user_input: str, 
        conversation_id: str, 
        model_id: Optional[str] = None, 
        reasoning_mode: Optional[str] = None,
        images: Optional[List[str]] = None
    ) -> str:
        """Processes user input, handles tool calls, and returns a response."""
        # Determine model to use
        target_model = model_id if model_id else self.active_model
        
        # If images are present, force vision model
        if images and len(images) > 0:
            target_model = settings.VISION_MODEL

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
        # Save user message to DB
        await memory_service.add_message(conversation_id, "user", user_input)
        
        # Retrieve context from DB
        history = await memory_service.get_messages(conversation_id)
        
        # Build the message context (last 15 messages)
        formatted_history = [{"role": m.role, "content": m.content} for m in history[-15:]]
        
        
        # Inject mode instruction into system prompt for this turn
        current_system_prompt = self.system_prompt + mode_instruction
        messages = [{"role": "system", "content": current_system_prompt}] + formatted_history

        # Inject images into the last user message if present
        if images and len(images) > 0 and messages[-1]["role"] == "user":
            messages[-1]["images"] = images

        # Only include tools when the user message contains file-related keywords
        # AND we are NOT in vision mode (files and vision might conflict or simplify logic)
        include_tools = False
        if not images:
            words = set(user_input.lower().split())
            if any(k in words for k in ["file", "read", "save", "create", "list", "search", "code", "script"]):
                include_tools = True
                logger.info("File-related keywords detected — tools enabled for this request.")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 1. Initial request (with or without tools)
                payload = {
                    "model": target_model,
                    "messages": messages,
                    "stream": False,
                }
                if include_tools:
                    payload["tools"] = self.tools

                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                message = data.get("message", {})

                # 2. Check for tool calls
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        fn_name = tool_call["function"]["name"]
                        args = tool_call["function"]["arguments"]
                        
                        logger.info(f"Bipod decided to use tool: {fn_name} with args {args}")
                        
                        result = ""
                        if fn_name == "search_files":
                            found = await file_service.search_host(args.get("pattern"))
                            result = f"Found files: {found}"
                        elif fn_name == "read_file":
                            content = await file_service.read_host_file(args.get("path"))
                            result = content if content else "File not found or empty."

                        # Add the tool result to messages
                        messages.append(message) # Add AI's tool call request
                        messages.append({
                            "role": "tool",
                            "content": result,
                        })

                    # 3. Get final response from LLM after tool execution
                    final_response = await client.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": target_model,
                            "messages": messages,
                            "stream": False,
                        },
                    )
                    final_response.raise_for_status()
                    data = final_response.json()
                    ai_message = data["message"]["content"]
                else:
                    ai_message = message.get("content", "")

                # Store AI message to DB
                await memory_service.add_message(conversation_id, "assistant", ai_message)
                return ai_message

        except httpx.HTTPStatusError as e:
            response = e.response
            if response.status_code == 404:
                return f"I seem to be missing the required model '{target_model}'. Please install it by running:\n\n`docker exec -it bipod_ollama ollama pull {target_model}`"
            logger.error(f"Brain failure (HTTP Status Error): {e}")
            return f"My thoughts are currently fragmented: Client error '{response.status_code} {response.reason_phrase}' for url '{response.url}'"
        except Exception as e:
            logger.error(f"Brain failure: {e}")
            return f"My thoughts are currently fragmented: {str(e)}"

    def clear_memory(self, conversation_id: str):
        # We don't really use this anymore as we store in DB, 
        # but we could implement clearing a specific conversation
        pass

brain_service = BrainService()

