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
            "You are Bipod, an AI agent running entirely on the user's local machine. "
            "You prioritize privacy — no data ever leaves this device. "
            "You are helpful, concise, and intelligent. "
            "You can have natural conversations on any topic. "
            "You also have filesystem tools available. You can search, read, create, or update files. "
            "You can also ANALYZE IMAGES and screenshots at specific paths on the host system. "
            "ONLY use these tools when the user EXPLICITLY asks you to perform a file or image operation. "
            "For normal conversation or general questions, respond directly without tools."
        )

        # Keywords that trigger tool inclusion
        self.FILE_KEYWORDS = {
            "file", "files", "find", "search", "read", "open",
            "look", "list", "directory", "folder", "path",
            "document", "documents", "locate", "show me", "save", "create", "write",
            "image", "png", "jpg", "jpeg", "picture", "screenshot", "describe"
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
            },
            {
                "type": "function",
                "function": {
                    "name": "save_file",
                    "description": "Creates or overwrites a file on the host filesystem with the provided content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "The absolute or relative path where the file should be saved."},
                            "content": {"type": "string", "description": "The exact text content to write into the file."},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_image_file",
                    "description": "Reads an image file from the host filesystem and describes its content using the vision model.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "The absolute or relative path to the image file on the host."},
                            "prompt": {"type": "string", "description": "Optional specific question or prompt about the image."},
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
        user_id: int,
        model_id: Optional[str] = None, 
        reasoning_mode: Optional[str] = None,
        images: Optional[List[str]] = None
    ) -> str:
        """Processes user input, handles tool calls, and returns a response."""
        # 1. Save current user message to DB first so it's part of context
        await memory_service.add_message(conversation_id, "user", user_input, images=images)
        
        # 2. Retrieve updated context (last 15 messages)
        history = await memory_service.get_messages(conversation_id, user_id)
        
        # 3. Build the message context and check for any images in the whole thread
        formatted_history = []
        thread_has_images = False
        for m in history[-15:]:
            msg_dict = {"role": m.role, "content": m.content}
            if m.images:
                msg_dict["images"] = m.images
                thread_has_images = True
            formatted_history.append(msg_dict)

        # 4. Determine model to use (Force vision if any images exist in current thread)
        target_model = model_id if model_id else self.active_model
        if thread_has_images:
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
        # Inject mode instruction into system prompt for this turn
        current_system_prompt = self.system_prompt + mode_instruction
        messages = [{"role": "system", "content": current_system_prompt}] + formatted_history

        # 5. Include tools only if NOT in vision mode (to avoid vision model complexity)
        include_tools = False
        if not thread_has_images:
            words = set(user_input.lower().split())
            if any(k in words for k in self.FILE_KEYWORDS):
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
                        elif fn_name == "save_file":
                            success = await file_service.write_host_file(args.get("path"), args.get("content"))
                            result = f"File saved successfully to {args.get('path')}" if success else "Failed to save file. Check permissions."
                        elif fn_name == "analyze_image_file":
                            b64 = await file_service.read_host_image(args.get("path"))
                            if b64:
                                prompt = args.get("prompt", "Describe this image in detail.")
                                result = await self._vision_request(b64, prompt)
                            else:
                                result = "Error: Could not find or read the image file at that path."
                        
                        logger.info(f"Tool {fn_name} returned: {result[:100]}...")

                        # Add the tool result to messages
                        messages.append(message) # Add AI's tool call request
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.get("id")
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

    async def _vision_request(self, b64_image: str, prompt: str) -> str:
        """Internal helper to call the vision model (Moondream/Llava) directly."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": settings.VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [b64_image]
                        }
                    ],
                    "stream": False,
                }
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "I saw the image but couldn't think of a description.")
        except Exception as e:
            logger.error(f"Vision tool failure: {e}")
            return f"Error analyzing image: {str(e)}"

brain_service = BrainService()

