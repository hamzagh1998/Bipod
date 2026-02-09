import json
import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logger import get_logger
from app.services.file_service import file_service

logger = get_logger("bipod.brain")

class BrainService:
    """The central intelligence service of Bipod with tool-calling capabilities."""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.active_model = settings.ACTIVE_MODEL
        self.history: List[Dict[str, Any]] = []
        
        # System Prompt following Bipod Philosophy
        self.system_prompt = (
            "You are Bipod, a weightless AI companion. "
            "You run locally and prioritize your user's privacy. "
            "You have access to the host filesystem via the provided tools. "
            "If a user asks to find or read a file, use the tools. "
            "Be concise, intelligent, and proactive."
        )

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

    async def think(self, user_input: str) -> str:
        """Processes user input, handles tool calls, and returns a response."""
        self.history.append({"role": "user", "content": user_input})
        
        # Build the message context
        messages = [{"role": "system", "content": self.system_prompt}] + self.history[-10:]

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 1. Initial request to see if LLM wants to use a tool
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.active_model,
                        "messages": messages,
                        "tools": self.tools,
                        "stream": False,
                    },
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
                            "model": self.active_model,
                            "messages": messages,
                            "stream": False,
                        },
                    )
                    final_response.raise_for_status()
                    data = final_response.json()
                    ai_message = data["message"]["content"]
                else:
                    ai_message = message.get("content", "")

                # Store history
                self.history.append({"role": "assistant", "content": ai_message})
                return ai_message

        except Exception as e:
            logger.error(f"Brain failure: {e}")
            return f"My thoughts are currently fragmented: {str(e)}"

    def clear_memory(self):
        self.history = []

brain_service = BrainService()
