import json
import os
import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logger import get_logger
from app.services.file_service import file_service

from app.services.memory_service import memory_service
from app.services.vector_service import vector_service

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
            "You also have filesystem tools available. You can search, read, create, or update TEXT files. "
            "You can execute shell commands for efficient searching (grep, find) or system tasks. "
            "You can ANALYZE and DESCRIBE existing images at specific paths using the vision model. "
            "\n\nCRITICAL RULES:\n"
            "1. ALWAYS PRIORITIZE the current user message. Execute exactly what the user asks.\n"
            "2. NEVER reference files, paths, or content from [RECOLLECTED HISTORICAL BACKGROUND] "
            "unless the user explicitly asks about past conversations or memories.\n"
            "3. If a user asks you to find a file, USE THE search_files TOOL or execute_system_command TOOL "
            "to actually search the filesystem. DO NOT guess or assume files based on memory.\n"
            "4. DO NOT say a task is 'too resource intensive' or that you 'cannot see' — just use the tools.\n"
            "5. If a file search or command is needed, do it IMMEDIATELY without waffling.\n"
            "6. When searching for files, use the search_files or execute_system_command tool "
            "with 'find' command. ALWAYS search before claiming a file does not exist.\n"
            "7. EACH CONVERSATION IS INDEPENDENT. Do not mix up content from different conversations.\n"
            "8. You CAN generate images using the `generate_image` tool. If asked to 'draw' or 'create' an image, use this tool. "
            "For 'fast' generation or if requested, use 'dalle-mini' model, otherwise default to 'stable-diffusion'.\n"
            "9. NEVER claim a file exists or was created unless a tool has confirmed it. "
            "Do NOT hallucinate file operations.\n"
            "10. ONLY create or save files on the host when the user EXPLICITLY asks you to. "
            "Always tell the user the actual path returned by the save_file tool."
        )

        # Keywords that trigger tool inclusion
        self.FILE_KEYWORDS = {
            "file", "files", "find", "search", "read", "open",
            "look", "list", "directory", "folder", "path",
            "document", "documents", "locate", "show me", "save", "create", "write",
            "image", "png", "jpg", "jpeg", "picture", "screenshot", "describe",
            "generate", "make", "move", "copy", "delete", "rename", "pdf"
        }

        # Define tools for Ollama (always available for file operations)
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
                            "root": {"type": "string", "description": "Optional directory to start search from (default: /)."},
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
                    "description": "Creates or overwrites a TEXT file on the host filesystem at the specified path. ONLY use this when the user explicitly asks to create, save, or write a file. This tool can only write TEXT content — it CANNOT create binary files like images, audio, or video.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "The absolute path on the host filesystem where the file should be saved (e.g. '/home/user/notes.txt')."},
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
                    "description": "Reads an EXISTING image file from the host filesystem and describes its content using the vision model (Moondream). This tool ONLY analyzes and describes existing images — it does NOT generate, create, or produce new images. If the image does not exist at the given path, it will return an error.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "The absolute path to an EXISTING image file on the host."},
                            "prompt": {"type": "string", "description": "Optional specific question or prompt about the image."},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generates a new image based on a text prompt using AI (Stable Diffusion). Returns the path to the generated image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Detailed description of the image to generate."},
                            "image_path": {"type": "string", "description": "Optional. Path to an existing image file (e.g. from user upload) to use as a starting point (Image-to-Image)."},
                            "model_type": {"type": "string", "enum": ["stable-diffusion", "dalle-mini"], "description": "Model to use. Default 'stable-diffusion'. Use 'dalle-mini' for faster/lower quality generation."},
                        },
                        "required": ["prompt"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_system_command",
                    "description": "Executes a shell command on the host system and returns the output. Use this for high-performance searching (grep, find), directory listings, or system diagnostics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The full shell command to execute."},
                        },
                        "required": ["command"],
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
        imagine_model: Optional[str] = None,
        attachments: Optional[List[dict]] = None
    ) -> str:
        """Processes user input, handles tool calls, and returns a response.
        
        Key design: Each conversation is isolated. Vector memories from OTHER
        conversations are only used as very light background context, never as
        instructions to act upon.
        """
        # 1. Save current user message to DB first so it's part of context
        user_msg = await memory_service.add_message(conversation_id, "user", user_input, attachments=attachments)
        
        # 2. Retrieve updated context (last 15 messages)
        history = await memory_service.get_messages(conversation_id, user_id)
        
        # 3. Build the message context and check for any images in the whole thread
        formatted_history = []
        thread_has_images = False
        pdf_texts = []

        current_image_paths = []
        # Process current turn's PDFs if any
        if attachments:
            from pypdf import PdfReader
            import io
            import base64
            for att in attachments:
                if att.get("type") == "pdf":
                    try:
                        pdf_bytes = base64.b64decode(att["content"])
                        reader = PdfReader(io.BytesIO(pdf_bytes))
                        text = f"\n--- ATTACHED DOCUMENT ({att.get('name', 'untitled')}) ---\n"
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        pdf_texts.append(text)
                        logger.info(f"Extracted {len(text)} chars from attached PDF: {att.get('name')}")
                    except Exception as e:
                        logger.error(f"Failed to extract text from uploaded PDF: {e}")
                elif att.get("type") == "image":
                    # Save image so it can be used for Img2Img generation
                    try:
                        img_bytes = base64.b64decode(att["content"])
                        filename = att.get("name", "upload.jpg")
                        # Sanitize filename
                        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        safe_filename = "".join(c for c in filename if c in valid_chars).replace(" ", "_")
                        if not safe_filename: safe_filename = "image_upload.jpg"
                        
                        file_path = os.path.join(settings.UPLOADS_DIR, safe_filename)
                        with open(file_path, "wb") as f:
                            f.write(img_bytes)
                        
                        current_image_paths.append(file_path)
                        logger.info(f"Saved attached image for processing: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to save image attachment: {e}")

        for m in history[-15:]:
            msg_dict = {"role": m.role, "content": m.content}
            if m.attachments:
                # Filter for images to pass to Ollama's vision capability
                images = [a["content"] for a in m.attachments if a.get("type") == "image"]
                if images:
                    msg_dict["images"] = images
                    thread_has_images = True
            formatted_history.append(msg_dict)

        # Inject PDF texts from THIS turn into the last user message if they exist
        if pdf_texts and formatted_history and formatted_history[-1]["role"] == "user":
            formatted_history[-1]["content"] += "\n" + "\n".join(pdf_texts)

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
        
        if imagine_model:
             current_system_prompt += f"\n[USER PREFERENCE]: When generating images, PREFER using the model '{imagine_model}'. Only deviate if explicitly asked otherwise."
        
        # Inject context about uploaded images (for Img2Img)
        if current_image_paths:
            img_instructions = (
                "\n\n[SYSTEM: IMAGE ATTACHED]\n"
                "The user has attached the following image(s). You can use 'generate_image' with 'image_path' to modify them:\n"
            )
            for path in current_image_paths:
                img_instructions += f"- {path}\n"
            current_system_prompt += img_instructions
        
        # 3.5. Retrieve Long-term Memories (Persistent Context)
        # IMPORTANT: Exclude current conversation to avoid echoing back what was just said.
        # Only retrieve memories from OTHER conversations for background context.
        memories = await vector_service.search_memories(
            user_input, 
            user_id,
            exclude_conversation_id=conversation_id
        )
        if memories:
            memory_context = (
                "\n\n[RECOLLECTED HISTORICAL BACKGROUND]:\n"
                "The following are FAINT memories from PAST conversations (NOT this one). "
                "RULES FOR USING THESE MEMORIES:\n"
                "- ONLY use these if the user EXPLICITLY asks about past conversations or personal preferences.\n"
                "- NEVER act on file paths, commands, or tasks mentioned in these memories.\n"
                "- NEVER confuse these memories with the user's CURRENT request.\n"
                "- If the user provides a DIRECT instruction (find a file, run a command), "
                "IGNORE these memories and execute the instruction using tools.\n"
                "Memories:\n- " + "\n- ".join(memories)
            )
            current_system_prompt += memory_context
            logger.info(f"Retrieved {len(memories)} long-term memories for context.")

        messages = [{"role": "system", "content": current_system_prompt}] + formatted_history

        # 5. Include tools only if NOT in vision mode (to avoid vision model complexity)
        include_tools = False
        if not thread_has_images:
            lower_input = user_input.lower()
            # Use substring matching instead of exact word matching
            # This catches multi-word phrases and partial matches
            if any(k in lower_input for k in self.FILE_KEYWORDS):
                include_tools = True
                logger.info("File-related keywords detected — tools enabled for this request.")
            # Also enable tools if the message references a filesystem path
            elif any(p in lower_input for p in ["/home", "/root", "/tmp", "/etc", "/var", "/opt", "c:\\", "d:\\"]):
                include_tools = True
                logger.info("Filesystem path detected — tools enabled for this request.")

        try:
            # Use longer timeout when tools are enabled (file searches can be slow)
            request_timeout = 120.0 if include_tools else 60.0
            async with httpx.AsyncClient(timeout=request_timeout) as client:
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
                            found = await file_service.search_host(
                                args.get("pattern"), 
                                root_dir=args.get("root")
                            )
                            result = f"Found files: {found}"
                        elif fn_name == "read_file":
                            content = await file_service.read_host_file(args.get("path"))
                            result = content if content else "File not found or empty."
                        elif fn_name == "save_file":
                            saved_path = await file_service.write_host_file(args.get("path"), args.get("content"))
                            if saved_path:
                                result = f"File saved successfully to: {saved_path}"
                            else:
                                result = "Failed to save file. Check the path and permissions."
                        elif fn_name == "analyze_image_file":
                            b64 = await file_service.read_host_image(args.get("path"))
                            if b64:
                                prompt = args.get("prompt", "Describe this image in detail.")
                                result = await self._vision_request(b64, prompt)
                            else:
                                result = "Error: Could not find or read the image file at that path."
                        elif fn_name == "generate_image":
                            prompt = args.get("prompt")
                            model_type = args.get("model_type", "stable-diffusion")
                            image_path = args.get("image_path")
                            result = await self._generate_image_request(prompt, model_type, image_path)
                        elif fn_name == "execute_system_command":
                            cmd = args.get("command")
                            logger.info(f"Executing system command: {cmd}")
                            import subprocess
                            try:
                                # Run command on host via subprocess
                                # Note: We join with host_root logic if it targets files, but often commands are general
                                process = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
                                result = process.stdout if process.returncode == 0 else f"Error ({process.returncode}): {process.stderr}"
                            except Exception as e:
                                result = f"Execution failed: {str(e)}"
                        
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
                
                # Store user input in Vector DB for long-term memory
                await vector_service.add_memory(user_input, user_id, user_msg.id, conversation_id)
                
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

    async def _generate_image_request(self, prompt: str, model_type: str, image_path: Optional[str] = None) -> str:
        """Internal helper to call the Imagine service."""
        try:
            logger.info(f"Requesting image generation: '{prompt}' via {model_type} (Img2Img: {bool(image_path)})")
            # Use a longer timeout for generation as it can take time
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "prompt": prompt,
                    "model_type": model_type,
                    "steps": 25 # Default
                }
                
                # If image_path is provided, read and encode it for Img2Img
                if image_path:
                    try:
                        import base64
                        with open(image_path, "rb") as f:
                            encoded_img = base64.b64encode(f.read()).decode("utf-8")
                        payload["image"] = encoded_img
                        logger.info(f"Attaching image from {image_path} for generation.")
                    except Exception as e:
                        logger.warning(f"Failed to read image for Img2Img: {e}. Falling back to Text2Img.")

                response = await client.post(f"{settings.IMAGINE_API_URL}/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "success":
                    import base64
                    import uuid
                    
                    # Decode and save to file
                    img_data = base64.b64decode(data["image_base64"])
                    filename = f"generated_{uuid.uuid4().hex[:8]}.jpg"
                    
                    # Save to Bipod's generated directory
                    # We use file_service logic manually here to access the path
                    filepath = os.path.join(settings.GENERATED_DIR, filename)
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                        
                    logger.info(f"Generated image saved to {filepath}")
                    return f"Image generated successfully! Saved to: {filepath}"
                else:
                   return f"Generation failed: {data}"

        except Exception as e:
            logger.error(f"Image generation failure: {e}")
            return f"Failed to generate image: {str(e)}. Ensure the 'imagine' service is running."

brain_service = BrainService()

