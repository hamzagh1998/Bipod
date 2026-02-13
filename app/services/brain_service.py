import json
import os
import httpx
import re
import asyncio
import platform
import multiprocessing
import subprocess
import datetime
import base64
import uuid
import io
from typing import List, Dict, Optional
from pypdf import PdfReader
from duckduckgo_search import DDGS

from app.core.config import settings
from app.core.logger import get_logger
from app.services.file_service import file_service
from app.services.memory_service import memory_service
from app.services.vector_service import vector_service
from app.services.intent_router import intent_router

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
            "You have a specialized tool to get real-time SYSTEM INFORMATION (CPU, GPU, OS, Time). "
            "### TOOL USAGE EXAMPLES:\n"
            "If you need to call a tool, output valid JSON within your response. Example:\n"
            "{\"name\": \"get_system_info\", \"arguments\": {}}\n"
            "{\"name\": \"execute_system_command\", \"arguments\": {\"command\": \"ls -la\"}}\n\n"
            "### CORE DIRECTIVES:\n"
            "1. ALWAYS PRIORITIZE the current user message. Execute exactly what the user asks.\n"
            "2. If a user asks to **draw**, **create**, **make**, or **generate** an image, you **MUST** use the `generate_image` tool. **NEVER** try to use `execute_system_command` for this.\n"
            "3. Use `search_files` to find filenames. Use `read_file` to see content. Use `save_file` to persist progress.\n"
            "4. DO NOT explain why you are using a tool unless it's a complex multi-step process.\n"
            "5. If a specialized tool exists (save_file, generate_image, search_files), use it instead of `execute_system_command`.\n"
            "6. EACH CONVERSATION IS INDEPENDENT. Do not mix up content from different sessions.\n"
            "7. For image generation, the `model_type` MUST be either 'stable-diffusion' (default) or 'dalle-mini' (only if 'fast' or 'rough' is requested). DO NOT use any other words for `model_type`.\n"
            "8. ALWAYS provide the actual file path returned by the tool when confirming a task (e.g. 'Image saved to /app/data/...'). If an image was generated, you MUST include the markdown preview (e.g. ![Generated Image](/generated/filename.jpg)) EXACTLY as returned by the tool so the user can see it in the chat.\n"
            "[SYSTEM: TOOLS]\n1. `read_file`: Reads text/PDF from the host. \n2. `save_file`: Writes text to the host. \n3. `move_file`: Moves/renames files/dirs on the host. Supports wildcards (e.g. *.pdf). \n4. `delete_file`: Deletes files/dirs on the host. Supports wildcards. \n5. `search_files`: Finds files by pattern. \n6. `execute_system_command`: Runs shell commands (use for complex tasks). Prefix host paths with /host.\n7. `get_system_info`: Returns CPU/GPU usage, model info, and current time. \n8. `web_search`: Searches the internet. \n9. `fetch_web_page`: Reads a URL.\n10. `organize_files`: Automatically sorts files in a directory into folders by their extension (e.g. 'pdf/', 'docx/').\n\n"
            "- If you decide to call a tool, you MUST NOT say anything. Your entire response must be ONLY the JSON tool call.\n"
            "- THE USER CANNOT SEE YOUR TOOL CALLS. If you narrate them, you are talking to yourself and confusing the user.\n"
            "- Just do the work. Once the tool finishes, you can give a final summary.\n"
            "9. If the user asks for the CURRENT TIME, CPU usage, GPU status, or OS info, you MUST use the `get_system_info` tool. DO NOT guess or claim you lack access. \n"
            "10. You have INTERNET ACCESS via `web_search` and `fetch_web_page`. Use them to find current information, news, or to summarize specific web pages. \n"
            "11. **ULTRA-CRITICAL - THE TRUTH DIRECTIVE**: You MUST acknowledge that your internal training data is OUTDATED. "
            "For current events (like 'Who is the Prime Minister?' or 'Bitcoin Price'), you MUST SEARCH THE WEB. "
            "SEARCH RESULTS ARE THE ABSOLUTE SOURCE OF TRUTH. Use 'latest', 'today', or 'current' in your search queries to get fresh snippets.\n"
            "The information you provide MUST come from the tool output. "
            "NEVER mention internal tool names or installation issues to the user. Just execute the tool. \n"
            "12. **SILENT TOOL CALLING**: When you decide to use a tool, your entire message should ONLY contain the JSON for the tool call. No preamble, no postamble. \n"
            "### FORMATTING RULES:\n"
            "1. ALWAYS wrap code snippets in triple backticks (```) and specify the programming language (e.g. ```python) for proper syntax highlighting.\n"
            "2. DO NOT just write the code as plain text.\n"
            "Keep your internal thoughts concise and focus only on the logic of the task."
        )

        self.router = intent_router

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
                    "description": "Reads an EXISTING image file from the host filesystem and describes its content using the vision model (Moondream). This tool ONLY analyzes and describes existing images — it does NOT generate, create, or produce new images.",
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
                    "description": "Generates a new premium-quality image based on a text prompt. The tool automatically handles high-resolution descriptors. Returns the path to the generated image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Description of the image. For best results, Bipod will expand this with quality tags like 'cinematic lighting, 8k, highly detailed, masterpiece'."},
                            "image_path": {"type": "string", "description": "Optional. Path to an existing image file to use for variations."},
                            "model_type": {"type": "string", "enum": ["stable-diffusion-xl", "dalle-mini"], "description": "Model to use. Default 'stable-diffusion-xl'."},
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
                            "src": {"type": "string", "description": "The current absolute path to the file/directory."},
                            "dest": {"type": "string", "description": "The new absolute path or destination directory."},
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
                            "path": {"type": "string", "description": "The absolute path to the file/directory to delete."},
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
                            "directory": {"type": "string", "description": "The absolute path to the directory whose files should be organized."},
                        },
                        "required": ["directory"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_system_command",
                    "description": "Executes a shell command on the host system. Use this ONLY for complex operations that specific tools (move_file, delete_file, search_files, organize_files) cannot handle. IMPORTANT: Any absolute paths starting with / MUST be prefixed with /host in your command string (e.g. 'ls /host/home/user').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The full shell command to execute. Prefix host paths with /host."},
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
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Searches the internet for the given query using DuckDuckGo. Returns a list of relevant search results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query to look up on the internet."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_web_page",
                    "description": "Retrieves the content of a specific web page/URL and returns its text content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL of the web page to fetch."}
                        },
                        "required": ["url"]
                    }
                }
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

        # 4. Determine model to use
        target_model = model_id if model_id else self.active_model
        
        # Determine if this is a generation request (to avoid switching to vision model which can't call tools)
        lower_input = user_input.lower()
        generation_keywords = {"generate", "draw", "make", "create", "paint", "imagine", "variation", "another", "better", "fix"}
        is_generation_request = any(k in lower_input for k in generation_keywords)

        # Only force vision model for specialized analysis, not for generation/search
        if thread_has_images and not is_generation_request:
            # If the user is asking to "see", "describe" or "analyze", we NEED the vision model
            vision_trigger = {"describe", "see", "what", "analyze", "explain", "look"}
            if any(v in lower_input for v in vision_trigger):
                target_model = settings.VISION_MODEL
                logger.info("Vision task detected — switching brain to specialized eyes.")

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
        # Inject real-time context and mode instruction into system prompt for this turn
        now = datetime.datetime.now()
        time_context = f"\n\n[SYSTEM: REAL-TIME CONTEXT]\n- Current Date/Time: {now.strftime('%A, %B %d, %Y %I:%M %p')}\n- Location: Host Machine (Bipod Space)"
        
        current_system_prompt = self.system_prompt + time_context + mode_instruction
        
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

        # 5. Semantic Intent Classification
        # We classify intent BEFORE calling the brain. This decides which tools to provide.
        intent = await self.router.classify(user_input)
        filtered_tools = self.router.get_tools_for_intent(intent, self.tools) if intent else []
        # Only enable tools if the router actually mapped the intent to real tools
        include_tools = len(filtered_tools) > 0

        if include_tools:
            logger.info(f"Classified intent: '{intent}'. Tools: {[t['function']['name'] for t in filtered_tools]}")
        else:
            logger.info("No tool-related intent detected — processing as pure chat.")


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
                    payload["tools"] = filtered_tools

                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                message = data.get("message", {})

                # 2. Recursive Tool Execution Loop (up to 5 turns)
                max_turns = 5
                turn = 0
                thoughts_buffer = ""
                executed_tool_calls_hash = set() # To prevent infinite loops (Robot Stutter)
                
                allowed_tool_names = {t["function"]["name"] for t in filtered_tools} if include_tools else set()

                while turn < max_turns:
                    # 2.1 Detect and Strip Hallucinated Tool Calls
                    hallucinated_calls, stripped_content = self._check_for_hallucinated_tools(
                        message.get("content", ""), allowed_tool_names
                    )
                    
                    tool_calls = message.get("tool_calls", [])
                    if include_tools and not tool_calls:
                        tool_calls = hallucinated_calls

                    stripped_content_lower = stripped_content.lower()
                    lazy_phrases = [
                        "i need to search", "check the web", "browse the internet", 
                        "search online", "look that up", "use the internet", "online data",
                        "research the", "looking up", "fetch the latest", "need to check",
                        "use the get_system_info", "use the tool", "check the motherboard"
                    ]
                    is_lazy_response = any(p in stripped_content_lower for p in lazy_phrases)
                    
                    # 2.2 Force-injection and Lazy Detection for Web Search
                    # If intent was web_search OR model says "I need to search" (lazy mode), force inject tool
                    should_force_search = (intent == "web_search" and turn == 0 and include_tools) or (is_lazy_response and include_tools)
                    
                    if not tool_calls and should_force_search:
                        logger.info(f"Forcing web_search (Intent: {intent}, Lazy: {is_lazy_response})")
                        tool_calls = [{
                            "id": f"call_{os.urandom(4).hex()}",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": user_input}
                            }
                        }]

                    # 2.4 Clean content of markers and JSON
                    message["content"] = stripped_content

                    # 2.5 Accumulate thoughts ONLY if this turn resulted in tools
                    # Filtering: skip turn 0 narration if it's just technical boilerplate
                    if message["content"] and tool_calls:
                        c_low = message["content"].lower()
                        # More aggressive filtering for boilerplate
                        skip_phrases = [
                            "here is", "json", "will use", "function call", "arguments", "using the tool",
                            "i will now use", "calling the tool", "i'm using the", "tool call", "tool output",
                            "i've decided to use", "i'll use the", "i'm going to use"
                        ]
                        is_boilerplate = len(message["content"]) < 200 and any(p in c_low for p in skip_phrases)
                        
                        if not is_boilerplate:
                            thoughts_buffer += message["content"].strip() + "\n\n"

                    if not tool_calls:
                        break  # No more tools, we're done

                    turn += 1
                    logger.info(f"Processing tool turn {turn}/{max_turns}...")
                    
                    # Ensure assistant message with tool_calls is in history for the brain
                    if "tool_calls" not in message:
                        message["tool_calls"] = tool_calls
                    messages.append(message)

                    filtered_tool_calls = []
                    for tc in tool_calls:
                        call_hash = f"{tc['function']['name']}:{json.dumps(tc['function']['arguments'], sort_keys=True)}"
                        if call_hash in executed_tool_calls_hash:
                            logger.warning(f"Skipping duplicate tool call to prevent loop: {tc['function']['name']}")
                            continue
                        executed_tool_calls_hash.add(call_hash)
                        filtered_tool_calls.append(tc)
                    
                    if not filtered_tool_calls:
                        break

                    for tool_call in filtered_tool_calls:
                        fn_name = tool_call["function"]["name"]
                        args = tool_call["function"]["arguments"]
                        
                        logger.info(f"Executing tool: {fn_name}")
                        
                        result = ""
                        try:
                            if fn_name == "search_files":
                                found = await file_service.search_host(args.get("pattern"), root_dir=args.get("root"))
                                result = f"Found files: {found}"
                            elif fn_name == "read_file":
                                res = await file_service.read_host_file(args.get("path"))
                                result = res if res else "File not found or empty."
                            elif fn_name == "save_file":
                                saved = await file_service.write_host_file(args.get("path"), args.get("content"))
                                result = f"File saved to: {saved}" if saved else "Failed to save file."
                            elif fn_name == "analyze_image_file":
                                b64 = await file_service.read_host_image(args.get("path"))
                                if b64:
                                    result = await self._vision_request(b64, args.get("prompt", "Describe this image."))
                                else:
                                    result = "Error: Could not read image."
                            elif fn_name == "generate_image":
                                override_model = imagine_model or args.get("model_type") or settings.ACTIVE_IMAGINE_MODEL
                                result = await self._generate_image_request(args.get("prompt"), override_model, args.get("image_path"))
                            elif fn_name == "move_file":
                                ok = await file_service.move_host_file(args.get("src"), args.get("dest"))
                                result = f"Successfully moved {args.get('src')} to {args.get('dest')}" if ok else "Failed to move file."
                            elif fn_name == "delete_file":
                                ok = await file_service.delete_host_file(args.get("path"))
                                result = f"Successfully deleted {args.get('path')}" if ok else "Failed to delete file."
                            elif fn_name == "organize_files":
                                organized_count = await file_service.organize_host_directory(args.get("directory"))
                                result = f"Successfully organized {organized_count} files in '{args.get('directory')}'."
                            elif fn_name == "execute_system_command":
                                import subprocess
                                # Handle 'cmd' alias (common hallucination)
                                command = args.get("command") or args.get("cmd")
                                if not command:
                                    result = "Error: Missing 'command' argument."
                                else:
                                    logger.info(f"Running system command: {command}")
                                    p = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
                                    result = p.stdout if p.returncode == 0 else f"Error: {p.stderr}"
                            elif fn_name == "get_system_info":
                                # 1. Basic OS/Time
                                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
                                cores = multiprocessing.cpu_count()
                                
                                # 2. GPU Info
                                gpu_info = "None detected."
                                try:
                                    gp = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                                    if gp.returncode == 0: gpu_info = gp.stdout.strip()
                                except: pass
                                
                                # 3. CPU/Performance
                                cpu_usage = "Unknown"
                                cpu_model = platform.processor() or "Unknown"
                                if platform.system() == "Linux":
                                    try:
                                        with open("/proc/cpuinfo", "r") as f:
                                            for line in f:
                                                if "model name" in line:
                                                    cpu_model = line.split(":", 1)[1].strip()
                                                    break
                                    except: pass

                                try:
                                    # Improved CPU usage check using 'top'
                                    cp_usage = subprocess.run("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'", shell=True, capture_output=True, text=True, timeout=2)
                                    if cp_usage.returncode == 0: 
                                        val = cp_usage.stdout.strip().replace(',', '.')
                                        cpu_usage = f"{val}%"
                                except: pass

                                # 4. Motherboard / Baseboard Info
                                mobo_info = "Unknown"
                                try:
                                    m_vendor = "Unknown"
                                    m_product = "Unknown"
                                    if os.path.exists("/sys/class/dmi/id/board_vendor"):
                                        with open("/sys/class/dmi/id/board_vendor", "r") as f: m_vendor = f.read().strip()
                                    if os.path.exists("/sys/class/dmi/id/board_name"):
                                        with open("/sys/class/dmi/id/board_name", "r") as f: m_product = f.read().strip()
                                    
                                    if m_vendor != "Unknown" or m_product != "Unknown":
                                        mobo_info = f"{m_vendor} {m_product}"
                                    else:
                                        # Fallback to dmidecode if available (often requires root, but checking)
                                        dm = subprocess.run("dmidecode -s baseboard-product-name", shell=True, capture_output=True, text=True, timeout=2)
                                        if dm.returncode == 0: mobo_info = dm.stdout.strip()
                                except: pass

                                result = (
                                    "### [REAL-TIME SYSTEM DATA]\n"
                                    f"- **Current Time**: {now}\n"
                                    f"- **Host OS**: {os_info}\n"
                                    f"- **CPU**: {cpu_model} ({cores} cores), {cpu_usage} usage\n"
                                    f"- **Motherboard**: {mobo_info}\n"
                                    f"- **GPU Status**: {gpu_info}"
                                )
                            elif fn_name == "web_search":
                                query = args.get("query")
                                
                                def _do_search():
                                    # Heuristic: if asking for "current", "latest", "news", or "today", force freshness
                                    time_sensitive_keywords = {
                                        "current", "latest", "news", "today", "now", "recent", "update",
                                        "price", "value", "cost", "score", "winner", "result", "finance", "stock"
                                    }
                                    is_fresh = any(k in query.lower() for k in time_sensitive_keywords)
                                    
                                    # 'w' = past week, 'm' = past month, None = no limit
                                    time_limit = "w" if is_fresh else None
                                    
                                    with DDGS(timeout=20) as ddgs:
                                        # First try with time limit if fresh
                                        results = []
                                        if is_fresh:
                                            try:
                                                # timelimit='w' (past week), 'm' (past month), 'y' (past year)
                                                results = list(ddgs.text(query, region='wt-wt', safesearch='off', timelimit=time_limit, max_results=8))
                                            except Exception:
                                                pass
                                        
                                        # Fallback to general search if no results or not fresh
                                        if not results:
                                            results = list(ddgs.text(query, region='wt-wt', safesearch='off', max_results=8))
                                            
                                        return results
                                
                                try:
                                    results = await asyncio.to_thread(_do_search)
                                    if not results:
                                        result = f"I searched internet for '{query}' but found no direct results. Please try a different query."
                                    else:
                                        res_str = f"Search results for '{query}':\n\n"
                                        for i, r in enumerate(results):
                                            res_str += f"{i+1}. **{r['title']}**\n   {r['href']}\n   {r['body']}\n\n"
                                        result = res_str
                                except Exception as e:
                                    logger.error(f"Search tool failed: {e}")
                                    result = f"Search failed: {str(e)}. I should verify my query or try another search tool if available."
                            elif fn_name == "fetch_web_page":
                                url = args.get("url")
                                async with httpx.AsyncClient(timeout=30.0) as tool_client:
                                    resp = await tool_client.get(url, follow_redirects=True)
                                    resp.raise_for_status()
                                    # Simple HTML cleaning: remove scripts and styles
                                    html = resp.text
                                    text = re.sub(r'<(script|style|header|footer|nav).*?>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
                                    text = re.sub(r'<.*?>', ' ', text)
                                    text = re.sub(r'\s+', ' ', text).strip()
                                    result = f"Content from {url}:\n\n{text[:5000]}..."
                        except Exception as e:
                            result = f"Exception executing tool: {str(e)}"

                        if isinstance(result, str) and len(result) > 25000:
                            result = await self._map_reduce_summarize(result)

                        messages.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call.get("id")
                        })

                    json_payload = {
                        "model": target_model, 
                        "messages": messages, 
                        "stream": False, 
                    }
                    if include_tools:
                        json_payload["tools"] = filtered_tools
                        
                    resp = await client.post(
                        f"{self.base_url}/api/chat",
                        json=json_payload,
                    )
                    resp.raise_for_status()
                    message = resp.json().get("message", {})

                final_answer = message.get("content", "")

                # 3. Ensure generated image tags are present in the final answer
                # If a generate_image tool was called successfully but the tag is missing from the brain's final answer, append it.
                generated_images = []
                tool_results_summary = []
                for m in messages:
                    if m.get("role") == "tool":
                        content = m.get("content", "")
                        if "![Generated Image]" in content:
                            match = re.search(r'(!\[Generated Image\]\(.*?\))', content)
                            if match:
                                generated_images.append(match.group(1))
                        # Keep track of all tool results as a fallback
                        tool_results_summary.append(content)

                for img_tag in generated_images:
                    if img_tag not in final_answer:
                        final_answer += f"\n\n{img_tag}"
                
                if thoughts_buffer.strip():
                    ai_message = f"{thoughts_buffer.strip()}\n\n{final_answer}".strip()
                else:
                    ai_message = final_answer
                
                if not ai_message.strip():
                    # Fallback if the brain didn't produce anything
                    if tool_results_summary:
                        # Use the last relevant tool result instead of a generic error
                        # This ensures the image or search results are SHOWN even if Ollama is silent.
                        ai_message = "\n\n".join(tool_results_summary).strip()
                        logger.info("Brain was silent after tool execution. Using tool results as fallback.")
                    else:
                        ai_message = "I'm listening, but I didn't quite get that. Could you rephrase your request?"

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

    async def clear_memory(self, conversation_id: str):
        await memory_service.clear_conversation(conversation_id)
        await vector_service.delete_conversation(conversation_id)

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
                                json={"model": name, "keep_alive": 0}
                            )
                        if not running:
                            logger.info("Ollama reports no models currently in VRAM.")
                    else:
                        raise Exception(f"PS endpoint returned {ps_resp.status_code}")
                except Exception as ps_e:
                    logger.warning(f"Could not query running models ({ps_e}), falling back to default list.")
                    # Fallback: Unload the usual suspects
                    for m in [self.active_model, settings.VISION_MODEL, settings.EMBEDDING_MODEL]:
                        await u_client.post(
                            f"{self.base_url}/api/generate", 
                            json={"model": m, "keep_alive": 0}
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
                    "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
                    "stream": False
                }
                response = await v_client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                return response.json().get("message", {}).get("content", "I saw the image but couldn't think of anything to say.")
        except Exception as e:
            logger.error(f"Vision tool failure: {e}")
            return f"Error analyzing image: {str(e)}"

    async def _generate_image_request(self, prompt: str, model_type: str = "stable-diffusion-xl", image_path: Optional[str] = None) -> str:
        """Internal helper to call the Imagine service."""
        try:
            # 1. First, tell Ollama to get out of the GPU
            await self._unload_ollama()

            logger.info(f"Requesting image generation: '{prompt}' via {model_type} (Img2Img: {bool(image_path)})")
            # Increase timeout to 10 minutes: first-run model download is ~4GB
            async with httpx.AsyncClient(timeout=600.0) as client:
                payload = {
                    "prompt": prompt,
                    "model_type": model_type,
                    "steps": 40 # Upgraded quality
                }
                
                # If image_path is provided, read and encode it for Img2Img
                if image_path:
                    try:
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

        except httpx.ReadTimeout:
            logger.error("Image generation timed out.")
            return "Generation timed out. Bipod is likely downloading the model weights for the first time (approx 4GB). Please wait a few minutes and try again — the download will continue in the background."
        except Exception as e:
            logger.error(f"Image generation failure: {e}")
            return f"Failed to generate image: {str(e)}. If this is your first time, Bipod might still be downloading the model (4GB) or the 'imagine' service is starting up."

    async def _map_reduce_summarize(self, text: str) -> str:
        """Summarizes large text chunks using a Map-Reduce approach."""
        chunk_size = 25000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
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

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": settings.ACTIVE_MODEL,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": text}
                        ],
                        "stream": False,
                    }
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:1000] + "... [Summary failed]"

    def _check_for_hallucinated_tools(self, content: str, allowed_tools: set[str] | None = None) -> tuple[List[Dict], str]:
        """Detects tool calls that the model output as plain text JSON instead of real tool_calls.
        Only strips blocks that are successfully identified as internal tool calls.
        """
        cleaned_content = content
        all_tool_names = {t["function"]["name"] for t in self.tools}
        valid_tools = allowed_tools if allowed_tools else all_tool_names
        
        tool_calls = []
        strip_ranges = [] # Track where internal content was found to strip it later
        
        # 1. Strip specific internal model tags that should NEVER be seen
        internal_markers = ["<|python_tag|>", "<|action_tag|>", "<|thought|>"]
        for marker in internal_markers:
            for m in re.finditer(re.escape(marker), cleaned_content):
                strip_ranges.append((m.start(), m.end()))
        
        # 2. Extract JSON blocks using brace-counting
        start_idx = -1
        brace_count = 0
        
        for i, char in enumerate(cleaned_content):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    potential_json = cleaned_content[start_idx:i+1]
                    
                    try:
                        data = json.loads(potential_json)
                        fn_name = data.get("name") or data.get("function", {}).get("name")
                        args = data.get("parameters") or data.get("arguments") or data.get("function", {}).get("arguments")
                        
                        # Alias 'cmd' to 'command' for shell tool
                        if fn_name == "shell" and args and "cmd" in args:
                            args["command"] = args.pop("cmd")

                        # Strip if it LOOKS like a tool call (has both 'name' AND 'arguments'/'parameters')
                        # This catches real tools AND completely hallucinated ones (e.g. 'generate_story')
                        # but avoids stripping random JSON data that just has a 'name' key
                        is_tool_shaped = fn_name and (args is not None or "arguments" in data or "parameters" in data)
                        if is_tool_shaped:
                            # Also try to strip surrounding backticks if present
                            current_start = start_idx
                            current_end = i + 1
                            
                            pre = cleaned_content[max(0, current_start-10):current_start]
                            post = cleaned_content[current_end:min(len(cleaned_content), current_end+11)]
                            
                            if "```json" in pre:
                                current_start = cleaned_content.rfind("```json", 0, current_start)
                            elif "```" in pre:
                                current_start = cleaned_content.rfind("```", 0, current_start)
                            
                            if "```" in post:
                                # Find the closing backticks after the JSON block
                                closing_backticks_idx = cleaned_content.find("```", current_end)
                                if closing_backticks_idx != -1:
                                    current_end = closing_backticks_idx + 3 # Include the '```'
                            
                            # Aggressively strip surrounding characters that common models leak (stray braces, backticks, newlines)
                            while current_start > 0 and cleaned_content[current_start-1] in [" ", "\n", "\r", "}", "]", "`", ":", ","]:
                                current_start -= 1
                            while current_end < len(cleaned_content) and cleaned_content[current_end] in [" ", "\n", "\r", "{", "[", "`", ":", ","]:
                                current_end += 1

                            strip_ranges.append((current_start, current_end))

                            if fn_name in valid_tools:
                                tool_calls.append({
                                    "id": f"call_{os.urandom(4).hex()}",
                                    "type": "function",
                                    "function": {
                                        "name": fn_name,
                                        "arguments": args
                                    }
                                })
                            elif fn_name in all_tool_names:
                                logger.warning(f"Blocked hallucinated call to '{fn_name}' — not in allowed tool set.")
                            else:
                                logger.warning(f"Stripped completely fake tool call: '{fn_name}' — tool does not exist.")
                    except:
                        # 2b. Attempt to recover malformed JSON for known tools (LLMs often output broken JSON)
                        # Regex to find "name": "something" pattern
                        name_match = re.search(r'"name"\s*:\s*"(\w+)"', potential_json)
                        if name_match:
                            found_name = name_match.group(1)
                            
                            # SAFETY: Only strip if it looks like a tool call (has args key) OR is a known tool
                            # This prevents stripping `{"name": "John"}` in normal code
                            has_args_key = "arguments" in potential_json or "parameters" in potential_json
                            is_known_tool = found_name in all_tool_names
                            
                            if has_args_key or is_known_tool:
                                # Determine range to strip (same logic as above)
                                current_start = start_idx
                                current_end = i + 1
                                # Expand to surrounding backticks
                                pre = cleaned_content[max(0, current_start-10):current_start]
                                if "```json" in pre: current_start = cleaned_content.rfind("```json", 0, current_start)
                                elif "```" in pre: current_start = cleaned_content.rfind("```", 0, current_start)
                                
                                post = cleaned_content[current_end:min(len(cleaned_content), current_end+11)]
                                if "```" in post:
                                    cb_idx = cleaned_content.find("```", current_end)
                                    if cb_idx != -1: current_end = cb_idx + 3

                                # Aggressively strip surrounding characters that common models leak (stray braces, backticks, newlines)
                                while current_start > 0 and cleaned_content[current_start-1] in [" ", "\n", "\r", "}", "]", "`", ":", ","]:
                                    current_start -= 1
                                while current_end < len(cleaned_content) and cleaned_content[current_end] in [" ", "\n", "\r", "{", "[", "`", ":", ","]:
                                    current_end += 1

                                # ALWAYS strip the broken tool-like JSON
                                strip_ranges.append((current_start, current_end))
                                
                                # RECOVERY: specialized fix for No-Arg tools like get_system_info
                                found_name_valid = found_name in valid_tools
                                if found_name == "get_system_info" and found_name_valid:
                                    logger.info("Recovered malformed JSON for get_system_info")
                                    tool_calls.append({
                                        "id": f"call_{os.urandom(4).hex()}",
                                        "type": "function",
                                        "function": {"name": "get_system_info", "arguments": {}}
                                    })
                                elif found_name_valid:
                                    # RECOVERY: specialized fix for shell commands with broken JSON
                                    if found_name == "execute_system_command":
                                        cmd_match = re.search(r'["\'](?:command|cmd)["\']\s*:\s*["\'](.*?)["\']', potential_json)
                                        if cmd_match:
                                            logger.info("Recovered malformed JSON for execute_system_command")
                                            tool_calls.append({
                                                "id": f"call_{os.urandom(4).hex()}",
                                                "type": "function",
                                                "function": {"name": "execute_system_command", "arguments": {"command": cmd_match.group(1)}}
                                            })
                                            continue
                                            
                                    logger.warning(f"Detected malformed JSON for '{found_name}' — cannot execute safely.")
                                else:
                                    logger.warning(f"Stripped hallucinated/malformed tool: '{found_name}'")

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
                    if raw_args.startswith('{'):
                        try: final_args = json.loads(raw_args)
                        except: continue
                    elif str_arg:
                        if fname == "execute_system_command": final_args = {"command": str_arg}
                        elif fname == "read_file": final_args = {"path": str_arg}
                        elif fname == "search_files": final_args = {"pattern": str_arg}
                    
                    tool_calls.append({
                        "id": f"call_{os.urandom(4).hex()}",
                        "type": "function",
                        "function": {
                            "name": fname,
                            "arguments": final_args
                        }
                    })

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
