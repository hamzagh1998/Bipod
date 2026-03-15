import datetime
import json
import multiprocessing
import os
import platform
import re
import subprocess
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from app.core.config import settings
from app.core.logger import get_logger
from app.services.brain.contracts import OrchestrationResult
from app.services.file_service import file_service

logger = get_logger("bipod.services.brain.tools")

ProgressCallback = Callable[[str, Dict[str, Any]], Awaitable[None]]


class ToolOrchestrator:
    """Runs iterative tool/model turns and executes approved tools."""

    WEB_SEARCH_SENTINEL_RE = re.compile(
        r"\[\[\s*BIPOD_WEB_SEARCH\s*:\s*(.+?)\s*\]\]",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        base_url: str,
        tools: List[Dict],
        options_provider: Callable[[], Dict[str, float | int]],
        check_hallucinated_tools: Callable[[str, Optional[set[str]]], tuple[List[Dict], str]],
        run_web_search: Callable[[str], Awaitable[str]],
        vision_request: Callable[[str, str], Awaitable[str]],
        generate_image_request: Callable[[str, str, Optional[str]], Awaitable[str]],
        map_reduce_summarize: Callable[[str], Awaitable[str]],
    ):
        self.base_url = base_url
        self.tools = tools
        self._options_provider = options_provider
        self._check_hallucinated_tools = check_hallucinated_tools
        self._run_web_search = run_web_search
        self._vision_request = vision_request
        self._generate_image_request = generate_image_request
        self._map_reduce_summarize = map_reduce_summarize

    async def run(
        self,
        client: httpx.AsyncClient,
        target_model: str,
        messages: List[Dict],
        filtered_tools: List[Dict],
        include_tools: bool,
        intent: Optional[str],
        user_input: str,
        imagine_model: Optional[str],
        configured_model: str,
        active_imagine_model: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> OrchestrationResult:
        await self._emit_progress(
            progress_callback,
            "model_request",
            label=f"Querying {target_model}",
            detail="Starting the first model pass.",
        )
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": self._options_provider(),
        }
        if include_tools:
            payload["tools"] = filtered_tools

        response = await client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        message = response.json().get("message", {})

        max_turns = 5
        turn = 0
        executed_tool_calls_hash = set()
        executed_tools: List[str] = []
        image_generation_result = ""
        allowed_tool_names = {t["function"]["name"] for t in filtered_tools} if include_tools else set()

        while turn < max_turns:
            hallucinated_calls, stripped_content = self._check_hallucinated_tools(
                message.get("content", ""), allowed_tool_names
            )

            tool_calls = message.get("tool_calls", [])
            if include_tools and not tool_calls:
                tool_calls = hallucinated_calls

            sentinel_match = self.WEB_SEARCH_SENTINEL_RE.search(stripped_content or "")
            should_force_search = (
                include_tools and turn == 0 and intent == "web_search"
            ) or (
                include_tools and sentinel_match is not None
            )
            if not tool_calls and should_force_search:
                forced_query = user_input
                if sentinel_match:
                    forced_query = re.sub(r"\s+", " ", sentinel_match.group(1)).strip() or user_input
                logger.info(f"Forcing web_search (Intent: {intent}, Sentinel: {bool(sentinel_match)})")
                tool_calls = [{
                    "id": f"call_{os.urandom(4).hex()}",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": forced_query}
                    }
                }]

            message["content"] = stripped_content
            if not tool_calls:
                break

            turn += 1
            logger.info(f"Processing tool turn {turn}/{max_turns}...")
            await self._emit_progress(
                progress_callback,
                "tool_turn",
                label=f"Running tool step {turn}",
                detail="The model requested external tools before answering.",
                turn=turn,
            )

            if "tool_calls" not in message:
                message["tool_calls"] = tool_calls
            messages.append(message)

            filtered_tool_calls = []
            for tool_call in tool_calls:
                call_hash = f"{tool_call['function']['name']}:{json.dumps(tool_call['function']['arguments'], sort_keys=True)}"
                if call_hash in executed_tool_calls_hash:
                    logger.warning(f"Skipping duplicate tool call to prevent loop: {tool_call['function']['name']}")
                    continue
                executed_tool_calls_hash.add(call_hash)
                filtered_tool_calls.append(tool_call)

            if not filtered_tool_calls:
                break

            for tool_call in filtered_tool_calls:
                executed_tools.append(tool_call["function"]["name"])
                tool_name = tool_call["function"]["name"]
                await self._emit_progress(
                    progress_callback,
                    "tool_call",
                    label=self._tool_status_label(tool_name),
                    detail=self._tool_status_detail(tool_name, tool_call["function"].get("arguments")),
                    tool_name=tool_name,
                )
                result = await self._execute_tool_call(
                    tool_call=tool_call,
                    imagine_model=imagine_model,
                    configured_model=configured_model,
                    target_model=target_model,
                    active_imagine_model=active_imagine_model,
                )
                if tool_name == "generate_image":
                    image_generation_result = result
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.get("id"),
                })

            json_payload = {
                "model": target_model,
                "messages": messages,
                "stream": False,
                "options": self._options_provider(),
            }
            if include_tools:
                json_payload["tools"] = filtered_tools

            await self._emit_progress(
                progress_callback,
                "model_request",
                label=f"Reasoning over tool results with {target_model}",
                detail="Feeding tool output back to the model.",
            )
            resp = await client.post(f"{self.base_url}/api/chat", json=json_payload)
            resp.raise_for_status()
            message = resp.json().get("message", {})

        final_answer = message.get("content", "")
        generated_images: List[str] = []
        tool_results_summary: List[str] = []
        for message_item in messages:
            if message_item.get("role") == "tool":
                content = message_item.get("content", "")
                if "![Generated Image]" in content:
                    match = re.search(r'(!\[Generated Image\]\(.*?\))', content)
                    if match:
                        generated_images.append(match.group(1))
                tool_results_summary.append(content)

        return OrchestrationResult(
            final_answer=final_answer,
            messages=messages,
            tool_results_summary=tool_results_summary,
            generated_images=generated_images,
            executed_tools=executed_tools,
            image_generation_result=image_generation_result,
        )

    async def _emit_progress(
        self,
        progress_callback: Optional[ProgressCallback],
        event: str,
        **payload: Any,
    ) -> None:
        if progress_callback is None:
            return
        await progress_callback(event, payload)

    def _tool_status_label(self, tool_name: str) -> str:
        labels = {
            "search_files": "Scanning files",
            "read_file": "Reading a file",
            "save_file": "Saving a file",
            "analyze_image_file": "Inspecting an image",
            "generate_image": "Generating an image",
            "move_file": "Moving files",
            "delete_file": "Deleting files",
            "organize_files": "Organizing files",
            "get_system_info": "Checking system info",
            "web_search": "Searching the web",
            "fetch_web_page": "Reading a web page",
        }
        return labels.get(tool_name, f"Running {tool_name}")

    def _tool_status_detail(self, tool_name: str, args: Optional[Dict[str, Any]]) -> str:
        if not isinstance(args, dict):
            return "Working on the requested task."

        if tool_name == "web_search":
            query = str(args.get("query", "")).strip()
            return f"Looking up: {query}" if query else "Looking up current information."
        if tool_name == "fetch_web_page":
            url = str(args.get("url", "")).strip()
            return f"Fetching: {url}" if url else "Fetching the requested page."
        if tool_name == "read_file":
            path = str(args.get("path", "")).strip()
            return f"Opening: {path}" if path else "Opening the requested file."
        if tool_name == "search_files":
            pattern = str(args.get("pattern", "")).strip()
            return f"Pattern: {pattern}" if pattern else "Scanning the requested location."
        if tool_name == "generate_image":
            return "Rendering the image request."

        return "Working on the requested task."

    async def _execute_tool_call(
        self,
        tool_call: Dict,
        imagine_model: Optional[str],
        configured_model: str,
        target_model: str,
        active_imagine_model: str,
    ) -> str:
        fn_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]
        logger.info(f"Executing tool: {fn_name}")

        result = ""
        try:
            if fn_name == "search_files":
                found = await file_service.search_host(args.get("pattern"), root_dir=args.get("root"))
                result = f"Found files: {found}"
            elif fn_name == "read_file":
                max_chars = args.get("max_chars")
                if not isinstance(max_chars, int) or max_chars <= 0:
                    max_chars = 5000
                res = await file_service.read_host_file(args.get("path"), max_chars=max_chars)
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
                result = (
                    "Error: execute_system_command is disabled for security reasons. "
                    "Use specialized tools such as search_files, read_file, save_file, move_file, or delete_file."
                )
            elif fn_name == "get_system_info":
                result = self._get_system_info_result(configured_model, target_model, active_imagine_model)
            elif fn_name == "web_search":
                result = await self._run_web_search(args.get("query"))
            elif fn_name == "fetch_web_page":
                url = args.get("url")
                async with httpx.AsyncClient(timeout=30.0) as tool_client:
                    resp = await tool_client.get(url, follow_redirects=True)
                    resp.raise_for_status()
                    html = resp.text
                    text = re.sub(r'<(script|style|header|footer|nav).*?>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<.*?>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    result = f"Content from {url}:\n\n{text[:5000]}..."
        except Exception as e:
            result = f"Exception executing tool: {str(e)}"

        if isinstance(result, str) and len(result) > 25000:
            result = await self._map_reduce_summarize(result)

        return result

    def _get_system_info_result(
        self,
        configured_model: str,
        target_model: str,
        active_imagine_model: str,
    ) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        cores = multiprocessing.cpu_count()

        gpu_info = "None detected."
        try:
            gp = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if gp.returncode == 0:
                gpu_info = gp.stdout.strip()
        except Exception:
            pass

        cpu_usage = "Unknown"
        cpu_model = platform.processor() or "Unknown"
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_model = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass

        try:
            cp_usage = subprocess.run(
                "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if cp_usage.returncode == 0:
                val = cp_usage.stdout.strip().replace(',', '.')
                cpu_usage = f"{val}%"
        except Exception:
            pass

        mobo_info = "Unknown"
        try:
            m_vendor = "Unknown"
            m_product = "Unknown"
            if os.path.exists("/sys/class/dmi/id/board_vendor"):
                with open("/sys/class/dmi/id/board_vendor", "r") as f:
                    m_vendor = f.read().strip()
            if os.path.exists("/sys/class/dmi/id/board_name"):
                with open("/sys/class/dmi/id/board_name", "r") as f:
                    m_product = f.read().strip()
            if m_vendor != "Unknown" or m_product != "Unknown":
                mobo_info = f"{m_vendor} {m_product}"
            else:
                dm = subprocess.run("dmidecode -s baseboard-product-name", shell=True, capture_output=True, text=True, timeout=2)
                if dm.returncode == 0:
                    mobo_info = dm.stdout.strip()
        except Exception:
            pass

        return (
            "### [REAL-TIME SYSTEM DATA]\n"
            f"- **Current Time**: {now}\n"
            f"- **Host OS**: {os_info}\n"
            f"- **CPU**: {cpu_model} ({cores} cores), {cpu_usage} usage\n"
            f"- **Motherboard**: {mobo_info}\n"
            f"- **GPU Status**: {gpu_info}\n"
            f"- **Configured Brain Model**: {configured_model}\n"
            f"- **Effective Model For This Request**: {target_model}\n"
            f"- **Vision Model**: {settings.VISION_MODEL}\n"
            f"- **Image Generation Model**: {active_imagine_model}"
        )
