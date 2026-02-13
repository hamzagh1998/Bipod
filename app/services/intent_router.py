import logging
import os
import asyncio
from typing import List, Optional, Dict
import numpy as np
from app.services.vector_service import vector_service
from app.core.logger import get_logger

logger = get_logger("bipod.services.intent")

class IntentRouter:
    """Ollama-powered intent classifier for Bipod.
    Uses Bipod's existing vector engine to classify prompts into tool categories.
    """
    
    def __init__(self):
        # Route definitions with example phrases
        self.routes = {
            "web_search": [
                "who is the current prime minister of the UK",
                "latest news about OpenAI",
                "current price of bitcoin today",
                "who won the game last night",
                "current weather",
                "latest world updates",
                "news headlines",
                "what's happening in the news",
                "recent events in space",
                "current news updates",
                "currency conversion rate",
                "how much is 1 usd in eur",
                "exchange rate for dollar",
                "convert money",
                "search for it",
                "google it",
                "use the web",
                "check online",
                "look it up",
                "research a topic",
                "find details about",
                "internet search",
                "make use the internet",
                "get up to date data",
                "pros and cons of",
                "capabilities and features",
            ],
            "image_generation": [
                "draw me a futuristic city",
                "generate an image",
                "create a painting of a cat",
                "make a picture",
                "visualize a dragon",
                "imagine a post-apocalyptic library",
                "draw a sketch",
                "paint digital art",
                "create a cool wallpaper",
                "generate a photo of a keyboard",
                "draw an rgb keyboard",
                "make a better variation",
                "create another version with more detail",
                "improve the quality of the image",
                "generate a higher resolution version",
            ],
            "system_info": [
                "what time is it now",
                "current time and date",
                "show me system statistics",
                "how much CPU usage",
                "what's the GPU status",
                "show memory usage",
                "what OS is this",
                "give me system info",
                "check hardware health",
                "current clock time",
                "what's my system kernel",
                "show linux version",
                "check my hardware usage",
                "what is the time",
                "time on the host machine",
                "what is my motherboard",
                "check pcie support",
                "motherboard model and manufacturer",
                "hardware specifications",
                "what chipset am i using",
            ],
            "file_operation": [
                "open the existing file",
                "read the contents of the document",
                "find a file on the disk",
                "list all files in the current folder",
                "save my current progress to a file",
                "write this text into a file named",
                "delete the specific file",
                "rename the file on my system",
                "show inside the project directory",
            ],
            "coding": [
                "generate python code",
                "write a bash script",
                "how do I use recursion",
                "help me debug this function",
                "explain this code snippet",
                "write a react component",
                "javascript async await tutorial",
                "write a loop in c++",
            ],
            "shell_command": [
                "run the command ls -la",
                "execute grep",
                "install package",
                "restart the docker",
                "ping the network",
                "run a terminal command",
                "execute shell script",
            ],
            "vision_analysis": [
                "what is in this image",
                "describe the picture",
                "analyze the screenshot",
                "look at this photo",
                "identify objects in image",
                "what do you see in the file",
            ],
            "fetch_web_page": [
                "summarize this URL",
                "read the link",
                "what does this website say",
                "extract text from page",
                "visit the website",
            ],
            "troubleshooting": [
                "check this",
                "what's the issue",
                "check these command and tell me what's the issue",
                "fix this error",
                "why is this failing",
                "debug this stack trace",
                "what does this error mean",
                "analyze this log",
                "help me fix this bug",
                "why is it not connecting",
                "resolve this exception",
                "troubleshoot this problem",
            ]
        }
        
        # Pre-calculated embeddings for the routes
        self.route_vectors = {}
        self._initialized = False

    async def _ensure_initialized(self):
        """Lazy-init: Calculate embeddings for all routes on first use."""
        if self._initialized:
            return

        logger.info("Initializing Bipod Intent Router (Semantic Matcher)...")
        try:
            for route_name, utterances in self.routes.items():
                vectors = []
                for text in utterances:
                    emb = await vector_service._get_embedding(text)
                    if emb:
                        vectors.append(emb)
                
                if vectors:
                    # Average embedding for the route "centroid"
                    self.route_vectors[route_name] = np.mean(vectors, axis=0)
            
            self._initialized = True
            logger.info("Bipod Intent Router initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Intent Router: {e}")

    async def classify(self, user_input: str) -> Optional[str]:
        """Classifies user input using cosine similarity against route centroids."""
        await self._ensure_initialized()
        
        if not self.route_vectors:
            return None
            
        try:
            query_emb = await vector_service._get_embedding(user_input)
            if not query_emb:
                return None
                
            best_route = None
            max_sim = -1.0
            
            # Normalize query vector for cosine similarity
            query_vec = np.array(query_emb)
            query_norm = np.linalg.norm(query_vec)
            
            for route_name, route_vec in self.route_vectors.items():
                # Cosine similarity calculation
                dot_product = np.dot(query_vec, route_vec)
                route_norm = np.linalg.norm(route_vec)
                similarity = dot_product / (query_norm * route_norm)
                
                if similarity > max_sim:
                    max_sim = similarity
                    best_route = route_name
            
            logger.info(f"Intent classified: '{best_route}' (Confidence: {max_sim:.2f})")
            
            # Semantic threshold: if similarity is too low, treat as pure chat.
            # 0.52 is a balanced threshold for nomic-embed-text
            if max_sim > 0.52:
                return best_route
            return None
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return None

    def get_tools_for_intent(self, intent: str, all_tools: List[Dict]) -> List[Dict]:
        """Filters the available tools based on the classified intent."""
        intent_map = {
            "web_search": ["web_search", "fetch_web_page"],
            "fetch_web_page": ["fetch_web_page", "web_search"],
            "image_generation": ["generate_image"],
            # System info often needs to execute commands or read files to get data
            "system_info": ["get_system_info", "execute_system_command", "read_file", "web_search"],
            # File operations might need search if user asks "find a file about X" (not just filename)
            "file_operation": ["read_file", "save_file", "search_files", "execute_system_command", "move_file", "delete_file", "organize_files", "web_search"],
            # Coding needs context (read files), execution (run scripts), and docs (web search)
            "coding": ["save_file", "read_file", "search_files", "move_file", "organize_files", "execute_system_command", "web_search"],
            # Shell commands almost always need search for flags/errors and reading logs
            "shell_command": ["execute_system_command", "search_files", "get_system_info", "organize_files", "read_file", "web_search"],
            "vision_analysis": ["analyze_image_file", "web_search"],
            # Troubleshooting is a mix of all: searching docs, reading code/logs, executing diagnostic commands
            "troubleshooting": ["web_search", "read_file", "search_files", "execute_system_command", "get_system_info", "fetch_web_page"]
        }
        
        allowed_tool_names = intent_map.get(intent, [])
        if not allowed_tool_names:
            return []
            
        return [t for t in all_tools if t["function"]["name"] in allowed_tool_names]

# Singleton instance
intent_router = IntentRouter()
