import asyncio
import os
import json
import re
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.brain_service import BrainService

async def test_refusal_and_hallucination_fix():
    print("Testing tool refusal and hallucination fixes...")
    brain = BrainService()
    
    # 1. Test tool filtering
    tools = brain._get_relevant_tools("what is the system time?")
    tool_names = [t["function"]["name"] for t in tools]
    print(f"Relevant tools for 'time': {tool_names}")
    assert "get_system_info" in tool_names

    # 2. Test hallucination detector (Python-style call)
    hallucinated_text = "Checking... execute_system_command(\"date\")"
    calls, cleaned = brain._check_for_hallucinated_tools(hallucinated_text)
    print(f"Hallucination detection - Calls: {calls}")
    print(f"Hallucination detection - Cleaned: '{cleaned}'")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "execute_system_command"
    assert calls[0]["function"]["arguments"]["command"] == "date"
    assert "execute_system_command" not in cleaned

    # 3. Test hallucination detector (JSON style)
    hallucinated_json = "I will check the info: {\"name\": \"get_system_info\", \"arguments\": {}}"
    calls, cleaned = brain._check_for_hallucinated_tools(hallucinated_json)
    print(f"JSON Hallucination - Calls: {calls}")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_system_info"

    print("\n--- ALL TESTS PASSED ---")

if __name__ == "__main__":
    asyncio.run(test_refusal_and_hallucination_fix())
