import asyncio
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.brain_service import BrainService

async def test_system_tool():
    print("Testing get_system_info tool...")
    brain = BrainService()
    
    # Mocking httpx responses
    mock_resp1 = MagicMock()
    mock_resp1.status_code = 200
    mock_resp1.raise_for_status = MagicMock()
    mock_resp1.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "Checking hardware...",
            "tool_calls": [{
                "id": "call_123",
                "function": {"name": "get_system_info", "arguments": {}}
            }]
        }
    }
    
    mock_resp2 = MagicMock()
    mock_resp2.status_code = 200
    mock_resp2.raise_for_status = MagicMock()
    mock_resp2.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "You are running on a powerful machine."
        }
    }
    
    mock_memory = AsyncMock()
    mock_memory.add_message.return_value = MagicMock(id=1)
    mock_memory.get_messages.return_value = []
    
    mock_vector = AsyncMock()
    mock_vector.search_memories.return_value = []
    mock_vector.add_memory.return_value = None

    with patch("httpx.AsyncClient.post") as mock_post, \
         patch("app.services.brain_service.memory_service", mock_memory), \
         patch("app.services.brain_service.vector_service", mock_vector):
        
        mock_post.side_effect = [mock_resp1, mock_resp2]
        
        # Test 1: Hardware inquiry
        result = await brain.think("How is the CPU and GPU doing?", "conv_1", 1)
        
        print(f"Final Output:\n{result}")
        if "Checking hardware..." in result and "You are running on a powerful machine." in result:
             print("Success: System tool triggered and thinking preserved!")
        else:
             print("Failure: Output did not match expected structure.")
             exit(1)

if __name__ == "__main__":
    asyncio.run(test_system_tool())
