import asyncio
import json

import pytest
from fastapi import HTTPException

import app.api as api_module
from app.api.schemas import ChatRequest


async def _collect_stream_chunks(streaming_response):
    chunks = []
    async for chunk in streaming_response.body_iterator:
        chunks.append(chunk)
    return chunks


def test_chat_stream_emits_progress_and_final_response(monkeypatch):
    async def fake_get_conversation(conv_id, user_id):
        return object()

    async def fake_think(
        user_input,
        conversation_id,
        user_id,
        model_id=None,
        reasoning_mode=None,
        imagine_model=None,
        attachments=None,
        progress_callback=None,
    ):
        await progress_callback(
            "status",
            {
                "label": "Routing the request",
                "detail": "Deciding between direct chat, tools, and handoffs.",
            },
        )
        await progress_callback(
            "tool_call",
            {
                "label": "Searching the web",
                "detail": "Looking up: weather in Tunis",
            },
        )
        return "Sunny today."

    monkeypatch.setattr(api_module.memory_service, "get_conversation", fake_get_conversation)
    monkeypatch.setattr(api_module.brain_service, "think", fake_think)

    response = asyncio.run(
        api_module.chat_stream(
            ChatRequest(message="what's the weather", conversation_id="conv-1"),
            user_id=123,
        )
    )
    raw_body = b"".join(asyncio.run(_collect_stream_chunks(response))).decode("utf-8")
    events = [json.loads(line) for line in raw_body.splitlines() if line]

    assert events == [
        {
            "type": "status",
            "label": "Routing the request",
            "detail": "Deciding between direct chat, tools, and handoffs.",
        },
        {
            "type": "tool_call",
            "label": "Searching the web",
            "detail": "Looking up: weather in Tunis",
        },
        {
            "type": "response",
            "text": "Sunny today.",
        },
        {
            "type": "done",
        },
    ]


def test_chat_stream_requires_existing_conversation(monkeypatch):
    async def fake_get_conversation(conv_id, user_id):
        return None

    monkeypatch.setattr(api_module.memory_service, "get_conversation", fake_get_conversation)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            api_module.chat_stream(
                ChatRequest(message="hello", conversation_id="missing-conv"),
                user_id=123,
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Conversation not found"
