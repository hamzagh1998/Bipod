from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
from app.services.brain_service import brain_service
from app.services.memory_service import memory_service

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    conversation_id: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    is_archived: Optional[bool] = None
    password: Optional[str] = None

class ArchiveUnlock(BaseModel):
    password: str

@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "Bipod's nervous system is functional."}

@router.get("/conversations")
async def list_conversations():
    return await memory_service.get_conversations()

@router.post("/conversations")
async def create_conversation(title: str = Body(..., embed=True)):
    conv_id = await memory_service.create_conversation(title)
    return {"id": conv_id, "title": title}

@router.get("/conversations/{conv_id}/messages")
async def get_messages(conv_id: str):
    return await memory_service.get_messages(conv_id)

@router.patch("/conversations/{conv_id}")
async def update_conversation(conv_id: str, update: ConversationUpdate):
    await memory_service.update_conversation(
        conv_id, 
        title=update.title, 
        is_archived=update.is_archived, 
        password=update.password
    )
    return {"status": "success"}

@router.post("/conversations/{conv_id}/unlock")
async def unlock_conversation(conv_id: str, unlock: ArchiveUnlock):
    is_valid = await memory_service.verify_archive_password(conv_id, unlock.password)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"status": "success"}

@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    await memory_service.delete_conversation(conv_id)
    return {"status": "success"}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """The main entry point for talking to Bipod."""
    try:
        # Check if conversation is archived & locked
        conv = await memory_service.get_conversation(request.conversation_id)
        if conv and conv.is_archived and conv.archive_password_hash:
            # In a real app, we'd check session/token here. 
            # For Bipod, we'll assume the frontend handles the unlock flow before sending chat.
            pass

        response_text = await brain_service.think(request.message, request.conversation_id)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_memory(conversation_id: str = Body(..., embed=True)):
    """Resets Bipod's short-term memory for a specific conversation."""
    brain_service.clear_memory(conversation_id)
    return {"status": "success", "message": f"Bipod has forgotten the recent past in {conversation_id}."}


