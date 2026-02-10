from fastapi import APIRouter, HTTPException, Body, Depends, Security
from typing import List, Optional
from app.services.brain_service import brain_service
from app.services.memory_service import memory_service
from app.services.auth_service import auth_service
from app.core.config import settings
from app.api.schemas import * # We'll use specific imports below to be safe

router = APIRouter()

# --- Auth Models ---
from app.api.schemas import (
    UserAuth, Token, UserResponse, ConversationResponse, MessageResponse,
    ChatRequest, ChatResponse, ConversationUpdate, ArchiveUnlock
)

@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "Bipod's nervous system is functional."}

@router.get("/system/config")
async def get_system_config():
    """Returns hardware capabilities and available models."""
    return {
        "hardware": settings.HARDWARE_TARGET,
        "use_gpu": settings.USE_GPU,
        "imagine_models": ["stable-diffusion", "dalle-mini"],
        "default_imagine_model": "stable-diffusion" if settings.USE_GPU else "dalle-mini"
    }

# --- Auth Endpoints ---
@router.post("/auth/signup", response_model=Token)
async def signup(user_data: UserAuth):
    existing_user = await memory_service.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = auth_service.get_password_hash(user_data.password)
    user = await memory_service.create_user(user_data.username, hashed_password)
    
    access_token = auth_service.create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token}

@router.post("/auth/login", response_model=Token)
async def login(user_data: UserAuth):
    user = await memory_service.get_user_by_username(user_data.username)
    if not user or not auth_service.verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = auth_service.create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token}

@router.get("/auth/me")
async def get_me(current_user_id: int = Depends(auth_service.get_current_user)):
    user = await memory_service.get_user_by_id(current_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "username": user.username}

# --- Protected Chat Endpoints ---
@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(user_id: int = Depends(auth_service.get_current_user)):
    return await memory_service.get_conversations(user_id)

@router.post("/conversations")
async def create_conversation(
    title: str = Body(..., embed=True),
    user_id: int = Depends(auth_service.get_current_user)
):
    conv_id = await memory_service.create_conversation(user_id, title)
    return {"id": conv_id, "title": title}

@router.get("/conversations/{conv_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conv_id: str,
    user_id: int = Depends(auth_service.get_current_user)
):
    return await memory_service.get_messages(conv_id, user_id)

@router.patch("/conversations/{conv_id}")
async def update_conversation(
    conv_id: str, 
    update: ConversationUpdate,
    user_id: int = Depends(auth_service.get_current_user)
):
    await memory_service.update_conversation(
        conv_id, 
        user_id,
        title=update.title, 
        is_archived=update.is_archived, 
        password=update.password
    )
    return {"status": "success"}

@router.post("/conversations/{conv_id}/unlock")
async def unlock_conversation(
    conv_id: str, 
    unlock: ArchiveUnlock,
    user_id: int = Depends(auth_service.get_current_user)
):
    is_valid = await memory_service.verify_archive_password(conv_id, user_id, unlock.password)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"status": "success"}

@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: str,
    user_id: int = Depends(auth_service.get_current_user)
):
    await memory_service.delete_conversation(conv_id, user_id)
    return {"status": "success"}

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: int = Depends(auth_service.get_current_user)
):
    """The main entry point for talking to Bipod."""
    try:
        # Check if conversation exists and belongs to user
        conv = await memory_service.get_conversation(request.conversation_id, user_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        response_text = await brain_service.think(
            request.message, 
            request.conversation_id,
            user_id,
            model_id=request.model_id,
            reasoning_mode=request.reasoning_mode,
            imagine_model=request.imagine_model,
            attachments=[a.model_dump() for a in request.attachments] if request.attachments else None
        )
        return ChatResponse(response=response_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_memory(
    conversation_id: str = Body(..., embed=True),
    user_id: int = Depends(auth_service.get_current_user)
):
    # Verify owner
    await memory_service.get_conversation(conversation_id, user_id)
    brain_service.clear_memory(conversation_id)
    return {"status": "success", "message": f"Bipod has forgotten the recent past in {conversation_id}."}


