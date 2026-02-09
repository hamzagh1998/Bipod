from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.brain_service import brain_service

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "Bipod's nervous system is functional."}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """The main entry point for talking to Bipod."""
    try:
        response_text = await brain_service.think(request.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_memory():
    """Resets Bipod's short-term memory."""
    brain_service.clear_memory()
    return {"status": "success", "message": "Bipod has forgotten the recent past."}
