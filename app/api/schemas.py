from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# --- Auth Models ---
class UserAuth(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: int
    username: str

    class Config:
        from_attributes = True

# --- Chat Models ---
class Attachment(BaseModel):
    type: str # 'image' or 'pdf'
    content: str # b64
    name: Optional[str] = None

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    attachments: Optional[List[Attachment]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    id: str
    title: str
    is_archived: bool
    created_at: datetime

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str
    conversation_id: str
    model_id: Optional[str] = None
    reasoning_mode: Optional[str] = None
    imagine_model: Optional[str] = None # 'stable-diffusion' or 'dalle-mini'
    attachments: Optional[List[Attachment]] = None

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    is_archived: Optional[bool] = None
    password: Optional[str] = None

class ArchiveUnlock(BaseModel):
    password: str
