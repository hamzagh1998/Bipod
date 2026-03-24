from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Literal, Optional
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
    imagine_model: Optional[str] = None # 'sdxl-lightning', 'flux-schnell', etc.
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


class StudioProjectCreate(BaseModel):
    title: str


class StudioProjectResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    image_count: int = 0
    cover_image_url: Optional[str] = None


class StudioImageResponse(BaseModel):
    id: str
    project_id: str
    filename: str
    url: str
    mime_type: str
    file_extension: str
    width: Optional[int] = None
    height: Optional[int] = None
    created_at: datetime


class StudioPromptImproveRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    model_type: Optional[str] = None


class StudioPromptImproveResponse(BaseModel):
    prompt: str
    negative_prompt: str


# --- Coach Models ---
class CoachMistakeInput(BaseModel):
    category: str
    detail: str
    severity: str = "medium"
    suggestion: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CoachSessionCreate(BaseModel):
    title: Optional[str] = None
    target_language: str = "English"
    native_language: Optional[str] = None
    cefr_level: str = "A2"
    audio_retention_opt_in: bool = False
    focus_area: Optional[str] = None
    model_id: Optional[str] = None
    voice_profile_id: Optional[str] = None
    llm_device_preference: Literal["auto", "cpu", "cuda"] = "auto"
    tts_device_preference: Literal["auto", "cpu", "cuda"] = "auto"


class CoachSessionSettingsUpdate(BaseModel):
    model_id: Optional[str] = Field(default=None, max_length=120)
    llm_device_preference: Literal["auto", "cpu", "cuda"] = "auto"
    tts_device_preference: Literal["auto", "cpu", "cuda"] = "auto"


class CoachSessionResponse(BaseModel):
    id: str
    user_id: int
    title: str
    target_language: str
    native_language: Optional[str] = None
    cefr_level: str
    audio_retention_opt_in: bool = False
    focus_area: Optional[str] = None
    model_id: Optional[str] = None
    voice_profile_id: Optional[str] = None
    llm_device_preference: Literal["auto", "cpu", "cuda"] = "auto"
    tts_device_preference: Literal["auto", "cpu", "cuda"] = "auto"
    status: str
    created_at: datetime
    updated_at: datetime
    turn_count: int = 0
    mistake_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class CoachTurnCreate(BaseModel):
    transcript: str
    reply: str
    score: int
    correction: Optional[str] = None
    explanation: Optional[str] = None
    model_id: Optional[str] = None
    latency_ms: Optional[int] = None
    mistakes: List[CoachMistakeInput] = Field(default_factory=list)


class CoachMistakeResponse(BaseModel):
    id: int
    session_id: str
    turn_id: int
    user_id: int
    category: str
    detail: str
    severity: str
    suggestion: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CoachTurnResponse(BaseModel):
    id: int
    session_id: str
    user_id: int
    turn_index: int
    transcript: str
    reply: str
    correction: Optional[str] = None
    explanation: Optional[str] = None
    score: Optional[int] = None
    model_id: Optional[str] = None
    latency_ms: Optional[int] = None
    created_at: datetime
    mistakes: List[CoachMistakeResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class CoachProgressSummary(BaseModel):
    user_id: int
    total_sessions: int
    total_turns: int
    total_mistakes: int
    mistake_counts_by_category: Dict[str, int] = Field(default_factory=dict)
    turn_counts_by_model: Dict[str, int] = Field(default_factory=dict)
    active_sessions: int = 0
    latest_session_id: Optional[str] = None
    latest_session_title: Optional[str] = None
    latest_session_turns: int = 0


class CoachModelSelectionResponse(BaseModel):
    selected_model: str
    candidate_models: List[str] = Field(default_factory=list)
    latency_budget_ms: Optional[int] = None


class CoachTtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: Optional[str] = Field(default="English", max_length=60)
    voice_preset: Optional[str] = Field(default="default", max_length=40)
    persona_style: Optional[str] = Field(default=None, max_length=2000)
    tts_provider: Optional[str] = Field(default=None, max_length=40)
    preferred_model: Optional[str] = Field(default=None, min_length=1, max_length=120)
    voice_mode: Optional[Literal["preset", "cloned_profile", "cloned_session"]] = Field(default="preset")
    voice_profile_id: Optional[str] = Field(default=None, max_length=120)
    reference_clip_id: Optional[str] = Field(default=None, max_length=120)
    builtin_voice_id: Optional[str] = Field(default=None, max_length=60)
    session_id: Optional[str] = Field(default=None, max_length=120)
    llm_device_preference: Optional[Literal["auto", "cpu", "cuda"]] = "auto"
    tts_device_preference: Optional[Literal["auto", "cpu", "cuda"]] = "auto"


class CoachTextTurnRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=4000)
    preferred_model: Optional[str] = Field(default=None, min_length=1, max_length=120)
    persona_style: Optional[str] = Field(default=None, max_length=2000)
    llm_device_preference: Optional[Literal["auto", "cpu", "cuda"]] = "auto"


class CoachRuntimePreloadRequest(BaseModel):
    mode: Literal["voice", "text", "idle"] = "voice"


class CoachVoiceReferenceResponse(BaseModel):
    id: str
    title: str
    mime_type: str
    file_size_bytes: int
    language: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CoachVoiceProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    reference_clip_id: str = Field(min_length=1, max_length=120)
    language: Optional[str] = Field(default="English", max_length=60)


class CoachVoiceProfileResponse(BaseModel):
    id: str
    name: str
    provider: str
    language: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CoachBuiltinVoiceResponse(BaseModel):
    id: str
    name: str
    choice_id: str
    voice_mode: str
    voice_preset: str
    provider: str
    is_default: bool = False
    is_available: bool = False
    avatar_data_url: Optional[str] = None


class CoachEvent(BaseModel):
    type: Literal[
        "stt_partial",
        "stt_final",
        "coach_reply",
        "feedback",
        "score",
        "model_fallback",
        "done",
        "error",
    ]
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    message: Optional[str] = None
    detail: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
