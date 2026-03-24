from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response, StreamingResponse

from app.api.schemas import (
    CoachBuiltinVoiceResponse,
    CoachRuntimePreloadRequest,
    CoachSessionCreate,
    CoachSessionSettingsUpdate,
    CoachTextTurnRequest,
    CoachTtsRequest,
    CoachVoiceProfileCreate,
)
from app.services.auth_service import auth_service
from app.services.coach_service import coach_service

router = APIRouter()

_ALLOWED_STREAM_EVENT_TYPES = {
    "stt_partial",
    "stt_final",
    "coach_reply",
    "feedback",
    "score",
    "done",
    "error",
    "model_fallback",
}
_SESSION_TURN_LOCKS: dict[str, asyncio.Lock] = {}
_SESSION_TURN_LOCKS_GUARD = asyncio.Lock()


def _session_not_found(session_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Session not found: {session_id}")


def _encode_ndjson_event(event_type: str, **payload: Any) -> bytes:
    return (
        json.dumps(
            jsonable_encoder({"type": event_type, **payload}),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _validate_stream_event(event: Any) -> Dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("Coach service yielded a non-object event")
    event_type = event.get("type")
    if event_type not in _ALLOWED_STREAM_EVENT_TYPES:
        raise ValueError(f"Unsupported coach stream event type: {event_type}")
    return event


async def _session_turn_lock(session_id: str) -> asyncio.Lock:
    normalized = str(session_id or "").strip()
    if not normalized:
        normalized = "unknown"
    async with _SESSION_TURN_LOCKS_GUARD:
        lock = _SESSION_TURN_LOCKS.get(normalized)
        if lock is None:
            lock = asyncio.Lock()
            _SESSION_TURN_LOCKS[normalized] = lock
        return lock


def _serialize_session(session) -> dict:
    return {
        "id": session.id,
        "user_id": session.user_id,
        "title": session.title,
        "target_language": session.target_language,
        "native_language": session.native_language,
        "cefr_level": session.cefr_level,
        "audio_retention_opt_in": session.audio_retention_opt_in,
        "focus_area": session.focus_area,
        "model_id": session.model_id,
        "llm_device_preference": getattr(session, "llm_device_preference", "auto"),
        "tts_device_preference": getattr(session, "tts_device_preference", "auto"),
        "voice_profile_id": session.voice_profile_id,
        "status": session.status,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "turn_count": len(session.turns),
        "mistake_count": sum(len(turn.mistakes) for turn in session.turns),
    }


@router.post("/sessions")
async def create_session(
    payload: CoachSessionCreate,
    user_id: int = Depends(auth_service.get_current_user),
):
    try:
        created = await coach_service.create_session(
            user_id=user_id,
            title=payload.title,
            target_language=payload.target_language,
            native_language=payload.native_language,
            cefr_level=payload.cefr_level,
            audio_retention_opt_in=payload.audio_retention_opt_in,
            focus_area=payload.focus_area,
            model_id=payload.model_id,
            voice_profile_id=payload.voice_profile_id,
            llm_device_preference=payload.llm_device_preference,
            tts_device_preference=payload.tts_device_preference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sessions = await coach_service.list_sessions(user_id)
    for session in sessions:
        if session["id"] == created.id:
            return session
    raise HTTPException(status_code=500, detail="Session created but could not be serialized")


@router.get("/sessions")
async def list_sessions(user_id: int = Depends(auth_service.get_current_user)):
    return await coach_service.list_sessions(user_id)


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: UUID,
    user_id: int = Depends(auth_service.get_current_user),
):
    session = await coach_service.get_session(str(session_id), user_id)
    if not session:
        raise _session_not_found(str(session_id))
    return _serialize_session(session)


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    user_id: int = Depends(auth_service.get_current_user),
):
    deleted = await coach_service.delete_session(str(session_id), user_id)
    if not deleted:
        raise _session_not_found(str(session_id))
    return {"deleted": True, "session_id": str(session_id)}


@router.patch("/sessions/{session_id}/settings")
async def update_session_settings(
    session_id: UUID,
    payload: CoachSessionSettingsUpdate,
    user_id: int = Depends(auth_service.get_current_user),
):
    updated = await coach_service.update_session_settings(
        session_id=str(session_id),
        user_id=user_id,
        model_id=payload.model_id,
        llm_device_preference=payload.llm_device_preference,
        tts_device_preference=payload.tts_device_preference,
    )
    if not updated:
        raise _session_not_found(str(session_id))
    return _serialize_session(updated)


@router.get("/sessions/{session_id}/turns")
async def list_turns(
    session_id: UUID,
    user_id: int = Depends(auth_service.get_current_user),
):
    turns = await coach_service.list_turns(str(session_id), user_id)
    return [
        {
            "id": turn.id,
            "session_id": turn.session_id,
            "user_id": turn.user_id,
            "turn_index": turn.turn_index,
            "transcript": turn.transcript,
            "reply": turn.reply,
            "correction": turn.correction,
            "explanation": turn.explanation,
            "score": turn.score,
            "model_id": turn.model_id,
            "latency_ms": turn.latency_ms,
            "created_at": turn.created_at,
            "mistakes": [
                {
                    "id": mistake.id,
                    "category": mistake.category,
                    "detail": mistake.detail,
                    "severity": mistake.severity,
                    "suggestion": mistake.suggestion,
                }
                for mistake in turn.mistakes
            ],
        }
        for turn in turns
    ]


@router.get("/sessions/{session_id}/mistakes")
async def list_mistakes(
    session_id: UUID,
    user_id: int = Depends(auth_service.get_current_user),
):
    mistakes = await coach_service.list_mistakes(str(session_id), user_id)
    return [
        {
            "id": mistake.id,
            "session_id": mistake.session_id,
            "turn_id": mistake.turn_id,
            "user_id": mistake.user_id,
            "category": mistake.category,
            "detail": mistake.detail,
            "severity": mistake.severity,
            "suggestion": mistake.suggestion,
            "metadata_json": mistake.metadata_json,
            "created_at": mistake.created_at,
        }
        for mistake in mistakes
    ]


@router.get("/sessions/{session_id}/progress")
async def progress(
    session_id: UUID,
    user_id: int = Depends(auth_service.get_current_user),
):
    session = await coach_service.get_session(str(session_id), user_id)
    if not session:
        raise _session_not_found(str(session_id))
    return await coach_service.progress(user_id=user_id, session_id=str(session_id))


@router.post("/sessions/{session_id}/end")
async def end_session(
    session_id: UUID,
    user_id: int = Depends(auth_service.get_current_user),
):
    summary = await coach_service.end_session(user_id=user_id, session_id=str(session_id))
    if not summary:
        raise _session_not_found(str(session_id))
    return summary


@router.get("/progress")
async def progress_summary(user_id: int = Depends(auth_service.get_current_user)):
    return await coach_service.progress_summary(user_id=user_id)


@router.get("/languages/supported")
async def supported_languages(user_id: int = Depends(auth_service.get_current_user)):
    _ = user_id
    return coach_service.supported_languages()


async def _stream_turn_impl(
    *,
    session_id: str,
    audio: UploadFile,
    transcript_hint: Optional[str],
    preferred_model: Optional[str],
    preferred_asr_model: Optional[str],
    llm_device_preference: Optional[str],
    persona_style: Optional[str],
    user_id: int,
) -> StreamingResponse:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="An audio file name is required")
    if (
        audio.content_type
        and audio.content_type != "application/octet-stream"
        and not audio.content_type.startswith("audio/")
    ):
        raise HTTPException(status_code=400, detail="Upload an audio/* multipart file")

    session = await coach_service.get_session(session_id, user_id)
    if not session:
        raise _session_not_found(session_id)
    turn_lock = await _session_turn_lock(session_id)

    async def event_stream() -> AsyncIterator[bytes]:
        async with turn_lock:
            last_event_type: Optional[str] = None
            try:
                async for event in coach_service.stream_turn(
                    user_id=user_id,
                    session_id=session_id,
                    audio=audio,
                    transcript_hint=transcript_hint,
                    preferred_model=preferred_model,
                    preferred_asr_model=preferred_asr_model,
                    llm_device_preference=llm_device_preference,
                    persona_style=persona_style,
                ):
                    normalized = _validate_stream_event(event)
                    event_type = str(normalized.pop("type"))
                    last_event_type = event_type
                    yield _encode_ndjson_event(event_type, **normalized)
                    if event_type == "done":
                        return
                if last_event_type != "done":
                    yield _encode_ndjson_event("done")
            except ValueError as exc:
                yield _encode_ndjson_event("error", detail=str(exc), status_code=400)
                yield _encode_ndjson_event("done")
            except HTTPException as exc:
                yield _encode_ndjson_event("error", detail=exc.detail, status_code=exc.status_code)
                yield _encode_ndjson_event("done")
            except Exception as exc:
                yield _encode_ndjson_event("error", detail=str(exc), status_code=500)
                yield _encode_ndjson_event("done")

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/sessions/{session_id}/turns/stream")
async def stream_turn_by_path(
    session_id: UUID,
    audio: UploadFile = File(...),
    transcript_hint: Optional[str] = Form(default=None, max_length=4000),
    preferred_model: Optional[str] = Form(default=None, min_length=1, max_length=120),
    preferred_asr_model: Optional[str] = Form(default=None, max_length=40),
    llm_device_preference: Optional[str] = Form(default="auto", max_length=8),
    persona_style: Optional[str] = Form(default=None, max_length=2000),
    user_id: int = Depends(auth_service.get_current_user),
):
    return await _stream_turn_impl(
        session_id=str(session_id),
        audio=audio,
        transcript_hint=transcript_hint,
        preferred_model=preferred_model,
        preferred_asr_model=preferred_asr_model,
        llm_device_preference=llm_device_preference,
        persona_style=persona_style,
        user_id=user_id,
    )


@router.post("/turns/stream")
async def stream_turn(
    audio: UploadFile = File(...),
    session_id: str = Form(..., min_length=1, max_length=120),
    transcript_hint: Optional[str] = Form(default=None, max_length=4000),
    preferred_model: Optional[str] = Form(default=None, min_length=1, max_length=120),
    preferred_asr_model: Optional[str] = Form(default=None, max_length=40),
    llm_device_preference: Optional[str] = Form(default="auto", max_length=8),
    persona_style: Optional[str] = Form(default=None, max_length=2000),
    user_id: int = Depends(auth_service.get_current_user),
):
    return await _stream_turn_impl(
        session_id=str(session_id),
        audio=audio,
        transcript_hint=transcript_hint,
        preferred_model=preferred_model,
        preferred_asr_model=preferred_asr_model,
        llm_device_preference=llm_device_preference,
        persona_style=persona_style,
        user_id=user_id,
    )


@router.post("/turns/text")
async def text_turn(
    payload: CoachTextTurnRequest,
    user_id: int = Depends(auth_service.get_current_user),
):
    session = await coach_service.get_session(str(payload.session_id), user_id)
    if not session:
        raise _session_not_found(str(payload.session_id))
    turn_lock = await _session_turn_lock(str(payload.session_id))
    try:
        async with turn_lock:
            return await coach_service.process_text_turn(
                user_id=user_id,
                session_id=str(payload.session_id),
                text=payload.text,
                preferred_model=payload.preferred_model,
                llm_device_preference=payload.llm_device_preference,
                persona_style=payload.persona_style,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tts")
async def synthesize_tts(
    payload: CoachTtsRequest,
    user_id: int = Depends(auth_service.get_current_user),
):
    try:
        audio_bytes, media_type = await coach_service.synthesize_reply_audio(
            text=payload.text,
            language=payload.language,
            voice_preset=payload.voice_preset,
            persona_style=payload.persona_style,
            tts_provider=payload.tts_provider,
            preferred_model=payload.preferred_model,
            voice_mode=payload.voice_mode,
            voice_profile_id=payload.voice_profile_id,
            reference_clip_id=payload.reference_clip_id,
            builtin_voice_id=payload.builtin_voice_id,
            session_id=payload.session_id,
            llm_device_preference=payload.llm_device_preference,
            tts_device_preference=payload.tts_device_preference,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Coach-TTS": "coach",
        },
    )


@router.get("/tts/status")
async def tts_status(
    warm: bool = False,
    session_id: Optional[str] = None,
    preferred_model: Optional[str] = None,
    preferred_tts_provider: Optional[str] = None,
    llm_device_preference: Optional[str] = "auto",
    tts_device_preference: Optional[str] = "auto",
    user_id: int = Depends(auth_service.get_current_user),
):
    return await coach_service.get_tts_status(
        warm=warm,
        session_id=session_id,
        user_id=user_id,
        preferred_model=preferred_model,
        preferred_tts_provider=preferred_tts_provider,
        llm_device_preference=llm_device_preference,
        tts_device_preference=tts_device_preference,
    )


@router.get("/runtime/status")
async def runtime_status(
    warm: bool = False,
    mode: Optional[str] = None,
    session_id: Optional[str] = None,
    preferred_model: Optional[str] = None,
    preferred_tts_provider: Optional[str] = None,
    llm_device_preference: Optional[str] = "auto",
    tts_device_preference: Optional[str] = "auto",
    user_id: int = Depends(auth_service.get_current_user),
):
    return await coach_service.get_runtime_status(
        warm=warm,
        mode=mode,
        session_id=session_id,
        user_id=user_id,
        preferred_model=preferred_model,
        preferred_tts_provider=preferred_tts_provider,
        llm_device_preference=llm_device_preference,
        tts_device_preference=tts_device_preference,
    )


@router.post("/runtime/preload")
async def runtime_preload(
    payload: CoachRuntimePreloadRequest,
    user_id: int = Depends(auth_service.get_current_user),
):
    _ = user_id
    return await coach_service.preload_runtime(mode=payload.mode)


@router.post("/voices/reference")
async def upload_voice_reference(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None, max_length=100),
    language: Optional[str] = Form(default="English", max_length=60),
    user_id: int = Depends(auth_service.get_current_user),
):
    try:
        return await coach_service.save_voice_reference(
            user_id=user_id,
            file=file,
            title=title,
            language=language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/voices/profiles")
async def create_voice_profile(
    payload: CoachVoiceProfileCreate,
    user_id: int = Depends(auth_service.get_current_user),
):
    try:
        return await coach_service.create_voice_profile(
            user_id=user_id,
            name=payload.name,
            reference_clip_id=payload.reference_clip_id,
            language=payload.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/voices/profiles")
async def list_voice_profiles(user_id: int = Depends(auth_service.get_current_user)):
    return await coach_service.list_voice_profiles(user_id=user_id)


@router.get("/voices/library", response_model=list[CoachBuiltinVoiceResponse])
async def list_builtin_voices(user_id: int = Depends(auth_service.get_current_user)):
    return await coach_service.list_builtin_voices()


@router.delete("/voices/profiles/{profile_id}")
async def delete_voice_profile(
    profile_id: str,
    user_id: int = Depends(auth_service.get_current_user),
):
    deleted = await coach_service.delete_voice_profile(profile_id=profile_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice profile not found: {profile_id}")
    return {"deleted": True, "profile_id": profile_id}
