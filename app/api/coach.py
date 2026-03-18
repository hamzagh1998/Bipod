from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.api.schemas import CoachSessionCreate
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
    created = await coach_service.create_session(
        user_id=user_id,
        title=payload.title,
        target_language=payload.target_language,
        native_language=payload.native_language,
        cefr_level=payload.cefr_level,
        audio_retention_opt_in=payload.audio_retention_opt_in,
        focus_area=payload.focus_area,
        model_id=payload.model_id,
    )
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


@router.get("/progress")
async def progress_summary(user_id: int = Depends(auth_service.get_current_user)):
    return await coach_service.progress_summary(user_id=user_id)


async def _stream_turn_impl(
    *,
    session_id: str,
    audio: UploadFile,
    transcript_hint: Optional[str],
    preferred_model: Optional[str],
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

    async def event_stream() -> AsyncIterator[bytes]:
        last_event_type: Optional[str] = None
        try:
            async for event in coach_service.stream_turn(
                user_id=user_id,
                session_id=session_id,
                audio=audio,
                transcript_hint=transcript_hint,
                preferred_model=preferred_model,
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
    user_id: int = Depends(auth_service.get_current_user),
):
    return await _stream_turn_impl(
        session_id=str(session_id),
        audio=audio,
        transcript_hint=transcript_hint,
        preferred_model=preferred_model,
        user_id=user_id,
    )


@router.post("/turns/stream")
async def stream_turn(
    audio: UploadFile = File(...),
    session_id: str = Form(..., min_length=1, max_length=120),
    transcript_hint: Optional[str] = Form(default=None, max_length=4000),
    preferred_model: Optional[str] = Form(default=None, min_length=1, max_length=120),
    user_id: int = Depends(auth_service.get_current_user),
):
    return await _stream_turn_impl(
        session_id=str(session_id),
        audio=audio,
        transcript_hint=transcript_hint,
        preferred_model=preferred_model,
        user_id=user_id,
    )
