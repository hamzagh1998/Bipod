import asyncio
import io
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

import app.api.coach as coach_module


class _FakeUploadFile:
    def __init__(self, data: bytes, filename: str = "turn.wav", content_type: str = "audio/wav"):
        self._buffer = io.BytesIO(data)
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    async def seek(self, offset: int) -> None:
        self._buffer.seek(offset)


def _make_upload_file(data: bytes, filename: str = "turn.wav", content_type: str = "audio/wav") -> _FakeUploadFile:
    return _FakeUploadFile(data=data, filename=filename, content_type=content_type)


async def _collect_stream_chunks(streaming_response):
    chunks = []
    async for chunk in streaming_response.body_iterator:
        chunks.append(chunk)
    return chunks


def test_coach_session_endpoints_delegate_to_service(monkeypatch):
    session_id = str(uuid4())
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fake_session = SimpleNamespace(
        id=session_id,
        user_id=42,
        title="Focus Session",
        target_language="English",
        native_language="Arabic",
        cefr_level="B1",
        audio_retention_opt_in=False,
        focus_area="conversation",
        model_id="qwen3:8b",
        status="active",
        created_at=now,
        updated_at=now,
        turns=[],
    )

    async def fake_create_session(**kwargs):
        assert kwargs["user_id"] == 42
        assert kwargs["title"] == "Focus Session"
        return fake_session

    async def fake_list_sessions(user_id):
        assert user_id == 42
        return [
            {
                "id": session_id,
                "user_id": 42,
                "title": "Focus Session",
                "target_language": "English",
                "native_language": "Arabic",
                "cefr_level": "B1",
                "audio_retention_opt_in": False,
                "focus_area": "conversation",
                "model_id": "qwen3:8b",
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "turn_count": 0,
                "mistake_count": 0,
            }
        ]

    async def fake_get_session(requested_session_id, user_id):
        assert user_id == 42
        if requested_session_id != session_id:
            return None
        return fake_session

    monkeypatch.setattr(coach_module.coach_service, "create_session", fake_create_session)
    monkeypatch.setattr(coach_module.coach_service, "list_sessions", fake_list_sessions)
    monkeypatch.setattr(coach_module.coach_service, "get_session", fake_get_session)

    created = asyncio.run(
        coach_module.create_session(
            coach_module.CoachSessionCreate(
                title="Focus Session",
                target_language="English",
                native_language="Arabic",
                cefr_level="B1",
            ),
            user_id=42,
        )
    )
    assert created["id"] == session_id
    assert created["title"] == "Focus Session"
    assert created["target_language"] == "English"

    sessions = asyncio.run(coach_module.list_sessions(user_id=42))
    assert sessions == [created]

    fetched = asyncio.run(coach_module.get_session(session_id=session_id, user_id=42))  # type: ignore[arg-type]
    assert fetched["id"] == session_id


def test_coach_progress_and_lists(monkeypatch):
    session_id = str(uuid4())
    expected_session_id = session_id
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fake_turn = SimpleNamespace(
        id=1,
        session_id=session_id,
        user_id=9,
        turn_index=1,
        transcript="Explain the problem.",
        reply="Explain the problem in one sentence.",
        correction="Add an article before problem.",
        explanation="Target + native explanation",
        score=91,
        model_id="qwen3:8b",
        latency_ms=200,
        created_at=now,
        mistakes=[
            SimpleNamespace(id=11, category="grammar", detail="Missing article", severity="medium", suggestion="Use 'the problem'."),
        ],
    )
    fake_mistake = SimpleNamespace(
        id=11,
        session_id=session_id,
        turn_id=1,
        user_id=9,
        category="grammar",
        detail="Missing article",
        severity="medium",
        suggestion="Use 'the problem'.",
        metadata_json=None,
        created_at=now,
    )

    async def fake_get_session(requested_session_id, user_id):
        if requested_session_id == expected_session_id and user_id == 9:
            return SimpleNamespace(id=expected_session_id, turns=[])
        return None

    async def fake_list_turns(requested_session_id, user_id):
        assert requested_session_id == expected_session_id
        assert user_id == 9
        return [fake_turn]

    async def fake_list_mistakes(requested_session_id, user_id):
        assert requested_session_id == expected_session_id
        assert user_id == 9
        return [fake_mistake]

    async def fake_progress(user_id, session_id):
        assert session_id == expected_session_id
        assert user_id == 9
        return {"session_id": expected_session_id, "turn_count": 1, "mistake_count": 1, "average_score": 91.0, "latest_score": 91}

    monkeypatch.setattr(coach_module.coach_service, "get_session", fake_get_session)
    monkeypatch.setattr(coach_module.coach_service, "list_turns", fake_list_turns)
    monkeypatch.setattr(coach_module.coach_service, "list_mistakes", fake_list_mistakes)
    monkeypatch.setattr(coach_module.coach_service, "progress", fake_progress)

    turns = asyncio.run(coach_module.list_turns(session_id=session_id, user_id=9))  # type: ignore[arg-type]
    mistakes = asyncio.run(coach_module.list_mistakes(session_id=session_id, user_id=9))  # type: ignore[arg-type]
    progress = asyncio.run(coach_module.progress(session_id=session_id, user_id=9))  # type: ignore[arg-type]

    assert turns[0]["transcript"] == "Explain the problem."
    assert turns[0]["mistakes"][0]["category"] == "grammar"
    assert mistakes[0]["detail"] == "Missing article"
    assert progress["average_score"] == 91.0


def test_coach_stream_turn_emits_ndjson_events(monkeypatch):
    session_id = str(uuid4())
    expected_session_id = session_id
    seen_audio_reads = []

    async def fake_get_session(requested_session_id, user_id):
        assert requested_session_id == expected_session_id
        assert user_id == 7
        return SimpleNamespace(id=expected_session_id)

    async def fake_stream_turn(
        *,
        user_id,
        session_id,
        audio,
        transcript_hint=None,
        preferred_model=None,
    ):
        assert user_id == 7
        assert session_id == expected_session_id
        assert transcript_hint == "coach me"
        assert preferred_model == "small-model"
        seen_audio_reads.append(await audio.read())
        await audio.seek(0)
        yield {"type": "stt_partial", "text": "coach"}
        yield {"type": "stt_final", "text": "coach me"}
        yield {"type": "model_fallback", "requested_model": "small-model", "selected_model": "qwen2.5:7b"}
        yield {"type": "coach_reply", "text": "coach me better"}
        yield {"type": "feedback", "summary": "clear", "mistakes": []}
        yield {"type": "score", "value": 94}

    monkeypatch.setattr(coach_module.coach_service, "get_session", fake_get_session)
    monkeypatch.setattr(coach_module.coach_service, "stream_turn", fake_stream_turn)

    response = asyncio.run(
        coach_module.stream_turn(
            session_id=session_id,  # type: ignore[arg-type]
            audio=_make_upload_file(b"audio-bytes"),
            transcript_hint="coach me",
            preferred_model="small-model",
            user_id=7,
        )
    )

    raw_body = b"".join(asyncio.run(_collect_stream_chunks(response))).decode("utf-8")
    events = [json.loads(line) for line in raw_body.splitlines() if line]

    assert seen_audio_reads == [b"audio-bytes"]
    assert [event["type"] for event in events] == [
        "stt_partial",
        "stt_final",
        "model_fallback",
        "coach_reply",
        "feedback",
        "score",
        "done",
    ]
    assert events[2]["selected_model"] == "qwen2.5:7b"
    assert events[3]["text"] == "coach me better"
    assert events[-1] == {"type": "done"}


def test_coach_stream_turn_returns_clear_errors(monkeypatch):
    session_id = str(uuid4())

    async def fake_get_session(requested_session_id, user_id):
        return None

    monkeypatch.setattr(coach_module.coach_service, "get_session", fake_get_session)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            coach_module.stream_turn(
                session_id=session_id,  # type: ignore[arg-type]
                audio=_make_upload_file(b"audio-bytes"),
                user_id=1,
            )
        )

    assert exc_info.value.status_code == 404
    assert "Session not found" in exc_info.value.detail
