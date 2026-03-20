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
        voice_profile_id=None,
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
                "voice_profile_id": None,
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
        preferred_asr_model=None,
        persona_style=None,
    ):
        assert user_id == 7
        assert session_id == expected_session_id
        assert transcript_hint == "coach me"
        assert preferred_model == "small-model"
        assert preferred_asr_model == "accurate"
        assert persona_style == "calm tactical persona"
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
                preferred_asr_model="accurate",
                persona_style="calm tactical persona",
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


def test_coach_end_session_returns_summary(monkeypatch):
    session_id = str(uuid4())
    expected_session_id = session_id

    async def fake_end_session(user_id, session_id):
        assert user_id == 5
        assert session_id == expected_session_id
        return {
            "session_id": expected_session_id,
            "status": "completed",
            "subject": "Travel",
            "turn_count": 3,
            "average_score": 84.3,
            "latest_score": 90,
            "mistake_counts_by_category": {"grammar": 2},
            "feedback_summary": "Good flow, improve grammar precision.",
            "strengths": ["Stayed on topic"],
            "improvement_points": ["Use cleaner tense consistency"],
        }

    monkeypatch.setattr(coach_module.coach_service, "end_session", fake_end_session)

    summary = asyncio.run(coach_module.end_session(session_id=session_id, user_id=5))  # type: ignore[arg-type]
    assert summary["status"] == "completed"
    assert summary["turn_count"] == 3
    assert summary["average_score"] == 84.3


def test_coach_delete_session_endpoint(monkeypatch):
    session_id = str(uuid4())
    expected_session_id = session_id

    async def fake_delete_session(session_id, user_id):
        assert user_id == 12
        assert session_id == expected_session_id
        return True

    monkeypatch.setattr(coach_module.coach_service, "delete_session", fake_delete_session)

    payload = asyncio.run(coach_module.delete_session(session_id=session_id, user_id=12))  # type: ignore[arg-type]
    assert payload == {"deleted": True, "session_id": expected_session_id}


def test_coach_tts_endpoint_returns_audio(monkeypatch):
    captured = {}

    async def fake_synthesize_reply_audio(
        *,
        text,
        language,
        voice_preset,
        persona_style=None,
        tts_provider=None,
        preferred_model=None,
        voice_mode=None,
        voice_profile_id=None,
        reference_clip_id=None,
        builtin_voice_id=None,
        user_id=None,
    ):
        captured["text"] = text
        captured["language"] = language
        captured["voice_preset"] = voice_preset
        captured["persona_style"] = persona_style
        captured["tts_provider"] = tts_provider
        captured["preferred_model"] = preferred_model
        captured["voice_mode"] = voice_mode
        captured["voice_profile_id"] = voice_profile_id
        captured["reference_clip_id"] = reference_clip_id
        captured["builtin_voice_id"] = builtin_voice_id
        captured["user_id"] = user_id
        return b"RIFF....WAVE", "audio/wav"

    monkeypatch.setattr(coach_module.coach_service, "synthesize_reply_audio", fake_synthesize_reply_audio)

    response = asyncio.run(
        coach_module.synthesize_tts(
            coach_module.CoachTtsRequest(
                text="Hello there",
                language="English",
                voice_preset="anby",
                persona_style="Anby",
                tts_provider="cosyvoice",
                preferred_model="qwen3:8b",
                voice_mode="preset",
            ),
            user_id=3,
        )
    )

    assert captured == {
        "text": "Hello there",
        "language": "English",
        "voice_preset": "anby",
        "persona_style": "Anby",
        "tts_provider": "cosyvoice",
        "preferred_model": "qwen3:8b",
        "voice_mode": "preset",
        "voice_profile_id": None,
        "reference_clip_id": None,
        "builtin_voice_id": None,
        "user_id": 3,
    }
    assert response.media_type == "audio/wav"
    assert response.body == b"RIFF....WAVE"


def test_coach_tts_status_endpoint(monkeypatch):
    captured = {}

    async def fake_get_tts_status(*, warm, preferred_model=None, preferred_tts_provider=None):
        captured["warm"] = warm
        captured["preferred_model"] = preferred_model
        captured["preferred_tts_provider"] = preferred_tts_provider
        return {
            "ok": True,
            "engine": "cosyvoice",
            "provider": "cosyvoice",
            "ready": False,
            "state": "downloading",
            "detail": "Downloading voice model assets.",
            "model_id": "iic/CosyVoice-300M",
            "loaded_model_id": "",
            "warmup_active": True,
            "updated_at": 123.4,
        }

    monkeypatch.setattr(coach_module.coach_service, "get_tts_status", fake_get_tts_status)

    payload = asyncio.run(
        coach_module.tts_status(
            warm=True,
            preferred_model="qwen3:8b",
            preferred_tts_provider="openvoice",
            user_id=3,
        )
    )

    assert captured == {
        "warm": True,
        "preferred_model": "qwen3:8b",
        "preferred_tts_provider": "openvoice",
    }
    assert payload["engine"] == "cosyvoice"
    assert payload["ready"] is False
    assert payload["state"] == "downloading"


def test_coach_runtime_endpoints(monkeypatch):
    captured = {}

    async def fake_get_runtime_status(*, warm, mode, preferred_model=None, preferred_tts_provider=None):
        captured["status"] = {
            "warm": warm,
            "mode": mode,
            "preferred_model": preferred_model,
            "preferred_tts_provider": preferred_tts_provider,
        }
        return {"ok": True, "ready": False, "mode": mode or "voice", "state": "warming", "components": {}}

    async def fake_preload_runtime(*, mode):
        captured["preload"] = {"mode": mode}
        return {"ok": True, "ready": True, "mode": mode, "state": "ready", "components": {}}

    monkeypatch.setattr(coach_module.coach_service, "get_runtime_status", fake_get_runtime_status)
    monkeypatch.setattr(coach_module.coach_service, "preload_runtime", fake_preload_runtime)

    status_payload = asyncio.run(
        coach_module.runtime_status(
            warm=True,
            mode="text",
            preferred_model="llama3.2:3b",
            preferred_tts_provider="openvoice",
            user_id=3,
        )
    )
    preload_payload = asyncio.run(
        coach_module.runtime_preload(
            payload=coach_module.CoachRuntimePreloadRequest(mode="voice"),
            user_id=3,
        )
    )

    assert captured["status"] == {
        "warm": True,
        "mode": "text",
        "preferred_model": "llama3.2:3b",
        "preferred_tts_provider": "openvoice",
    }
    assert captured["preload"] == {"mode": "voice"}
    assert status_payload["state"] == "warming"
    assert preload_payload["state"] == "ready"


def test_coach_text_turn_endpoint(monkeypatch):
    session_id = str(uuid4())
    captured = {}

    async def fake_get_session(requested_session_id, user_id):
        assert requested_session_id == session_id
        assert user_id == 7
        return SimpleNamespace(id=session_id)

    async def fake_process_text_turn(*, user_id, session_id, text, preferred_model, persona_style):
        captured.update(
            {
                "user_id": user_id,
                "session_id": session_id,
                "text": text,
                "preferred_model": preferred_model,
                "persona_style": persona_style,
            }
        )
        return {"id": 1, "session_id": session_id, "transcript": text, "reply": "Good. Keep going.", "score": 82, "mistakes": []}

    monkeypatch.setattr(coach_module.coach_service, "get_session", fake_get_session)
    monkeypatch.setattr(coach_module.coach_service, "process_text_turn", fake_process_text_turn)

    payload = asyncio.run(
        coach_module.text_turn(
            payload=coach_module.CoachTextTurnRequest(
                session_id=session_id,
                text="I usually wake up at six and review my tasks.",
                preferred_model="qwen3:8b",
                persona_style="calm tactical persona",
            ),
            user_id=7,
        )
    )
    assert payload["score"] == 82
    assert captured["preferred_model"] == "qwen3:8b"


def test_coach_supported_languages_endpoint(monkeypatch):
    expected = [
        {
            "code": "en",
            "name": "English",
            "asr_supported": True,
            "tts_supported": True,
            "languagetool_supported": False,
            "selectable": True,
            "is_default": True,
        }
    ]

    monkeypatch.setattr(coach_module.coach_service, "supported_languages", lambda: expected)
    payload = asyncio.run(coach_module.supported_languages(user_id=5))
    assert payload == expected


def test_coach_voice_profile_endpoints(monkeypatch):
    now = datetime(2026, 3, 19, tzinfo=timezone.utc)
    uploaded = {
        "id": "sample-1",
        "title": "sample.wav",
        "mime_type": "audio/wav",
        "file_size_bytes": 100,
        "language": "English",
        "created_at": now,
    }
    created_profile = {
        "id": "profile-1",
        "name": "My Voice",
        "provider": "cosyvoice",
        "language": "English",
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }

    async def fake_save_voice_reference(*, user_id, file, title, language):
        assert user_id == 7
        assert language == "English"
        assert title == "sample.wav"
        assert await file.read() == b"wav-bytes"
        await file.seek(0)
        return uploaded

    async def fake_create_voice_profile(*, user_id, name, reference_clip_id, language):
        assert user_id == 7
        assert name == "My Voice"
        assert reference_clip_id == "sample-1"
        assert language == "English"
        return created_profile

    async def fake_list_voice_profiles(*, user_id):
        assert user_id == 7
        return [created_profile]

    async def fake_delete_voice_profile(*, profile_id, user_id):
        assert profile_id == "profile-1"
        assert user_id == 7
        return True

    monkeypatch.setattr(coach_module.coach_service, "save_voice_reference", fake_save_voice_reference)
    monkeypatch.setattr(coach_module.coach_service, "create_voice_profile", fake_create_voice_profile)
    monkeypatch.setattr(coach_module.coach_service, "list_voice_profiles", fake_list_voice_profiles)
    monkeypatch.setattr(coach_module.coach_service, "delete_voice_profile", fake_delete_voice_profile)

    uploaded_payload = asyncio.run(
        coach_module.upload_voice_reference(
            file=_make_upload_file(b"wav-bytes"),
            title="sample.wav",
            language="English",
            user_id=7,
        )
    )
    assert uploaded_payload["id"] == "sample-1"

    created_payload = asyncio.run(
        coach_module.create_voice_profile(
            payload=coach_module.CoachVoiceProfileCreate(
                name="My Voice",
                reference_clip_id="sample-1",
                language="English",
            ),
            user_id=7,
        )
    )
    assert created_payload["id"] == "profile-1"

    listed_payload = asyncio.run(coach_module.list_voice_profiles(user_id=7))
    assert len(listed_payload) == 1
    assert listed_payload[0]["name"] == "My Voice"

    deleted_payload = asyncio.run(coach_module.delete_voice_profile(profile_id="profile-1", user_id=7))
    assert deleted_payload == {"deleted": True, "profile_id": "profile-1"}


def test_coach_builtin_voice_library_endpoint(monkeypatch):
    payload = [
        {
            "id": "anby",
            "name": "Anby",
            "choice_id": "builtin:anby",
            "voice_mode": "preset",
            "voice_preset": "default",
            "provider": "cosyvoice",
            "is_default": True,
            "is_available": True,
            "avatar_data_url": "data:image/jpeg;base64,AAA",
        }
    ]

    async def fake_list_builtin_voices():
        return payload

    monkeypatch.setattr(coach_module.coach_service, "list_builtin_voices", fake_list_builtin_voices)
    result = asyncio.run(coach_module.list_builtin_voices(user_id=7))
    assert result == payload
