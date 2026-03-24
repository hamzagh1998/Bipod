import asyncio
import io
import sys
import types

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.services.coach_service as coach_service_module
from app.db.database import Base
from app.db.models import User
from app.services.coach_service import coach_service


class _AsyncSessionAdapter:
    def __init__(self, session):
        self._session = session

    async def execute(self, *args, **kwargs):
        return self._session.execute(*args, **kwargs)

    def add(self, obj):
        self._session.add(obj)

    async def commit(self):
        self._session.commit()

    async def refresh(self, obj):
        self._session.refresh(obj)

    async def flush(self):
        self._session.flush()

    async def delete(self, obj):
        self._session.delete(obj)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._session.close()


class _AsyncSessionFactory:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def __call__(self):
        return _AsyncSessionAdapter(self._session_factory())


@pytest.fixture
def coach_db(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'coach.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(coach_service_module, "AsyncSessionLocal", _AsyncSessionFactory(session_factory))
    yield engine
    engine.dispose()


async def _seed_user(username: str = "coach-user") -> User:
    async with coach_service_module.AsyncSessionLocal() as session:
        user = User(username=username, hashed_password="hash")
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


def test_coach_session_turn_mistake_flow(coach_db):
    async def scenario():
        user = await _seed_user()

        session = await coach_service.create_session(
            user_id=user.id,
            title="Pair Programming",
            target_language="English",
            native_language="Arabic",
            cefr_level="B1",
            audio_retention_opt_in=False,
            focus_area="debugging",
            model_id="coach-smart",
        )
        assert session.user_id == user.id
        assert session.title == "Pair Programming"
        assert session.target_language == "English"
        assert session.cefr_level == "B1"
        assert session.focus_area == "debugging"

        first_turn = await coach_service.save_turn_with_mistakes(
            session.id,
            user.id,
            transcript="I try iterative approach.",
            reply="Try an iterative approach.",
            score=81,
            correction="Use 'an iterative approach'.",
            explanation="Target: fix article usage. Native: add 'an' before iterative approach.",
            model_id="coach-smart",
            latency_ms=180,
            mistakes=[
                {
                    "category": "logic",
                    "detail": "Skipped the edge case for empty input.",
                    "severity": "high",
                    "suggestion": "Handle the empty list before the main loop.",
                },
                {
                    "category": "style",
                    "detail": "The explanation was too verbose.",
                },
            ],
        )
        assert first_turn is not None
        assert first_turn.turn_index == 1
        assert first_turn.reply == "Try an iterative approach."
        assert first_turn.score == 81
        assert len(first_turn.mistakes) == 2

        second_turn = await coach_service.save_turn_with_mistakes(
            session.id,
            user.id,
            transcript="What if the input is empty?",
            reply="Great question. Handle empty input before the loop.",
            score=88,
            model_id="coach-light",
            latency_ms=55,
            mistakes=[
                {
                    "category": "coverage",
                    "detail": "Did not address the empty-input branch.",
                }
            ],
        )
        assert second_turn is not None
        assert second_turn.turn_index == 2
        assert len(second_turn.mistakes) == 1

        loaded_session = await coach_service.get_session(session.id, user.id)
        assert loaded_session is not None
        assert len(loaded_session.turns) == 2

        sessions = await coach_service.list_sessions(user.id)
        assert len(sessions) == 1
        assert sessions[0]["target_language"] == "English"
        assert sessions[0]["native_language"] == "Arabic"
        assert sessions[0]["cefr_level"] == "B1"
        assert sessions[0]["turn_count"] == 2
        assert sessions[0]["mistake_count"] == 3

        turns = await coach_service.list_turns(session.id, user.id)
        assert [turn.turn_index for turn in turns] == [1, 2]
        assert turns[0].transcript == "I try iterative approach."
        assert turns[1].reply == "Great question. Handle empty input before the loop."
        assert [len(turn.mistakes) for turn in turns] == [2, 1]

        mistakes = await coach_service.list_mistakes(session.id, user.id)
        assert [mistake.category for mistake in mistakes] == ["logic", "style", "coverage"]

        summary = await coach_service.progress_summary(user.id)
        assert summary == {
            "user_id": user.id,
            "total_sessions": 1,
            "total_turns": 2,
            "total_mistakes": 3,
            "mistake_counts_by_category": {
                "logic": 1,
                "style": 1,
                "coverage": 1,
            },
            "turn_counts_by_model": {
                "coach-smart": 1,
                "coach-light": 1,
            },
            "active_sessions": 1,
            "latest_session_id": session.id,
            "latest_session_title": "Pair Programming",
            "latest_session_turns": 2,
        }

        session_progress = await coach_service.progress(user.id, session.id)
        assert session_progress == {
            "session_id": session.id,
            "turn_count": 2,
            "mistake_count": 3,
            "average_score": 84.5,
            "latest_score": 88,
        }

    asyncio.run(scenario())


def test_coach_save_turn_rejects_unknown_session(coach_db):
    async def scenario():
        user = await _seed_user("other-user")

        result = await coach_service.save_turn_with_mistakes(
            "missing-session",
            user.id,
            transcript="No session here.",
            reply="No session here.",
            score=0,
            mistakes=[{"category": "logic", "detail": "Missing session"}],
        )
        assert result is None

    asyncio.run(scenario())


def test_model_selection_uses_quality_order_and_latency_hooks():
    order = coach_service.get_quality_first_model_order(
        ["quality", "balanced", "fast"]
    )
    assert order == ["quality", "balanced", "fast"]

    selected = coach_service.select_model(
        candidate_order=["quality", "balanced", "fast"],
        latency_budget_ms=100.0,
        latency_probe=lambda model: {"quality": 240.0, "balanced": 125.0, "fast": 35.0}[model],
    )
    assert selected == "fast"

    fallback_selected = coach_service.select_model(
        candidate_order=["quality", "balanced"],
        latency_budget_ms=10.0,
        latency_probe=lambda model: 200.0,
        latency_fallback=lambda candidates: candidates[-1],
    )
    assert fallback_selected == "balanced"


def test_end_session_marks_completed_and_returns_summary(coach_db):
    async def scenario():
        user = await _seed_user("summary-user")
        session = await coach_service.create_session(
            user_id=user.id,
            title="Interview Prep",
            focus_area="job interview",
            target_language="English",
            cefr_level="B1",
        )

        await coach_service.save_turn_with_mistakes(
            session.id,
            user.id,
            transcript="I led a small team.",
            reply="Great. Tell me about a challenge.",
            score=82,
            explanation="Target: Add one concrete metric. Native: Give one measurable result.",
            mistakes=[
                {"category": "detail", "detail": "Missing measurable result.", "severity": "medium"},
            ],
        )
        await coach_service.save_turn_with_mistakes(
            session.id,
            user.id,
            transcript="We improved speed by 30 percent.",
            reply="Nice. What did you learn?",
            score=90,
            mistakes=[],
        )

        summary = await coach_service.end_session(user.id, session.id)
        assert summary is not None
        assert summary["status"] == "completed"
        assert summary["turn_count"] == 2
        assert summary["scored_turn_count"] == 2
        assert summary["average_score"] == 86.0
        assert "feedback_summary" in summary
        assert isinstance(summary["improvement_points"], list)

        sessions = await coach_service.list_sessions(user.id)
        assert sessions[0]["status"] == "completed"

    asyncio.run(scenario())


def test_get_whisper_model_uses_local_only_settings(monkeypatch):
    captured = {}

    class _FakeWhisperModel:
        def __init__(self, model_ref, **kwargs):
            captured["model_ref"] = model_ref
            captured["kwargs"] = kwargs

    fake_module = types.SimpleNamespace(WhisperModel=_FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(coach_service_module.settings, "COACH_WHISPER_MODEL", "small")
    monkeypatch.setattr(coach_service_module.settings, "COACH_WHISPER_MODEL_PATH", "/tmp/local-whisper-small")
    monkeypatch.setattr(coach_service_module.settings, "COACH_WHISPER_LOCAL_FILES_ONLY", True)
    monkeypatch.setattr(coach_service_module.settings, "COACH_WHISPER_DOWNLOAD_ROOT", "/tmp/hf-cache")
    monkeypatch.setattr(coach_service_module.settings, "USE_GPU", False)
    monkeypatch.delenv("COACH_WHISPER_MODEL", raising=False)
    monkeypatch.delenv("COACH_WHISPER_MODEL_PATH", raising=False)
    monkeypatch.delenv("COACH_WHISPER_LOCAL_FILES_ONLY", raising=False)
    monkeypatch.delenv("COACH_WHISPER_DOWNLOAD_ROOT", raising=False)

    coach_service._whisper_model = None

    async def scenario():
        model = await coach_service._get_whisper_model()
        assert model is not None

    asyncio.run(scenario())

    assert captured["model_ref"] == "/tmp/local-whisper-small"
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["compute_type"] == "int8"
    assert captured["kwargs"]["local_files_only"] is True
    assert captured["kwargs"]["download_root"] == "/tmp/hf-cache"
    coach_service._whisper_model = None


def test_transcript_evaluable_gate_filters_noise_placeholders():
    assert coach_service._is_transcript_evaluable("captured 93201 bytes of audio") is False
    assert coach_service._is_transcript_evaluable("cough") is False
    assert coach_service._is_transcript_evaluable("uh um") is False
    assert coach_service._is_transcript_evaluable("oh") is False
    assert coach_service._is_transcript_evaluable("I usually wake up at seven.") is True


def test_normalize_asr_preference_aliases():
    assert coach_service._normalize_asr_preference(None) == "auto"
    assert coach_service._normalize_asr_preference("fast") == "fast"
    assert coach_service._normalize_asr_preference("speed") == "fast"
    assert coach_service._normalize_asr_preference("accurate") == "accurate"
    assert coach_service._normalize_asr_preference("high_accuracy") == "accurate"
    assert coach_service._normalize_asr_preference("unknown-mode") == "auto"


def test_target_language_code_mapping():
    assert coach_service._target_language_code("English") == "en"
    assert coach_service._target_language_code("Spanish") == "es"
    assert coach_service._target_language_code("French (fr)") == "fr"
    assert coach_service._target_language_code("ar") == "ar"


def test_coach_hardware_profile_gpu_constrained(monkeypatch):
    monkeypatch.setattr(coach_service_module.settings, "USE_GPU", True)
    monkeypatch.setattr(coach_service_module.settings, "GPU_VRAM", 8.0)
    monkeypatch.setattr(coach_service_module.settings, "COACH_RUNTIME_PROFILE", "auto")
    monkeypatch.setattr(coach_service_module.settings, "COACH_HIGH_VRAM_THRESHOLD_GB", 16.0)
    monkeypatch.delenv("COACH_RUNTIME_PROFILE", raising=False)
    profile = coach_service._coach_hardware_profile()
    assert profile["name"] == "gpu_constrained"
    assert profile["asr_device"] == "cpu"
    assert profile["tts_device"] == "cpu"
    assert profile["asr_fast_model"] == "medium"


def test_coach_hardware_profile_gpu_full(monkeypatch):
    monkeypatch.setattr(coach_service_module.settings, "USE_GPU", True)
    monkeypatch.setattr(coach_service_module.settings, "GPU_VRAM", 24.0)
    monkeypatch.setattr(coach_service_module.settings, "COACH_RUNTIME_PROFILE", "auto")
    monkeypatch.setattr(coach_service_module.settings, "COACH_HIGH_VRAM_THRESHOLD_GB", 16.0)
    monkeypatch.delenv("COACH_RUNTIME_PROFILE", raising=False)
    profile = coach_service._coach_hardware_profile()
    assert profile["name"] == "gpu_full"
    assert profile["asr_device"] == "cuda"
    assert profile["tts_device"] == "cuda"
    assert profile["asr_fast_model"] == "large-v3"


def test_fast_whisper_model_ref_uses_profile_when_auto(monkeypatch):
    monkeypatch.setattr(coach_service_module.settings, "USE_GPU", True)
    monkeypatch.setattr(coach_service_module.settings, "GPU_VRAM", 24.0)
    monkeypatch.setattr(coach_service_module.settings, "COACH_WHISPER_FAST_MODEL", "auto")
    monkeypatch.setattr(coach_service_module.settings, "COACH_WHISPER_MODEL", "auto")
    monkeypatch.setattr(coach_service_module.settings, "COACH_WHISPER_MODEL_PATH", "")
    monkeypatch.delenv("COACH_WHISPER_FAST_MODEL", raising=False)
    monkeypatch.delenv("COACH_WHISPER_MODEL", raising=False)
    monkeypatch.delenv("COACH_WHISPER_MODEL_PATH", raising=False)
    model_ref = coach_service._fast_whisper_model_ref()
    assert model_ref == "large-v3"


def test_supported_languages_contains_english():
    payload = coach_service.supported_languages()
    assert payload
    english = next((item for item in payload if item["name"] == "English"), None)
    assert english is not None
    assert english["selectable"] is True


def test_create_session_rejects_unsupported_language(coach_db):
    async def scenario():
        user = await _seed_user("unsupported-language-user")
        with pytest.raises(ValueError) as exc_info:
            await coach_service.create_session(
                user_id=user.id,
                title="Invalid language",
                target_language="Klingon",
            )
        assert "not supported" in str(exc_info.value)

    asyncio.run(scenario())


def test_delete_session_removes_owned_session(coach_db):
    async def scenario():
        user = await _seed_user("delete-user")
        session = await coach_service.create_session(
            user_id=user.id,
            title="Delete me",
            focus_area="cleanup",
        )
        deleted = await coach_service.delete_session(session.id, user.id)
        assert deleted is True

        loaded = await coach_service.get_session(session.id, user.id)
        assert loaded is None

    asyncio.run(scenario())


def test_resolve_tts_voice_preset_anby():
    voice, rate, pitch = coach_service._resolve_tts_voice(language="English", voice_preset="anby")
    assert voice.startswith("en")
    assert rate == 145
    assert pitch == 38


def test_get_tts_status_returns_ready_for_espeak(monkeypatch):
    monkeypatch.setenv("COACH_TTS_PROVIDER", "espeak")

    async def scenario():
        payload = await coach_service.get_tts_status(warm=True)
        assert payload["engine"] == "espeak"
        assert payload["ready"] is True
        assert payload["state"] == "ready"

    asyncio.run(scenario())


def test_get_tts_status_normalizes_cosyvoice_payload(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "ok": True,
                "engine": "cosyvoice",
                "ready": False,
                "state": "downloading",
                "detail": "Downloading model",
                "model_id": "iic/CosyVoice-300M",
                "loaded_model_id": "",
                "warmup_active": True,
                "updated_at": 456.0,
            }

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        async def get(self, url, params=None):
            captured["url"] = url
            captured["params"] = params
            return _FakeResponse()

    monkeypatch.setenv("COACH_TTS_PROVIDER", "cosyvoice")
    monkeypatch.setenv("COACH_COSYVOICE_BASE_URL", "http://cosyvoice:5001")
    monkeypatch.setenv("COACH_COSYVOICE_MODEL_ID", "iic/CosyVoice-300M")
    monkeypatch.setattr(coach_service_module.httpx, "AsyncClient", _FakeAsyncClient)

    async def scenario():
        payload = await coach_service.get_tts_status(warm=True)
        assert payload["engine"] == "cosyvoice"
        assert payload["provider"] == "cosyvoice"
        assert payload["ready"] is False
        assert payload["state"] == "downloading"
        assert payload["warmup_active"] is True
        assert payload["model_id"] == "iic/CosyVoice-300M"
        assert payload["planned_runtime_device"] in {"cpu", "cuda"}
        assert payload["allocation_policy"] in {"auto_balance", "prioritize_llm", "prioritize_tts"}

    asyncio.run(scenario())
    assert captured["url"] == "http://cosyvoice:5001/status"
    assert captured["params"]["warm"] == "true"
    assert captured["params"]["runtime_device"] in {"cpu", "cuda"}


def test_get_tts_status_normalizes_openvoice_payload(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "ok": True,
                "engine": "openvoice",
                "ready": True,
                "state": "ready",
                "detail": "OpenVoice model ready.",
                "model_id": "openvoice-v2",
                "loaded_model_id": "openvoice-v2",
                "warmup_active": False,
                "updated_at": 987.0,
                "runtime_device": "cpu",
            }

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        async def get(self, url, params=None):
            captured["url"] = url
            captured["params"] = params
            return _FakeResponse()

    monkeypatch.setenv("COACH_TTS_PROVIDER", "cosyvoice")
    monkeypatch.setenv("COACH_OPENVOICE_BASE_URL", "http://openvoice:5002")
    monkeypatch.setenv("COACH_OPENVOICE_MODEL_ID", "openvoice-v2")
    monkeypatch.setattr(coach_service_module.httpx, "AsyncClient", _FakeAsyncClient)

    async def scenario():
        payload = await coach_service.get_tts_status(
            warm=True,
            preferred_tts_provider="openvoice",
        )
        assert payload["engine"] == "openvoice"
        assert payload["provider"] == "openvoice"
        assert payload["ready"] is True
        assert payload["state"] == "ready"
        assert payload["model_id"] == "openvoice-v2"
        assert payload["planned_runtime_device"] in {"cpu", "cuda"}

    asyncio.run(scenario())
    assert captured["url"] == "http://openvoice:5002/status"
    assert captured["params"]["warm"] == "true"
    assert captured["params"]["runtime_device"] in {"cpu", "cuda"}


def test_runtime_allocation_respects_manual_pin(monkeypatch):
    monkeypatch.setenv("COACH_ALLOCATION_POLICY", "auto_balance")
    monkeypatch.setattr(coach_service_module.settings, "USE_GPU", True)
    profile = {
        "name": "gpu_constrained",
        "use_gpu": True,
        "gpu_vram_gb": 6.0,
        "llm_primary_model": "qwen3:8b",
        "tts_device": "cpu",
    }
    allocation = coach_service._runtime_allocation(
        profile=profile,
        mode="voice",
        preferred_model="qwen3:8b",
    )
    assert allocation["policy"] == "auto_balance"
    assert allocation["llm_device"] == "cuda"
    assert allocation["tts_device"] == "cpu"
    assert allocation["reason"] == "manual_llm_pin"


def test_get_runtime_status_aggregates_components(monkeypatch):
    async def fake_get_tts_status(
        *,
        warm,
        session_id=None,
        user_id=None,
        preferred_model=None,
        preferred_tts_provider=None,
        llm_device_preference="auto",
        tts_device_preference="auto",
    ):
        assert warm is True
        assert session_id is None
        assert user_id is None
        assert preferred_model is None
        assert preferred_tts_provider is None
        assert llm_device_preference == "auto"
        assert tts_device_preference == "auto"
        return {"engine": "cosyvoice", "ready": True, "state": "ready", "detail": "ok"}

    async def fake_asr_status(*, warm):
        assert warm is True
        return {"engine": "faster-whisper", "ready": True, "state": "ready", "detail": "ok"}

    async def fake_ollama_status(
        *,
        warm,
        mode,
        preferred_model=None,
        llm_device_preference="auto",
        tts_device_preference="auto",
    ):
        assert warm is True
        assert mode == "voice"
        assert preferred_model is None
        assert llm_device_preference == "auto"
        assert tts_device_preference == "auto"
        return {"engine": "ollama", "ready": True, "state": "ready", "detail": "ok"}

    async def fake_languagetool_status():
        return {"engine": "languagetool", "enabled": True, "ready": False, "state": "error", "detail": "offline"}

    monkeypatch.setattr(coach_service, "get_tts_status", fake_get_tts_status)
    monkeypatch.setattr(coach_service, "_asr_status", fake_asr_status)
    monkeypatch.setattr(coach_service, "_ollama_status", fake_ollama_status)
    monkeypatch.setattr(coach_service, "_languagetool_status", fake_languagetool_status)
    monkeypatch.setattr(
        coach_service,
        "_coach_hardware_profile",
        lambda: {
            "name": "gpu_constrained",
            "use_gpu": True,
            "gpu_vram_gb": 8.0,
            "high_vram_threshold_gb": 16.0,
            "llm_primary_model": "qwen3:8b",
            "asr_device": "cpu",
            "asr_fast_model": "medium",
            "asr_accurate_model": "large-v3",
            "tts_device": "cpu",
        },
    )

    async def scenario():
        payload = await coach_service.get_runtime_status(warm=True, mode="voice")
        assert payload["ok"] is True
        assert payload["mode"] == "voice"
        assert payload["ready"] is True
        assert payload["components"]["languagetool"]["ready"] is False
        assert payload["runtime_profile"]["name"] == "gpu_constrained"

    asyncio.run(scenario())


def test_get_runtime_status_voice_autowarms_asr(monkeypatch):
    async def fake_get_tts_status(
        *,
        warm,
        session_id=None,
        user_id=None,
        preferred_model=None,
        preferred_tts_provider=None,
        llm_device_preference="auto",
        tts_device_preference="auto",
    ):
        assert warm is False
        return {"engine": "cosyvoice", "ready": True, "state": "ready", "detail": "ok"}

    async def fake_asr_status(*, warm, auto_warm=False):
        assert warm is False
        assert auto_warm is True
        return {"engine": "faster-whisper", "ready": True, "state": "ready", "detail": "ok"}

    async def fake_ollama_status(
        *,
        warm,
        mode,
        preferred_model=None,
        llm_device_preference="auto",
        tts_device_preference="auto",
    ):
        assert warm is False
        assert mode == "voice"
        return {"engine": "ollama", "ready": True, "state": "ready", "detail": "ok"}

    async def fake_languagetool_status():
        return {"engine": "languagetool", "enabled": True, "ready": False, "state": "error", "detail": "offline"}

    monkeypatch.setattr(coach_service, "get_tts_status", fake_get_tts_status)
    monkeypatch.setattr(coach_service, "_asr_status", fake_asr_status)
    monkeypatch.setattr(coach_service, "_ollama_status", fake_ollama_status)
    monkeypatch.setattr(coach_service, "_languagetool_status", fake_languagetool_status)

    async def scenario():
        payload = await coach_service.get_runtime_status(warm=False, mode="voice")
        assert payload["ok"] is True
        assert payload["mode"] == "voice"
        assert payload["ready"] is True

    asyncio.run(scenario())


def test_process_text_turn_adds_languagetool_findings(coach_db, monkeypatch):
    async def scenario():
        user = await _seed_user("text-user")
        session = await coach_service.create_session(
            user_id=user.id,
            title="Text mode",
            target_language="English",
            focus_area="Daily life",
        )

        async def fake_available_models():
            return ["qwen3:8b"]

        async def fake_coach_with_model(**_kwargs):
            return (
                {
                    "reply": "Nice. Tell me one more detail.",
                    "follow_up_question": "What time do you start work?",
                    "score": 84,
                    "correction": "Use present simple consistently.",
                    "explanation": "Target: keep tense consistent. Native: حافظ على نفس الزمن.",
                    "mistakes": [
                        {
                            "category": "grammar",
                            "detail": "Tense shift detected.",
                            "severity": "medium",
                            "suggestion": "Use present simple.",
                        }
                    ],
                },
                120,
            )

        async def fake_languagetool_check(*, text, language):
            assert "usually" in text
            assert language == "English"
            return [
                {
                    "category": "grammar",
                    "detail": "Possible article issue.",
                    "severity": "low",
                    "suggestion": "Add 'the'.",
                    "metadata": {"source": "languagetool"},
                }
            ]

        monkeypatch.setattr(coach_service, "_available_ollama_models", fake_available_models)
        monkeypatch.setattr(coach_service, "_coach_with_model", fake_coach_with_model)
        monkeypatch.setattr(coach_service, "_languagetool_check", fake_languagetool_check)

        payload = await coach_service.process_text_turn(
            user_id=user.id,
            session_id=session.id,
            text="I usually wake up at 6 and go work.",
            preferred_model="qwen3:8b",
            persona_style="calm tactical persona",
        )
        assert payload["session_id"] == session.id
        assert payload["score"] == 84
        assert len(payload["mistakes"]) >= 2
        assert any("Possible article issue." in str(item.get("detail")) for item in payload["mistakes"])
        assert "LanguageTool flagged" in str(payload.get("explanation") or "")

    asyncio.run(scenario())


def test_save_turn_filters_low_value_typo_mistakes(coach_db):
    async def scenario():
        user = await _seed_user("mistake-filter-user")
        session = await coach_service.create_session(
            user_id=user.id,
            title="Mistake filtering",
            target_language="English",
            focus_area="Daily life",
        )

        turn = await coach_service.save_turn_with_mistakes(
            session_id=session.id,
            user_id=user.id,
            transcript="I am here with you.",
            reply="Good. Continue.",
            score=80,
            mistakes=[
                {
                    "category": "grammar",
                    "detail": "missing 'e' in 'you'",
                    "severity": "low",
                    "suggestion": "you",
                },
                {
                    "category": "grammar",
                    "detail": "Use present simple here.",
                    "severity": "medium",
                    "suggestion": "Use present simple.",
                },
            ],
        )

        assert turn is not None
        assert len(turn.mistakes) == 1
        assert turn.mistakes[0].detail == "Use present simple here."

    asyncio.run(scenario())


def test_stream_turn_keeps_explicit_user_model_when_slow(coach_db, monkeypatch):
    class _Upload:
        def __init__(self, data: bytes, filename: str = "voice.webm"):
            self._buffer = io.BytesIO(data)
            self.filename = filename

        async def read(self, size: int = -1):
            return self._buffer.read(size)

        async def seek(self, offset: int):
            self._buffer.seek(offset)

    async def scenario():
        user = await _seed_user("voice-model-lock-user")
        session = await coach_service.create_session(
            user_id=user.id,
            title="Model lock",
            target_language="English",
            focus_area="Daily life",
        )

        async def fake_transcribe_audio_adaptive(**_kwargs):
            return {
                "text": "I usually walk to work and buy coffee on the way.",
                "model": "medium",
                "retry_used": False,
                "selection": "auto",
                "confidence": 0.93,
                "confidence_band": "high",
                "avg_logprob": -0.2,
                "no_speech_prob": 0.01,
            }

        async def fake_available_models():
            return ["qwen3:8b", "qwen2.5:7b"]

        async def fake_coach_with_model(*, model, **_kwargs):
            latency = 9999 if model == "qwen3:8b" else 120
            return (
                {
                    "reply": "Nice sentence. Add one detail about time.",
                    "follow_up_question": "What time do you leave home?",
                    "score": 86,
                    "correction": "I usually walk to work and buy coffee on the way.",
                    "explanation": "Target: include a time marker. Native: أضف وقتا محددا.",
                    "mistakes": [],
                },
                latency,
            )

        monkeypatch.setenv("COACH_ENABLE_TWO_PASS_VOICE", "false")
        monkeypatch.setattr(coach_service, "default_latency_budget_ms", 50.0)
        monkeypatch.setattr(coach_service, "_transcribe_audio_adaptive", fake_transcribe_audio_adaptive)
        monkeypatch.setattr(coach_service, "_available_ollama_models", fake_available_models)
        monkeypatch.setattr(coach_service, "_coach_with_model", fake_coach_with_model)

        events = []
        async for event in coach_service.stream_turn(
            user_id=user.id,
            session_id=session.id,
            audio=_Upload(b"fake-audio"),
            preferred_model="qwen3:8b",
            preferred_asr_model="auto",
            persona_style="Anby persona",
        ):
            events.append(event)

        fallback_events = [
            event
            for event in events
            if str(event.get("type") or "") == "model_fallback"
            and str(event.get("reason") or "") == "latency_budget_exceeded"
        ]
        assert fallback_events == []
        score_events = [event for event in events if str(event.get("type") or "") == "score"]
        assert score_events
        assert score_events[-1].get("model_id") == "qwen3:8b"

        turns = await coach_service.list_turns(session.id, user.id)
        assert turns
        assert turns[-1].model_id == "qwen3:8b"

    asyncio.run(scenario())


def test_asr_confidence_band_low_for_noise():
    score, band = coach_service._asr_confidence_band(
        avg_logprob=-1.6,
        no_speech_prob=0.9,
        text="uh um",
    )
    assert band == "low"
    assert score < 0.45


def test_prepend_explicit_recast_for_sentence_correction():
    reply = coach_service._prepend_explicit_recast(
        reply="Good effort. Tell me one more detail.",
        correction="I went to the coffee shop yesterday",
        mistakes=[{"category": "grammar", "detail": "Incorrect verb tense.", "severity": "medium"}],
        confidence_band="high",
    )
    assert reply.startswith("You mean: I went to the coffee shop yesterday.")


def test_prepend_explicit_recast_skips_instructional_correction():
    reply = coach_service._prepend_explicit_recast(
        reply="Good effort. Tell me one more detail.",
        correction="Use past simple.",
        mistakes=[{"category": "grammar", "detail": "Incorrect verb tense.", "severity": "medium"}],
        confidence_band="high",
    )
    assert reply == "Good effort. Tell me one more detail."


def test_synthesize_reply_audio_requires_engine(monkeypatch):
    monkeypatch.setattr(coach_service_module.shutil, "which", lambda _cmd: None)

    async def scenario():
        with pytest.raises(RuntimeError) as exc_info:
            await coach_service.synthesize_reply_audio(
                text="hello",
                language="English",
                voice_preset="default",
                persona_style=None,
            )
        assert "espeak-ng" in str(exc_info.value)

    asyncio.run(scenario())


def test_synthesize_reply_audio_disables_espeak_fallback_for_clone_voice(monkeypatch):
    monkeypatch.setattr(coach_service_module.shutil, "which", lambda _cmd: "/usr/bin/espeak-ng")
    monkeypatch.setattr(coach_service, "_builtin_voice_record", lambda _voice_id: {"id": "anby", "sample_path": "/tmp/fake"})

    async def fake_synthesize_with_cosyvoice(**_kwargs):
        raise RuntimeError("CosyVoice clone unavailable")

    monkeypatch.setattr(coach_service, "_synthesize_with_cosyvoice", fake_synthesize_with_cosyvoice)
    monkeypatch.setattr(
        coach_service,
        "_load_reference_audio_bytes",
        lambda **_kwargs: asyncio.sleep(0, result=b"fake-reference-audio"),
    )
    monkeypatch.setenv("COACH_TTS_ALLOW_ESPEAK_FALLBACK", "true")

    async def scenario():
        with pytest.raises(RuntimeError) as exc_info:
            await coach_service.synthesize_reply_audio(
                text="hello",
                language="English",
                voice_preset="default",
                persona_style=None,
                tts_provider="cosyvoice",
                voice_mode="preset",
                builtin_voice_id="anby",
                user_id=1,
            )
        assert "local fallback is disabled for clone voices" in str(exc_info.value)

    asyncio.run(scenario())


def test_synthesize_reply_audio_uses_openvoice_provider(monkeypatch):
    monkeypatch.setattr(coach_service_module.shutil, "which", lambda _cmd: "/usr/bin/espeak-ng")

    captured = {}

    async def fake_synthesize_with_openvoice(**kwargs):
        captured.update(kwargs)
        return b"openvoice-audio", "audio/wav"

    monkeypatch.setattr(coach_service, "_synthesize_with_openvoice", fake_synthesize_with_openvoice)

    async def scenario():
        audio, media_type = await coach_service.synthesize_reply_audio(
            text="Bonjour",
            language="French",
            voice_preset="default",
            persona_style=None,
            tts_provider="openvoice",
            voice_mode="preset",
            user_id=1,
        )
        assert media_type == "audio/wav"
        assert audio == b"openvoice-audio"
        assert captured["voice_mode"] == "preset"
        assert captured["language"] == "French"
        assert captured["reference_audio_bytes"] is None

    asyncio.run(scenario())


def test_synthesize_reply_audio_uses_local_tts_for_non_english_builtin_clone(monkeypatch):
    monkeypatch.setattr(coach_service_module.shutil, "which", lambda _cmd: "/usr/bin/espeak-ng")
    monkeypatch.setattr(coach_service, "_builtin_voice_record", lambda _voice_id: {"id": "anby", "sample_path": "/tmp/fake"})
    monkeypatch.setenv("COACH_TTS_ALLOW_ESPEAK_FALLBACK", "true")
    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr(coach_service_module.asyncio, "to_thread", fake_to_thread)

    called = {"cosyvoice": 0, "espeak": 0}

    async def fake_synthesize_with_cosyvoice(**_kwargs):
        called["cosyvoice"] += 1
        raise AssertionError("CosyVoice should not be called for non-English built-in clone voices")

    def fake_synthesize_with_espeak(**_kwargs):
        called["espeak"] += 1
        return b"wav-bytes"

    monkeypatch.setattr(coach_service, "_synthesize_with_cosyvoice", fake_synthesize_with_cosyvoice)
    monkeypatch.setattr(coach_service, "_synthesize_with_espeak", fake_synthesize_with_espeak)

    async def scenario():
        audio, media_type = await coach_service.synthesize_reply_audio(
            text="Bonjour, comment ca va ?",
            language="French",
            voice_preset="default",
            persona_style="Impersonate Anby Demara",
            tts_provider="cosyvoice",
            voice_mode="preset",
            builtin_voice_id="anby",
            user_id=1,
        )
        assert media_type == "audio/wav"
        assert audio == b"wav-bytes"
        assert called["cosyvoice"] == 0
        assert called["espeak"] == 1

    asyncio.run(scenario())


def test_voice_sample_profile_crud_flow(coach_db):
    class _Upload:
        def __init__(self, data: bytes, filename: str = "voice.wav", content_type: str = "audio/wav"):
            self._buffer = io.BytesIO(data)
            self.filename = filename
            self.content_type = content_type

        async def read(self, size: int = -1):
            return self._buffer.read(size)

        async def seek(self, offset: int):
            self._buffer.seek(offset)

    async def scenario():
        user = await _seed_user("voice-user")
        sample = await coach_service.save_voice_reference(
            user_id=user.id,
            file=_Upload(b"RIFFfakewav"),
            title="sample.wav",
            language="English",
        )
        assert sample["id"]
        assert sample["title"] == "sample.wav"

        profile = await coach_service.create_voice_profile(
            user_id=user.id,
            name="My cloned voice",
            reference_clip_id=sample["id"],
            language="English",
        )
        assert profile["id"]
        assert profile["name"] == "My cloned voice"

        listed = await coach_service.list_voice_profiles(user_id=user.id)
        assert len(listed) == 1
        assert listed[0]["id"] == profile["id"]

        deleted = await coach_service.delete_voice_profile(profile_id=profile["id"], user_id=user.id)
        assert deleted is True

        listed_after_delete = await coach_service.list_voice_profiles(user_id=user.id)
        assert listed_after_delete == []

    asyncio.run(scenario())


def test_builtin_voice_library_and_session_binding(coach_db, monkeypatch, tmp_path):
    library_root = tmp_path / "voice-library"
    samples_dir = library_root / "clone samples"
    images_dir = library_root / "images"
    samples_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / "[Anby] sample.mp3").write_bytes(b"fake-mp3-audio")
    (images_dir / "anby.jpg").write_bytes(b"fake-image")
    monkeypatch.setattr(coach_service_module.settings, "COACH_VOICE_LIBRARY_DIR", str(library_root))

    async def scenario():
        user = await _seed_user("builtin-voice-user")
        voices = await coach_service.list_builtin_voices()
        assert len(voices) == 1
        assert voices[0]["id"] == "anby"
        assert voices[0]["is_default"] is True
        assert voices[0]["choice_id"] == "builtin:anby"
        assert voices[0]["avatar_data_url"].startswith("data:image/")

        session = await coach_service.create_session(
            user_id=user.id,
            title="Built-in voice session",
            target_language="English",
            voice_profile_id="builtin:anby",
        )
        assert session.voice_profile_id == "builtin:anby"

    asyncio.run(scenario())


def test_normalize_learner_level_aliases():
    assert coach_service._normalize_learner_level("A1") == "novice"
    assert coach_service._normalize_learner_level("B1") == "medium"
    assert coach_service._normalize_learner_level("C2") == "fluent"
    assert coach_service._normalize_learner_level("ignorant") == "ignorant"


def test_append_native_translation():
    rendered = coach_service._append_native_translation(
        message="Let's practice one short sentence.",
        translation="لنتمرن على جملة قصيرة واحدة.",
        target_language="English",
        native_language="Arabic",
    )
    assert "[Arabic]" in rendered
    assert "لنتمرن" in rendered


def test_append_native_translation_parses_mapping_string():
    rendered = coach_service._append_native_translation(
        message="Parfait, on continue.",
        translation="{'en': 'Great, let us continue.', 'fr': 'Parfait, on continue.'}",
        target_language="French",
        native_language="English",
    )
    assert "[English]" in rendered
    assert "Great, let us continue." in rendered
    assert "{'en':" not in rendered


def test_append_native_translation_parses_mapping_with_prefix_label():
    rendered = coach_service._append_native_translation(
        message="Parfait, on continue.",
        translation="[English] {'en': 'Great, let us continue.', 'fr': 'Parfait, on continue.'}",
        target_language="French",
        native_language="English",
    )
    assert "[English]" in rendered
    assert "Great, let us continue." in rendered
    assert "{'en':" not in rendered


def test_append_native_translation_parses_mapping_object():
    rendered = coach_service._append_native_translation(
        message="Parfait, on continue.",
        translation={"en": "Great, let us continue.", "fr": "Parfait, on continue."},
        target_language="French",
        native_language="English",
    )
    assert "[English]" in rendered
    assert "Great, let us continue." in rendered


def test_prevent_repeat_loop_changes_duplicate_reply():
    rewritten = coach_service._prevent_repeat_loop(
        candidate_reply="Can you give one more example?",
        last_coach_reply="Can you give one more example?",
        target_language="French",
        native_language="Arabic",
        learner_level="novice",
        focus_area="Daily life",
    )
    assert rewritten != "Can you give one more example?"
    assert "Arabic" in rewritten


def test_prevent_repeat_loop_changes_near_duplicate_reply():
    rewritten = coach_service._prevent_repeat_loop(
        candidate_reply="I did not understand. Could you repeat that?",
        last_coach_reply="I did not understand, could you repeat that please?",
        target_language="French",
        native_language="English",
        learner_level="medium",
        focus_area="Daily life",
        learner_transcript="what's your name?",
    )
    assert rewritten != "I did not understand. Could you repeat that?"
    assert "what's your name?" in rewritten


def test_apply_direct_question_guard_answers_name():
    guarded = coach_service._apply_direct_question_guard(
        transcript="what's your name?",
        candidate_reply="Let's continue. Tell me more.",
        target_language="French",
        persona_style="Impersonate Anby Demara for this full session.",
    )
    assert guarded.startswith("Je m'appelle Anby.")


def test_sanitize_coach_tone_rewrites_accusatory_phrase():
    rewritten = coach_service._sanitize_coach_tone(
        "You're avoiding the question. What specific activity do you enjoy?"
    )
    assert "avoiding the question" not in rewritten.lower()
    assert rewritten.startswith("Thanks. Let's make your answer more specific.")
