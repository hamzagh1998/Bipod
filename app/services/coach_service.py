import asyncio
import ast
import base64
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Mapping, Optional, Sequence

import httpx
from cryptography.fernet import Fernet, InvalidToken # type: ignore
from sqlalchemy import desc, func, select
from sqlalchemy.orm import selectinload

from app.core.config import recommend_coach_runtime_profile, settings
from app.core.logger import get_logger
from app.db.database import AsyncSessionLocal
from app.db.models import CoachMistake, CoachSession, CoachTurn, CoachVoiceProfile, CoachVoiceSample

logger = get_logger("bipod.services.coach")

LatencyProbe = Callable[[str], Optional[float]]
LatencyFallback = Callable[[List[str]], Optional[str]]


class CoachService:
    OLLAMA_CONNECT_TIMEOUT_SEC = 10.0
    OLLAMA_WRITE_TIMEOUT_SEC = 30.0
    OLLAMA_POOL_TIMEOUT_SEC = 30.0
    COACH_MAX_AUDIO_BYTES = 20 * 1024 * 1024
    COACH_VOICE_MAX_AUDIO_BYTES = 12 * 1024 * 1024
    WHISPER_LANGUAGE_MAP = {
        "arabic": "ar",
        "english": "en",
        "french": "fr",
        "spanish": "es",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "russian": "ru",
        "turkish": "tr",
        "hindi": "hi",
        "urdu": "ur",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "dutch": "nl",
        "swedish": "sv",
        "polish": "pl",
        "ukrainian": "uk",
        "greek": "el",
    }
    TTS_LANGUAGE_VOICE_MAP = {
        "ar": "ar",
        "de": "de",
        "el": "el",
        "en": "en-us",
        "es": "es",
        "fr": "fr-fr",
        "hi": "hi",
        "it": "it",
        "ja": "ja",
        "ko": "ko",
        "nl": "nl",
        "pl": "pl",
        "pt": "pt",
        "ru": "ru",
        "sv": "sv",
        "tr": "tr",
        "uk": "uk",
        "ur": "ur",
        "zh": "zh",
    }
    BUILTIN_VOICE_ORDER = ("anby", "bmo", "goku", "gute")
    BUILTIN_VOICE_LABELS = {
        "anby": "Anby",
        "bmo": "BMO",
        "goku": "Goku",
        "gute": "Gute",
    }
    BUILTIN_CLONE_SAFE_LANGUAGE_CODES = {"en"}
    CODE_TO_LANGUAGE_NAME = {
        "ar": "Arabic",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "hi": "Hindi",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "ru": "Russian",
        "sv": "Swedish",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "zh": "Chinese",
    }
    LOW_VALUE_MISSING_CHAR_PATTERN = re.compile(
        r"missing\s+'(?P<char>[^']{1,3})'\s+in\s+'(?P<word>[^']+)'",
        flags=re.IGNORECASE,
    )
    LEARNER_LEVELS = ("ignorant", "novice", "medium", "high", "fluent")
    LEARNER_LEVEL_ALIASES = {
        "a0": "ignorant",
        "beginner": "ignorant",
        "starter": "ignorant",
        "ignorant": "ignorant",
        "a1": "novice",
        "a2": "novice",
        "novice": "novice",
        "basic": "novice",
        "b1": "medium",
        "intermediate": "medium",
        "medium": "medium",
        "mid": "medium",
        "b2": "high",
        "advanced": "high",
        "high": "high",
        "c1": "fluent",
        "c2": "fluent",
        "fluent": "fluent",
        "proficient": "fluent",
    }
    ALLOCATION_POLICIES = ("auto_balance", "prioritize_llm", "prioritize_tts")
    DEVICE_PREFERENCES = ("auto", "cpu", "cuda")

    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL
        self.default_latency_budget_ms = 15000.0
        self._whisper_model = None
        self._whisper_accurate_model = None
        self._whisper_lock = asyncio.Lock()
        self._whisper_runtime_device: Optional[str] = None
        self._whisper_accurate_runtime_device: Optional[str] = None
        self._asr_warmup_task: Optional[asyncio.Task[Any]] = None
        self._voice_cipher: Optional[Fernet] = None
        self._voice_cipher_key: Optional[str] = None

    def _env_text(self, key: str, fallback: Any) -> str:
        return str(os.environ.get(key, fallback) or fallback).strip()

    def _env_bool(self, key: str, fallback: bool) -> bool:
        raw = self._env_text(key, str(fallback)).lower()
        if raw == "auto":
            return fallback
        return raw in {"1", "true", "yes", "on"}

    def _env_float(self, key: str, fallback: float, *, minimum: Optional[float] = None) -> float:
        raw = self._env_text(key, fallback)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float(fallback)
        if minimum is not None:
            value = max(minimum, value)
        return value

    def _model_size_billion(self, model_name: Optional[str]) -> Optional[float]:
        raw = str(model_name or "").strip().lower()
        if not raw:
            return None
        match = re.search(r":(\d+(?:\.\d+)?)b\b", raw)
        if not match:
            return None
        try:
            parsed = float(match.group(1))
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    def _allocation_policy_name(self) -> str:
        configured = self._env_text("COACH_ALLOCATION_POLICY", settings.COACH_ALLOCATION_POLICY).strip().lower()
        aliases = {
            "auto": "auto_balance",
            "balanced": "auto_balance",
            "auto_balance": "auto_balance",
            "llm": "prioritize_llm",
            "prioritize_llm": "prioritize_llm",
            "tts": "prioritize_tts",
            "prioritize_tts": "prioritize_tts",
        }
        policy = aliases.get(configured, "auto_balance")
        if policy not in self.ALLOCATION_POLICIES:
            return "auto_balance"
        return policy

    def _normalize_device_preference(self, value: Optional[str]) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in self.DEVICE_PREFERENCES:
            return normalized
        return "auto"

    def _effective_device_preference(
        self,
        *,
        explicit_preference: Optional[str],
        persisted_preference: Optional[str],
    ) -> str:
        normalized_explicit = self._normalize_device_preference(explicit_preference)
        if normalized_explicit != "auto":
            return normalized_explicit
        return self._normalize_device_preference(persisted_preference)

    def _ollama_num_gpu_option(self, llm_device: Optional[str]) -> Optional[int]:
        normalized = str(llm_device or "").strip().lower()
        if normalized == "cpu":
            return 0
        return None

    def _runtime_allocation(
        self,
        *,
        profile: Mapping[str, Any],
        mode: str,
        preferred_model: Optional[str] = None,
        llm_device_preference: Optional[str] = None,
        tts_device_preference: Optional[str] = None,
    ) -> dict[str, Any]:
        policy = self._allocation_policy_name()
        runtime_mode = self._coach_runtime_mode(mode)
        gpu_enabled = bool(settings.USE_GPU) and bool(profile.get("use_gpu"))
        gpu_vram_gb = max(0.0, float(profile.get("gpu_vram_gb") or 0.0))
        selected_model = str(preferred_model or profile.get("llm_primary_model") or "").strip()
        profile_name = str(profile.get("name") or "").strip().lower()
        pinned_model = bool(str(preferred_model or "").strip() and str(preferred_model or "").strip().lower() != "auto")
        model_size_b = self._model_size_billion(selected_model)
        llm_pref = self._normalize_device_preference(llm_device_preference)
        tts_pref = self._normalize_device_preference(tts_device_preference)

        llm_device = "cuda" if gpu_enabled else "cpu"
        tts_device = str(profile.get("tts_device") or ("cuda" if gpu_enabled else "cpu")).strip().lower() or "cpu"
        reason = "balanced_default"
        warnings: list[str] = []

        if not gpu_enabled:
            if llm_pref == "cuda":
                warnings.append("LLM GPU preference was requested but CUDA is unavailable; using CPU.")
            if tts_pref == "cuda":
                warnings.append("TTS GPU preference was requested but CUDA is unavailable; using CPU.")
            if llm_pref != "auto" or tts_pref != "auto":
                reason = "cuda_unavailable"
            return {
                "policy": policy,
                "reason": reason,
                "llm_device": "cpu",
                "tts_device": "cpu",
                "gpu_vram_gb": gpu_vram_gb,
                "selected_model": selected_model,
                "model_size_b": model_size_b,
                "pinned_model": pinned_model,
                "llm_device_preference": llm_pref,
                "tts_device_preference": tts_pref,
                "allocation_warnings": warnings,
            }

        if runtime_mode == "idle":
            llm_device = "cuda"
            tts_device = "cpu"
            reason = "idle_mode"

        if policy == "prioritize_tts":
            if pinned_model and (model_size_b is None or model_size_b > 4.0):
                # Do not force-switch user-pinned heavy models.
                llm_device = "cuda"
                tts_device = "cpu"
                reason = "manual_llm_pin"
            else:
                llm_device = "cpu"
                tts_device = "cuda"
                reason = "policy_prioritize_tts"
        elif policy == "prioritize_llm":
            llm_device = "cuda"
            tts_device = "cpu"
            reason = "policy_prioritize_llm"
        else:
            llm_device = "cuda"
            if profile_name == "gpu_full":
                tts_device = "cuda"
                reason = "balanced_gpu_full"
            elif pinned_model and (model_size_b is None or model_size_b > 4.0):
                tts_device = "cpu"
                reason = "manual_llm_pin"
            elif model_size_b is not None and model_size_b <= 3.5 and gpu_vram_gb >= 8.0:
                tts_device = "cuda"
                reason = "balanced_light_llm"
            else:
                tts_device = "cpu"
                reason = "insufficient_vram"

        manual_pref_applied = False
        fallback_due_to_cuda_unavailable = False

        if llm_pref != "auto":
            manual_pref_applied = True
            if llm_pref == "cuda":
                if gpu_enabled:
                    llm_device = "cuda"
                else:
                    llm_device = "cpu"
                    fallback_due_to_cuda_unavailable = True
                    warnings.append("LLM GPU preference is unavailable; falling back to CPU.")
            else:
                llm_device = "cpu"

        if tts_pref != "auto":
            manual_pref_applied = True
            if tts_pref == "cuda":
                if gpu_enabled:
                    tts_device = "cuda"
                else:
                    tts_device = "cpu"
                    fallback_due_to_cuda_unavailable = True
                    warnings.append("TTS GPU preference is unavailable; falling back to CPU.")
            else:
                tts_device = "cpu"

        if manual_pref_applied:
            if fallback_due_to_cuda_unavailable:
                reason = "cuda_unavailable"
            elif llm_pref == "cpu" and tts_pref == "cuda":
                reason = "manual_split_pref"
            else:
                reason = "manual_component_pref"

        if (
            gpu_enabled
            and llm_device == "cuda"
            and tts_device == "cuda"
            and profile_name != "gpu_full"
        ):
            warnings.append(
                "Both LLM and TTS are pinned to GPU on constrained hardware; performance may drop."
            )

        return {
            "policy": policy,
            "reason": reason,
            "llm_device": llm_device,
            "tts_device": tts_device,
            "gpu_vram_gb": gpu_vram_gb,
            "selected_model": selected_model,
            "model_size_b": model_size_b,
            "pinned_model": pinned_model,
            "llm_device_preference": llm_pref,
            "tts_device_preference": tts_pref,
            "allocation_warnings": warnings,
        }

    def _coach_hardware_profile(self) -> dict[str, Any]:
        high_vram_threshold = self._env_float(
            "COACH_HIGH_VRAM_THRESHOLD_GB",
            float(settings.COACH_HIGH_VRAM_THRESHOLD_GB),
            minimum=8.0,
        )
        override = self._env_text("COACH_RUNTIME_PROFILE", settings.COACH_RUNTIME_PROFILE).lower()
        if override not in {"auto", "cpu", "gpu_constrained", "gpu_full"}:
            override = "auto"

        if override == "auto":
            profile_name = recommend_coach_runtime_profile(
                use_gpu=bool(settings.USE_GPU),
                gpu_vram_gb=max(0.0, float(settings.GPU_VRAM or 0.0)),
                high_vram_threshold_gb=high_vram_threshold,
            )
        else:
            profile_name = override

        if profile_name == "gpu_full":
            llm_order = [settings.HEAVY_MODEL, settings.SMART_MODEL, settings.MEDIUM_MODEL, settings.LIGHT_MODEL]
            return {
                "name": profile_name,
                "high_vram_threshold_gb": high_vram_threshold,
                "use_gpu": bool(settings.USE_GPU),
                "gpu_vram_gb": max(0.0, float(settings.GPU_VRAM or 0.0)),
                "llm_primary_model": settings.HEAVY_MODEL,
                "llm_candidate_order": llm_order,
                "asr_device": "cuda",
                "asr_compute_type": "float16",
                "asr_fast_model": "large-v3",
                "asr_accurate_model": "large-v3",
                "asr_retry_enabled": False,
                "asr_retry_threshold": 0.35,
                "asr_preload_accurate": True,
                "tts_device": "cuda",
            }

        if profile_name == "cpu":
            llm_order = [settings.MEDIUM_MODEL, settings.SMART_MODEL, settings.LIGHT_MODEL, settings.HEAVY_MODEL]
            if settings.HARDWARE_TARGET == "arm64":
                llm_order = [settings.LIGHT_MODEL, settings.MEDIUM_MODEL, settings.SMART_MODEL, settings.HEAVY_MODEL]
            return {
                "name": profile_name,
                "high_vram_threshold_gb": high_vram_threshold,
                "use_gpu": bool(settings.USE_GPU),
                "gpu_vram_gb": max(0.0, float(settings.GPU_VRAM or 0.0)),
                "llm_primary_model": llm_order[0],
                "llm_candidate_order": llm_order,
                "asr_device": "cpu",
                "asr_compute_type": "int8",
                "asr_fast_model": "medium",
                "asr_accurate_model": "large-v3",
                "asr_retry_enabled": True,
                "asr_retry_threshold": 0.45,
                "asr_preload_accurate": False,
                "tts_device": "cpu",
            }

        # gpu_constrained
        llm_order = [settings.HEAVY_MODEL, settings.SMART_MODEL, settings.MEDIUM_MODEL, settings.LIGHT_MODEL]
        return {
            "name": "gpu_constrained",
            "high_vram_threshold_gb": high_vram_threshold,
            "use_gpu": bool(settings.USE_GPU),
            "gpu_vram_gb": max(0.0, float(settings.GPU_VRAM or 0.0)),
            "llm_primary_model": settings.HEAVY_MODEL,
            "llm_candidate_order": llm_order,
            "asr_device": "cpu",
            "asr_compute_type": "int8",
            "asr_fast_model": "medium",
            "asr_accurate_model": "large-v3",
            "asr_retry_enabled": True,
            "asr_retry_threshold": 0.45,
            "asr_preload_accurate": False,
            "tts_device": "cpu",
        }

    def _resolve_runtime_device(self, *, env_key: str, configured_default: str, profile_default: str) -> str:
        configured = self._env_text(env_key, configured_default).lower()
        if configured in {"cpu", "cuda"}:
            return configured
        if configured != "auto":
            configured = "auto"
        resolved = profile_default if configured == "auto" else configured
        if resolved == "cuda" and not settings.USE_GPU:
            return "cpu"
        return resolved

    def _normalize_model_candidates(self, models: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for model in models:
            normalized = str(model).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    def get_quality_first_model_order(
        self,
        candidate_order: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if candidate_order is not None:
            return self._normalize_model_candidates(candidate_order)
        profile = self._coach_hardware_profile()
        return self._normalize_model_candidates(profile.get("llm_candidate_order", []))

    def select_model(
        self,
        available_models: Optional[Sequence[str]] = None,
        *,
        candidate_order: Optional[Sequence[str]] = None,
        latency_budget_ms: Optional[float] = None,
        latency_probe: Optional[LatencyProbe] = None,
        latency_fallback: Optional[LatencyFallback] = None,
    ) -> str:
        candidates = self.get_quality_first_model_order(candidate_order)
        if available_models is not None:
            allowed = {str(model).strip() for model in available_models if str(model).strip()}
            candidates = [model for model in candidates if model in allowed]
        if not candidates:
            candidates = self.get_quality_first_model_order(candidate_order)

        if latency_probe is not None and latency_budget_ms is not None:
            for model in candidates:
                observed_latency = latency_probe(model)
                if observed_latency is None or observed_latency <= latency_budget_ms:
                    return model
            if latency_fallback is not None:
                fallback_model = latency_fallback(list(candidates))
                if fallback_model:
                    return fallback_model
            return candidates[-1]

        if latency_fallback is not None and latency_probe is not None:
            fallback_model = latency_fallback(list(candidates))
            if fallback_model:
                return fallback_model

        return candidates[0]

    def _coerce_mapping(self, payload: Any) -> Dict[str, Any]:
        if payload is None:
            return {}
        if hasattr(payload, "model_dump"):
            return dict(payload.model_dump())
        if isinstance(payload, Mapping):
            return dict(payload)
        return dict(payload)

    def _is_low_value_mistake(self, detail: str, *, suggestion: Optional[str] = None) -> bool:
        normalized_detail = str(detail or "").strip().lower()
        if not normalized_detail:
            return True
        match = self.LOW_VALUE_MISSING_CHAR_PATTERN.search(normalized_detail)
        if not match:
            return False
        missing_char = str(match.group("char") or "").strip().lower()
        target_word = str(match.group("word") or "").strip().lower()
        normalized_suggestion = str(suggestion or "").strip().lower()
        if missing_char and target_word and missing_char in target_word:
            return True
        if target_word and normalized_suggestion and normalized_suggestion == target_word:
            return True
        return False

    def _session_counts(self, session: CoachSession) -> tuple[int, int]:
        turn_count = len(session.turns)
        mistake_count = sum(len(turn.mistakes) for turn in session.turns)
        return turn_count, mistake_count

    def _serialize_turn(self, turn: CoachTurn) -> dict[str, Any]:
        return {
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
                for mistake in (turn.mistakes or [])
            ],
        }

    async def create_session(
        self,
        user_id: int,
        title: Optional[str] = None,
        target_language: str = "English",
        native_language: Optional[str] = None,
        cefr_level: str = "A2",
        audio_retention_opt_in: bool = False,
        focus_area: Optional[str] = None,
        model_id: Optional[str] = None,
        voice_profile_id: Optional[str] = None,
        llm_device_preference: Optional[str] = "auto",
        tts_device_preference: Optional[str] = "auto",
    ) -> CoachSession:
        session_id = str(uuid.uuid4())
        normalized_target_language = self._resolve_supported_language_name(target_language)
        if normalized_target_language is None:
            raise ValueError("Selected target_language is not supported by the active coach stack")
        normalized_voice_profile_id = voice_profile_id.strip() if isinstance(voice_profile_id, str) else None
        if normalized_voice_profile_id:
            if normalized_voice_profile_id.startswith("builtin:"):
                builtin_id = normalized_voice_profile_id.split(":", 1)[1].strip().lower()
                if self._builtin_voice_record(builtin_id) is None:
                    raise ValueError("Built-in voice not found")
            else:
                profile = await self.get_voice_profile(profile_id=normalized_voice_profile_id, user_id=user_id)
                if profile is None:
                    raise ValueError("Voice profile not found")
        async with AsyncSessionLocal() as session:
            coach_session = CoachSession(
                id=session_id,
                user_id=user_id,
                title=(title or "Coach Session").strip() or "Coach Session",
                target_language=normalized_target_language,
                native_language=native_language.strip() if isinstance(native_language, str) else native_language,
                cefr_level=(cefr_level or "A2").strip() or "A2",
                audio_retention_opt_in=bool(audio_retention_opt_in),
                focus_area=focus_area.strip() if isinstance(focus_area, str) else focus_area,
                model_id=model_id.strip() if isinstance(model_id, str) else model_id,
                llm_device_preference=self._normalize_device_preference(llm_device_preference),
                tts_device_preference=self._normalize_device_preference(tts_device_preference),
                voice_profile_id=normalized_voice_profile_id,
            )
            session.add(coach_session)
            await session.commit()
            await session.refresh(coach_session)
            return coach_session

    async def list_sessions(self, user_id: int) -> List[dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachSession)
                .where(CoachSession.user_id == user_id)
                .options(selectinload(CoachSession.turns).selectinload(CoachTurn.mistakes))
                .order_by(desc(CoachSession.created_at), desc(CoachSession.id))
            )
            sessions = result.scalars().all()

        response: List[dict] = []
        for coach_session in sessions:
            turn_count, mistake_count = self._session_counts(coach_session)
            response.append(
                {
                    "id": coach_session.id,
                    "user_id": coach_session.user_id,
                    "title": coach_session.title,
                    "target_language": coach_session.target_language,
                    "native_language": coach_session.native_language,
                    "cefr_level": coach_session.cefr_level,
                    "audio_retention_opt_in": coach_session.audio_retention_opt_in,
                    "focus_area": coach_session.focus_area,
                    "model_id": coach_session.model_id,
                    "llm_device_preference": self._normalize_device_preference(
                        getattr(coach_session, "llm_device_preference", "auto")
                    ),
                    "tts_device_preference": self._normalize_device_preference(
                        getattr(coach_session, "tts_device_preference", "auto")
                    ),
                    "voice_profile_id": coach_session.voice_profile_id,
                    "status": coach_session.status,
                    "created_at": coach_session.created_at,
                    "updated_at": coach_session.updated_at,
                    "turn_count": turn_count,
                    "mistake_count": mistake_count,
                }
            )
        return response

    async def update_session_settings(
        self,
        *,
        session_id: str,
        user_id: int,
        model_id: Optional[str] = None,
        llm_device_preference: Optional[str] = "auto",
        tts_device_preference: Optional[str] = "auto",
    ) -> Optional[CoachSession]:
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            return None
        normalized_model_id = str(model_id or "").strip() or None
        normalized_llm_pref = self._normalize_device_preference(llm_device_preference)
        normalized_tts_pref = self._normalize_device_preference(tts_device_preference)

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachSession)
                .where(
                    CoachSession.id == normalized_session_id,
                    CoachSession.user_id == user_id,
                )
                .options(selectinload(CoachSession.turns).selectinload(CoachTurn.mistakes))
            )
            coach_session = result.scalar_one_or_none()
            if coach_session is None:
                return None
            coach_session.model_id = normalized_model_id
            coach_session.llm_device_preference = normalized_llm_pref
            coach_session.tts_device_preference = normalized_tts_pref
            await session.commit()
            await session.refresh(coach_session)
            return coach_session

    async def get_session(self, session_id: str, user_id: int) -> Optional[CoachSession]:
        session_id = str(session_id)
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachSession)
                .where(CoachSession.id == session_id, CoachSession.user_id == user_id)
                .options(selectinload(CoachSession.turns).selectinload(CoachTurn.mistakes))
            )
            return result.scalar_one_or_none()

    async def delete_session(self, session_id: str, user_id: int) -> bool:
        session_id = str(session_id)
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachSession).where(CoachSession.id == session_id, CoachSession.user_id == user_id)
            )
            coach_session = result.scalar_one_or_none()
            if coach_session is None:
                return False
            await session.delete(coach_session)
            await session.commit()
            return True

    async def save_turn_with_mistakes(
        self,
        session_id: str,
        user_id: int,
        transcript: str,
        reply: str,
        score: Optional[int],
        *,
        correction: Optional[str] = None,
        explanation: Optional[str] = None,
        model_id: Optional[str] = None,
        latency_ms: Optional[int] = None,
        mistakes: Optional[Sequence[Any]] = None,
    ) -> Optional[CoachTurn]:
        session_id = str(session_id)
        async with AsyncSessionLocal() as session:
            session_result = await session.execute(
                select(CoachSession)
                .where(CoachSession.id == session_id, CoachSession.user_id == user_id)
                .options(selectinload(CoachSession.turns))
            )
            coach_session = session_result.scalar_one_or_none()
            if not coach_session:
                return None

            turn_index_result = await session.execute(
                select(func.count(CoachTurn.id)).where(CoachTurn.session_id == session_id)
            )
            turn_index = int(turn_index_result.scalar_one() or 0) + 1

            coach_turn = CoachTurn(
                session_id=session_id,
                user_id=user_id,
                turn_index=turn_index,
                role="coach",
                content=reply,
                transcript=transcript,
                reply=reply,
                correction=correction,
                explanation=explanation,
                score=score,
                model_id=model_id.strip() if isinstance(model_id, str) else model_id,
                latency_ms=latency_ms,
            )
            session.add(coach_turn)
            await session.flush()

            for mistake_payload in mistakes or []:
                data = self._coerce_mapping(mistake_payload)
                detail = str(
                    data.get("detail")
                    or data.get("message")
                    or data.get("description")
                    or ""
                ).strip()
                if not detail:
                    continue
                suggestion = (
                    str(data["suggestion"]).strip()
                    if data.get("suggestion") is not None
                    else None
                )
                if self._is_low_value_mistake(detail, suggestion=suggestion):
                    continue

                mistake = CoachMistake(
                    session_id=session_id,
                    turn_id=coach_turn.id,
                    user_id=user_id,
                    category=str(data.get("category") or "general").strip() or "general",
                    detail=detail,
                    severity=str(data.get("severity") or "medium").strip() or "medium",
                    suggestion=suggestion,
                    metadata_json=data.get("metadata") or data.get("metadata_json"),
                )
                session.add(mistake)

            await session.commit()

            result = await session.execute(
                select(CoachTurn)
                .where(CoachTurn.id == coach_turn.id, CoachTurn.user_id == user_id)
                .options(selectinload(CoachTurn.mistakes))
            )
            return result.scalar_one_or_none()

    def _is_transcript_evaluable(self, transcript: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(transcript or "")).strip().lower()
        if not normalized:
            return False
        if re.fullmatch(r"captured\s+\d+\s+bytes\s+of\s+audio", normalized):
            return False
        if normalized in {"(no transcript)", "no transcript", "[silence]", "[noise]"}:
            return False
        if len(normalized) < 10:
            return False

        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", normalized)
        if len(words) < 3:
            return False
        filler_tokens = {
            "uh",
            "um",
            "hmm",
            "huh",
            "ah",
            "eh",
            "erm",
            "mm",
            "cough",
            "coughing",
            "noise",
            "silence",
        }
        meaningful_words = [word for word in words if word not in filler_tokens]
        if len(meaningful_words) < 3:
            return False
        return True

    def _target_language_code(self, target_language: Optional[str]) -> Optional[str]:
        value = str(target_language or "").strip()
        if not value:
            return None

        normalized = value.lower()
        if normalized in self.WHISPER_LANGUAGE_MAP:
            return self.WHISPER_LANGUAGE_MAP[normalized]

        code_match = re.search(r"\b([a-z]{2})\b", normalized)
        if code_match:
            return code_match.group(1)

        return normalized[:2] if len(normalized) >= 2 else None

    def _supported_language_codes(self) -> list[str]:
        asr_codes = set(self.WHISPER_LANGUAGE_MAP.values())
        tts_codes = set(self.TTS_LANGUAGE_VOICE_MAP.keys())
        common = sorted(asr_codes.intersection(tts_codes))
        return [code for code in common if code in self.CODE_TO_LANGUAGE_NAME]

    def supported_languages(self) -> list[dict[str, Any]]:
        enabled_lt, _, _ = self._languagetool_config()
        codes = self._supported_language_codes()
        payload: list[dict[str, Any]] = []
        for code in codes:
            payload.append(
                {
                    "code": code,
                    "name": self.CODE_TO_LANGUAGE_NAME.get(code, code.upper()),
                    "asr_supported": code in self.WHISPER_LANGUAGE_MAP.values(),
                    "tts_supported": code in self.TTS_LANGUAGE_VOICE_MAP,
                    "languagetool_supported": enabled_lt,
                    "selectable": True,
                    "is_default": code == "en",
                }
            )
        return payload

    def _resolve_supported_language_name(self, value: Optional[str]) -> Optional[str]:
        raw = str(value or "").strip()
        if not raw:
            return None
        normalized = raw.lower()
        if normalized in self.CODE_TO_LANGUAGE_NAME:
            normalized_name = self.CODE_TO_LANGUAGE_NAME[normalized]
            for item in self.supported_languages():
                if str(item.get("name") or "").lower() == normalized_name.lower():
                    return normalized_name
        for item in self.supported_languages():
            name = str(item.get("name") or "").strip()
            if name.lower() == normalized:
                return name
        return None

    def _normalize_learner_level(self, value: Optional[str]) -> str:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return "medium"
        if normalized in self.LEARNER_LEVELS:
            return normalized
        return self.LEARNER_LEVEL_ALIASES.get(normalized, "medium")

    def _coaching_level_instructions(self, level: str) -> str:
        normalized = self._normalize_learner_level(level)
        if normalized in {"ignorant", "novice"}:
            return (
                "Learner level is beginner. Be patient, positive, and step-by-step. "
                "Allow the learner to answer in native language when needed, then bridge to target language with simple examples. "
                "Use short practical drills and one tiny next step."
            )
        if normalized == "medium":
            return (
                "Learner level is medium. Give thorough corrections with clear examples. "
                "When helpful, explain in both target and native language."
            )
        if normalized == "high":
            return (
                "Learner level is high. Be rigorous about precision, nuance, and natural phrasing while staying respectful and encouraging. "
                "Critique the learner's sentence, not the learner."
            )
        return (
            "Learner level is fluent. Be highly rigorous and focus on advanced accuracy, style, and naturalness without being dismissive. "
            "Challenge the learner with concise, high-value feedback and actionable alternatives."
        )

    def _should_attach_translation(
        self,
        *,
        target_language: Optional[str],
        native_language: Optional[str],
    ) -> bool:
        target = str(target_language or "").strip().lower()
        native = str(native_language or "").strip()
        if not native:
            return False
        native_supported = self._resolve_supported_language_name(native)
        if native_supported is None:
            return False
        return native_supported.lower() != target

    def _append_native_translation(
        self,
        *,
        message: str,
        translation: Any,
        target_language: Optional[str],
        native_language: Optional[str],
    ) -> str:
        base = re.sub(r"\s+", " ", str(message or "")).strip()
        if not base:
            return base
        if not self._should_attach_translation(
            target_language=target_language,
            native_language=native_language,
        ):
            return base
        translated = self._normalize_translation_payload(
            translation=translation,
            native_language=native_language,
        )
        if not translated:
            return base
        native = self._resolve_supported_language_name(native_language) or str(native_language or "").strip() or "Native"
        if translated.lower() in base.lower():
            return base
        return f"{base}\n[{native}] {translated}"

    def _normalize_translation_payload(
        self,
        *,
        translation: Any,
        native_language: Optional[str],
    ) -> str:
        payload = translation
        if isinstance(payload, str):
            normalized = payload.strip()
            if not normalized:
                return ""
            parsed = self._try_parse_translation_mapping(normalized)
            if parsed is not None:
                payload = parsed
            else:
                return re.sub(r"\s+", " ", normalized)
        if isinstance(payload, Mapping):
            picked = self._pick_translation_from_mapping(
                payload=payload,
                native_language=native_language,
            )
            return re.sub(r"\s+", " ", picked).strip()
        if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray, str)):
            for item in payload:
                rendered = self._normalize_translation_payload(
                    translation=item,
                    native_language=native_language,
                )
                if rendered:
                    return rendered
            return ""
        rendered = re.sub(r"\s+", " ", str(payload or "")).strip()
        if rendered.startswith("{") and rendered.endswith("}"):
            parsed = self._try_parse_translation_mapping(rendered)
            if parsed is not None:
                return self._pick_translation_from_mapping(
                    payload=parsed,
                    native_language=native_language,
                )
        return rendered

    def _try_parse_translation_mapping(self, raw: str) -> Optional[dict[str, Any]]:
        value = str(raw or "").strip()
        if not value:
            return None
        if not (value.startswith("{") and value.endswith("}")):
            match = re.search(r"\{.*\}", value, flags=re.DOTALL)
            if not match:
                return None
            value = match.group(0).strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except Exception:
                continue
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return None

    def _translation_native_code(self, native_language: Optional[str]) -> str:
        raw = str(native_language or "").strip().lower()
        if raw in self.CODE_TO_LANGUAGE_NAME:
            return raw
        if len(raw) == 2 and raw in self.CODE_TO_LANGUAGE_NAME:
            return raw
        native_name = self._resolve_supported_language_name(native_language)
        if native_name:
            normalized_name = native_name.lower()
            for code, name in self.CODE_TO_LANGUAGE_NAME.items():
                if name.lower() == normalized_name:
                    return code
        return ""

    def _coerce_translation_leaf(self, value: Any) -> str:
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip()
        if isinstance(value, Mapping):
            for key in ("text", "translation", "value", "content"):
                rendered = re.sub(r"\s+", " ", str(value.get(key) or "")).strip()
                if rendered:
                    return rendered
            return ""
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            for item in value:
                rendered = self._coerce_translation_leaf(item)
                if rendered:
                    return rendered
            return ""
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _pick_translation_from_mapping(
        self,
        *,
        payload: Mapping[str, Any],
        native_language: Optional[str],
    ) -> str:
        native_name = self._resolve_supported_language_name(native_language) or str(native_language or "").strip()
        native_code = self._translation_native_code(native_language)
        keys: list[str] = []
        if native_code:
            keys.extend(
                [
                    native_code,
                    native_code.lower(),
                    native_code.upper(),
                    native_code.replace("-", "_"),
                    native_code.replace("_", "-"),
                ]
            )
        if native_name:
            keys.extend([native_name, native_name.lower(), native_name.upper()])
        keys.extend(["native", "translation", "text", "message", "value"])

        for key in keys:
            if key not in payload:
                continue
            rendered = self._coerce_translation_leaf(payload.get(key))
            if rendered:
                return rendered

        for value in payload.values():
            rendered = self._coerce_translation_leaf(value)
            if rendered:
                return rendered
        return ""

    def _sanitize_tts_text(self, text: str) -> str:
        lines = [
            re.sub(r"\s+", " ", str(line or "")).strip()
            for line in str(text or "").splitlines()
        ]
        lines = [line for line in lines if line]
        if not lines:
            return ""

        # Keep spoken coach content and skip appended translation-only lines.
        spoken_lines = [
            line for line in lines if not re.match(r"^\[[^\]]+\]\s+.+$", line)
        ]
        if not spoken_lines:
            spoken_lines = [lines[0]]

        spoken = " ".join(spoken_lines)
        spoken = re.sub(r"`{1,3}[^`]*`{1,3}", " ", spoken)
        spoken = re.sub(r"^\s*[-*•]\s*", "", spoken)
        spoken = re.sub(r"\s+", " ", spoken).strip()
        if not spoken:
            return ""

        max_chars_raw = self._env_text("COACH_TTS_MAX_TEXT_CHARS", "600")
        try:
            max_chars = int(max_chars_raw)
        except (TypeError, ValueError):
            max_chars = 600
        max_chars = max(160, min(max_chars, 2000))

        if len(spoken) <= max_chars:
            return spoken

        clipped = spoken[:max_chars]
        sentence_break = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        if sentence_break >= int(max_chars * 0.6):
            return clipped[: sentence_break + 1].strip()
        shortened = clipped.rsplit(" ", 1)[0].strip()
        return shortened or clipped.strip()

    def _normalized_message_key(self, value: Optional[str]) -> str:
        normalized = re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _reply_similarity(self, first: str, second: str) -> float:
        a = self._normalized_message_key(first)
        b = self._normalized_message_key(second)
        if not a or not b:
            return 0.0
        return float(SequenceMatcher(None, a, b).ratio())

    def _persona_display_name(self, persona_style: Optional[str]) -> str:
        normalized = str(persona_style or "").strip().lower()
        for voice_id, label in self.BUILTIN_VOICE_LABELS.items():
            if voice_id in normalized or label.lower() in normalized:
                return label
        return "Coach AI"

    def _is_name_question(self, transcript: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(transcript or "").lower()).strip()
        if not normalized:
            return False
        patterns = (
            r"\bwhat(?:'s| is)\s+your\s+name\b",
            r"\bwho\s+are\s+you\b",
            r"\bur\s+name\b",
            r"\bton\s+nom\b",
            r"\btu\s+t[' ]?appell(?:es)?\b",
            r"\bcomment\s+tu\s+t[' ]?appell(?:es)?\b",
            r"\bcomment\s+vous\s+vous\s+appelez\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _name_reply(self, *, target_language: str, persona_style: Optional[str]) -> str:
        name = self._persona_display_name(persona_style)
        normalized = str(target_language or "").strip().lower()
        if normalized.startswith("french"):
            return f"Je m'appelle {name}."
        return f"My name is {name}."

    def _apply_direct_question_guard(
        self,
        *,
        transcript: str,
        candidate_reply: str,
        target_language: str,
        persona_style: Optional[str],
    ) -> str:
        if not self._is_name_question(transcript):
            return candidate_reply
        name_reply = self._name_reply(target_language=target_language, persona_style=persona_style)
        if self._normalized_message_key(name_reply) in self._normalized_message_key(candidate_reply):
            return candidate_reply
        return f"{name_reply} {candidate_reply}".strip()

    def _sanitize_coach_tone(self, text: str) -> str:
        rendered = re.sub(r"\s+", " ", str(text or "")).strip()
        if not rendered:
            return ""
        replacements = (
            (
                r"\byou(?:'re| are)\s+avoiding\s+the\s+question\b[.!]?\s*",
                "Thanks. Let's make your answer more specific. ",
            ),
            (
                r"\byou(?:'re| are)\s+not\s+answering\s+the\s+question\b[.!]?\s*",
                "Let's answer the question directly with one short sentence. ",
            ),
            (
                r"\bstop\s+avoiding\s+the\s+question\b[.!]?\s*",
                "Let's refocus with one clear sentence. ",
            ),
            (
                r"\bwhy\s+are\s+you\s+avoiding\s+the\s+question\b[!?]?\s*",
                "Let's focus on one direct answer. ",
            ),
        )
        for pattern, replacement in replacements:
            rendered = re.sub(pattern, replacement, rendered, flags=re.IGNORECASE)
        rendered = re.sub(r"\s+", " ", rendered).strip()
        return rendered

    def _prevent_repeat_loop(
        self,
        *,
        candidate_reply: str,
        last_coach_reply: Optional[str],
        target_language: str,
        native_language: Optional[str],
        learner_level: str,
        focus_area: Optional[str],
        learner_transcript: Optional[str] = None,
    ) -> str:
        current_key = self._normalized_message_key(candidate_reply)
        previous_key = self._normalized_message_key(last_coach_reply)
        if not current_key:
            return candidate_reply
        is_repeat = current_key == previous_key
        if not is_repeat and previous_key:
            similarity = self._reply_similarity(candidate_reply, str(last_coach_reply or ""))
            overlap = current_key in previous_key or previous_key in current_key
            is_repeat = similarity >= 0.82 or overlap
        if not is_repeat:
            return candidate_reply

        level = self._normalize_learner_level(learner_level)
        topic = str(focus_area or "this topic").strip() or "this topic"
        learner_hint = re.sub(r"\s+", " ", str(learner_transcript or "")).strip()
        if learner_hint:
            topic = learner_hint
        if level in {"ignorant", "novice"} and native_language:
            native = self._resolve_supported_language_name(native_language) or str(native_language).strip()
            return (
                f"Let's slow down and make it easy. You can answer in {native}, and I'll help you build the same idea in {target_language}. "
                f"What is one simple sentence you want to say about {topic}?"
            )
        if level == "medium":
            return (
                f"Let's reset with one clear step. Give me one short sentence about {topic}, "
                f"then I'll improve it and explain why."
            )
        return (
            f"Let's move to a sharper variation: give one precise sentence about {topic} with better word choice and grammar."
        )

    def _resolve_voice_cipher(self) -> Fernet:
        configured = str(
            os.environ.get("COACH_VOICE_SECRET_KEY", settings.COACH_VOICE_SECRET_KEY) or ""
        ).strip()
        if not configured:
            configured = str(settings.SECRET_KEY)
        if self._voice_cipher is not None and self._voice_cipher_key == configured:
            return self._voice_cipher

        digest = hashlib.sha256(configured.encode("utf-8")).digest()
        key = base64.urlsafe_b64encode(digest)
        self._voice_cipher = Fernet(key)
        self._voice_cipher_key = configured
        return self._voice_cipher

    def _voice_storage_root(self) -> Path:
        root = Path(settings.DATA_DIR) / "coach_voices"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _voice_library_root(self) -> Path:
        configured = str(
            os.environ.get("COACH_VOICE_LIBRARY_DIR", settings.COACH_VOICE_LIBRARY_DIR)
            or settings.COACH_VOICE_LIBRARY_DIR
        ).strip()
        path = Path(configured or (Path(settings.DATA_DIR) / "coach_voice_library"))
        return path

    def _voice_library_samples_dir(self) -> Path:
        return self._voice_library_root() / "clone samples"

    def _voice_library_images_dir(self) -> Path:
        return self._voice_library_root() / "images"

    def _voice_library_lookup(self) -> dict[str, dict]:
        voices: dict[str, dict] = {
            voice_id: {
                "id": voice_id,
                "name": self.BUILTIN_VOICE_LABELS.get(voice_id, voice_id.title()),
                "sample_path": None,
                "avatar_path": None,
            }
            for voice_id in self.BUILTIN_VOICE_ORDER
        }
        sample_dir = self._voice_library_samples_dir()
        if sample_dir.exists():
            for file_path in sample_dir.iterdir():
                if not file_path.is_file():
                    continue
                lower_name = file_path.name.lower()
                for voice_id in self.BUILTIN_VOICE_ORDER:
                    token = f"[{voice_id}]"
                    if token in lower_name or voice_id in lower_name:
                        voices[voice_id]["sample_path"] = file_path
                        break

        image_dir = self._voice_library_images_dir()
        image_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")
        for voice_id in self.BUILTIN_VOICE_ORDER:
            voice = voices[voice_id]
            for ext in image_exts:
                candidate = image_dir / f"{voice_id}{ext}"
                if candidate.exists() and candidate.is_file():
                    voice["avatar_path"] = candidate
                    break
        return voices

    def _avatar_data_url(self, image_path: Optional[Path]) -> Optional[str]:
        if image_path is None or not image_path.exists():
            return None
        try:
            raw = image_path.read_bytes()
        except OSError:
            return None
        if not raw:
            return None
        mime_type, _ = mimetypes.guess_type(str(image_path))
        return f"data:{mime_type or 'image/jpeg'};base64,{base64.b64encode(raw).decode('ascii')}"

    def _serialize_builtin_voice(self, record: Mapping[str, Any]) -> dict:
        voice_id = str(record.get("id") or "").strip().lower()
        sample_path = record.get("sample_path")
        avatar_path = record.get("avatar_path")
        return {
            "id": voice_id,
            "name": str(record.get("name") or voice_id.title()),
            "choice_id": f"builtin:{voice_id}",
            "voice_mode": "preset",
            "voice_preset": "default",
            "provider": "cosyvoice",
            "is_default": voice_id == "anby",
            "is_available": bool(sample_path and Path(sample_path).exists()),
            "avatar_data_url": self._avatar_data_url(Path(avatar_path)) if avatar_path else None,
        }

    async def list_builtin_voices(self) -> List[dict]:
        voices = self._voice_library_lookup()
        serialized = [self._serialize_builtin_voice(voices[voice_id]) for voice_id in self.BUILTIN_VOICE_ORDER]
        return [item for item in serialized if item["is_available"]]

    def _builtin_voice_record(self, voice_id: str) -> Optional[dict]:
        normalized = str(voice_id or "").strip().lower()
        if not normalized:
            return None
        voices = self._voice_library_lookup()
        record = voices.get(normalized)
        if not record:
            return None
        sample_path = record.get("sample_path")
        if not sample_path:
            return None
        return record

    def _voice_user_dir(self, user_id: int, kind: str) -> Path:
        path = self._voice_storage_root() / f"user_{int(user_id)}" / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_audio_extension(self, filename: Optional[str], mime_type: Optional[str]) -> str:
        lower_name = str(filename or "").lower()
        lower_mime = str(mime_type or "").lower()
        if lower_name.endswith(".wav") or "wav" in lower_mime:
            return ".wav"
        if lower_name.endswith(".mp3") or "mpeg" in lower_mime or "mp3" in lower_mime:
            return ".mp3"
        if lower_name.endswith(".m4a") or "m4a" in lower_mime or "mp4" in lower_mime:
            return ".m4a"
        if lower_name.endswith(".ogg") or "ogg" in lower_mime:
            return ".ogg"
        return ".wav"

    def _encrypt_voice_bytes(self, payload: bytes) -> bytes:
        cipher = self._resolve_voice_cipher()
        return cipher.encrypt(payload)

    def _decrypt_voice_bytes(self, payload: bytes) -> bytes:
        cipher = self._resolve_voice_cipher()
        try:
            return cipher.decrypt(payload)
        except InvalidToken as exc:
            raise RuntimeError("Stored voice sample cannot be decrypted with current key.") from exc

    def _serialize_voice_sample(self, sample: CoachVoiceSample) -> dict:
        return {
            "id": sample.id,
            "title": sample.title,
            "mime_type": sample.mime_type,
            "file_size_bytes": sample.file_size_bytes,
            "language": sample.language,
            "created_at": sample.created_at,
        }

    def _serialize_voice_profile(self, profile: CoachVoiceProfile) -> dict:
        return {
            "id": profile.id,
            "name": profile.name,
            "provider": profile.provider,
            "language": profile.language,
            "status": profile.status,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
        }

    async def save_voice_reference(
        self,
        *,
        user_id: int,
        file,
        title: Optional[str],
        language: Optional[str],
    ) -> dict:
        filename = str(getattr(file, "filename", "") or "").strip()
        mime_type = str(getattr(file, "content_type", "") or "application/octet-stream").strip()
        if not filename:
            raise ValueError("An audio file name is required")
        if mime_type and not mime_type.startswith("audio/") and mime_type != "application/octet-stream":
            raise ValueError("Upload an audio file")

        raw = await file.read(self.COACH_VOICE_MAX_AUDIO_BYTES + 1)
        if not raw:
            raise ValueError("Uploaded audio file is empty")
        if len(raw) > self.COACH_VOICE_MAX_AUDIO_BYTES:
            raise ValueError("Voice sample is too large")
        await file.seek(0)

        sample_id = str(uuid.uuid4())
        safe_title = str(title or filename or "Voice sample").strip()[:100] or "Voice sample"
        extension = self._safe_audio_extension(filename=filename, mime_type=mime_type)
        sample_dir = self._voice_user_dir(user_id=user_id, kind="samples")
        sample_path = sample_dir / f"{sample_id}{extension}.enc"
        encrypted = self._encrypt_voice_bytes(raw)
        sample_path.write_bytes(encrypted)

        async with AsyncSessionLocal() as session:
            sample = CoachVoiceSample(
                id=sample_id,
                user_id=user_id,
                title=safe_title,
                file_path=str(sample_path),
                mime_type=mime_type or "audio/wav",
                file_size_bytes=len(raw),
                language=(language or "").strip() or None,
                status="active",
            )
            session.add(sample)
            await session.commit()
            await session.refresh(sample)
            return self._serialize_voice_sample(sample)

    async def _get_active_voice_sample(self, *, sample_id: str, user_id: int) -> Optional[CoachVoiceSample]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachVoiceSample).where(
                    CoachVoiceSample.id == str(sample_id),
                    CoachVoiceSample.user_id == int(user_id),
                    CoachVoiceSample.status == "active",
                )
            )
            return result.scalar_one_or_none()

    async def create_voice_profile(
        self,
        *,
        user_id: int,
        name: str,
        reference_clip_id: str,
        language: Optional[str],
    ) -> dict:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("Profile name is required")

        sample = await self._get_active_voice_sample(sample_id=reference_clip_id, user_id=user_id)
        if sample is None:
            raise ValueError("Reference clip not found")

        sample_path = Path(sample.file_path)
        if not sample_path.exists():
            raise ValueError("Reference clip file is missing")

        profile_id = str(uuid.uuid4())
        profile_dir = self._voice_user_dir(user_id=user_id, kind="profiles")
        source_ext = "".join(sample_path.suffixes) or ".wav.enc"
        profile_path = profile_dir / f"{profile_id}{source_ext}"
        shutil.copyfile(sample_path, profile_path)

        async with AsyncSessionLocal() as session:
            profile = CoachVoiceProfile(
                id=profile_id,
                user_id=user_id,
                sample_id=sample.id,
                name=clean_name[:100],
                provider="cosyvoice",
                file_path=str(profile_path),
                mime_type=sample.mime_type,
                file_size_bytes=sample.file_size_bytes,
                language=(language or sample.language or "").strip() or None,
                status="active",
            )
            session.add(profile)
            await session.commit()
            await session.refresh(profile)
            return self._serialize_voice_profile(profile)

    async def list_voice_profiles(self, *, user_id: int) -> List[dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachVoiceProfile)
                .where(
                    CoachVoiceProfile.user_id == int(user_id),
                    CoachVoiceProfile.status == "active",
                )
                .order_by(desc(CoachVoiceProfile.updated_at), desc(CoachVoiceProfile.created_at))
            )
            profiles = result.scalars().all()
            return [self._serialize_voice_profile(profile) for profile in profiles]

    async def get_voice_profile(self, *, profile_id: str, user_id: int) -> Optional[CoachVoiceProfile]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachVoiceProfile).where(
                    CoachVoiceProfile.id == str(profile_id),
                    CoachVoiceProfile.user_id == int(user_id),
                    CoachVoiceProfile.status == "active",
                )
            )
            return result.scalar_one_or_none()

    async def delete_voice_profile(self, *, profile_id: str, user_id: int) -> bool:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachVoiceProfile).where(
                    CoachVoiceProfile.id == str(profile_id),
                    CoachVoiceProfile.user_id == int(user_id),
                    CoachVoiceProfile.status == "active",
                )
            )
            profile = result.scalar_one_or_none()
            if profile is None:
                return False

            profile.status = "deleted"
            await session.commit()
            return True

    def _load_builtin_voice_reference_bytes(self, *, voice_id: str) -> bytes:
        record = self._builtin_voice_record(voice_id)
        if record is None:
            raise ValueError(f"Built-in voice not found: {voice_id}")
        sample_path = Path(str(record["sample_path"]))
        if not sample_path.exists():
            raise ValueError(f"Built-in voice sample is missing: {voice_id}")
        payload = sample_path.read_bytes()
        if not payload:
            raise ValueError(f"Built-in voice sample is empty: {voice_id}")
        if len(payload) > self.COACH_VOICE_MAX_AUDIO_BYTES:
            raise ValueError("Built-in voice sample is too large")
        return payload

    async def _load_reference_audio_bytes(
        self,
        *,
        user_id: int,
        voice_mode: str,
        voice_profile_id: Optional[str],
        reference_clip_id: Optional[str],
        builtin_voice_id: Optional[str],
    ) -> Optional[bytes]:
        normalized_builtin_voice_id = str(builtin_voice_id or "").strip().lower()
        if normalized_builtin_voice_id:
            return self._load_builtin_voice_reference_bytes(voice_id=normalized_builtin_voice_id)

        target_path: Optional[str] = None
        if voice_mode == "cloned_profile":
            if not voice_profile_id:
                raise ValueError("voice_profile_id is required for cloned_profile mode")
            profile = await self.get_voice_profile(profile_id=voice_profile_id, user_id=user_id)
            if profile is None:
                raise ValueError("Voice profile not found")
            target_path = profile.file_path
        elif voice_mode == "cloned_session":
            if not reference_clip_id:
                raise ValueError("reference_clip_id is required for cloned_session mode")
            sample = await self._get_active_voice_sample(sample_id=reference_clip_id, user_id=user_id)
            if sample is None:
                raise ValueError("Reference clip not found")
            target_path = sample.file_path

        if not target_path:
            return None

        sample_path = Path(target_path)
        if not sample_path.exists():
            raise ValueError("Voice sample file is missing")
        encrypted = sample_path.read_bytes()
        if not encrypted:
            raise ValueError("Voice sample file is empty")
        return self._decrypt_voice_bytes(encrypted)

    def _resolve_tts_voice(
        self,
        *,
        language: Optional[str],
        voice_preset: Optional[str],
    ) -> tuple[str, int, int]:
        language_code = (self._target_language_code(language) or "en").lower()
        base_voice = self.TTS_LANGUAGE_VOICE_MAP.get(language_code, "en-us")
        preset = str(voice_preset or "default").strip().lower()

        voice_name = base_voice
        rate = 165
        pitch = 45

        if preset == "male":
            if base_voice.startswith("en"):
                voice_name = "en-us+m3"
            pitch = 40
        elif preset == "female":
            if base_voice.startswith("en"):
                voice_name = "en-us+f3"
            pitch = 55
        elif preset == "anby":
            # "Anby" is a style preset, not a cloned copyrighted voice.
            if base_voice.startswith("en"):
                voice_name = "en-us+f2"
            rate = 145
            pitch = 38

        return voice_name, rate, pitch

    def _builtin_voice_fallback_preset(
        self,
        *,
        builtin_voice_id: str,
        current_preset: Optional[str],
    ) -> str:
        normalized_current = str(current_preset or "default").strip().lower() or "default"
        if normalized_current != "default":
            return normalized_current

        normalized_voice_id = str(builtin_voice_id or "").strip().lower()
        if normalized_voice_id in {"goku", "gute"}:
            return "male"
        if normalized_voice_id == "anby":
            return "female"
        return "default"

    def _synthesize_with_espeak(
        self,
        *,
        binary: str,
        text: str,
        voice_name: str,
        rate: int,
        pitch: int,
    ) -> bytes:
        output_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_path = temp_file.name

            subprocess.run(
                [
                    binary,
                    "-v",
                    voice_name,
                    "-s",
                    str(rate),
                    "-p",
                    str(pitch),
                    "-w",
                    output_path,
                    text,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with open(output_path, "rb") as output_file:
                audio_bytes = output_file.read()
            if not audio_bytes:
                raise ValueError("TTS generated empty audio")
            return audio_bytes
        except subprocess.CalledProcessError as exc:
            stderr = str(exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(stderr or "TTS engine failed") from exc
        finally:
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

    async def _synthesize_with_cosyvoice(
        self,
        *,
        text: str,
        language: Optional[str],
        voice_preset: Optional[str],
        persona_style: Optional[str],
        voice_mode: str,
        reference_audio_bytes: Optional[bytes],
        builtin_voice_id: Optional[str] = None,
        runtime_device: Optional[str] = None,
    ) -> tuple[bytes, str]:
        base_url, model_id, timeout_sec = self._cosyvoice_runtime_config()

        payload: dict[str, Any] = {
            "text": text,
            "language": language or "English",
            "voice_preset": voice_preset or "default",
            "persona_style": persona_style or "",
            "voice_mode": voice_mode,
            "model_id": model_id,
        }
        normalized_runtime_device = str(runtime_device or "").strip().lower()
        if normalized_runtime_device in {"cpu", "cuda"}:
            payload["runtime_device"] = normalized_runtime_device
        normalized_builtin_voice_id = str(builtin_voice_id or "").strip().lower()
        if normalized_builtin_voice_id:
            payload["builtin_voice_id"] = normalized_builtin_voice_id
        if reference_audio_bytes:
            payload["reference_audio_b64"] = base64.b64encode(reference_audio_bytes).decode("ascii")

        timeout = httpx.Timeout(connect=10.0, read=float(timeout_sec), write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url.rstrip('/')}/synthesize", json=payload)

        if response.status_code >= 400:
            detail = response.text.strip() or f"CosyVoice request failed ({response.status_code})"
            raise RuntimeError(detail)

        audio_bytes = response.content
        if not audio_bytes:
            raise RuntimeError("CosyVoice returned empty audio")
        media_type = response.headers.get("content-type", "audio/wav")
        return audio_bytes, media_type

    async def _synthesize_with_openvoice(
        self,
        *,
        text: str,
        language: Optional[str],
        voice_preset: Optional[str],
        persona_style: Optional[str],
        voice_mode: str,
        reference_audio_bytes: Optional[bytes],
        builtin_voice_id: Optional[str] = None,
        runtime_device: Optional[str] = None,
    ) -> tuple[bytes, str]:
        base_url, model_id, timeout_sec = self._openvoice_runtime_config()
        payload: dict[str, Any] = {
            "text": text,
            "language": language or "English",
            "voice_preset": voice_preset or "default",
            "persona_style": persona_style or "",
            "voice_mode": voice_mode,
            "model_id": model_id,
        }
        normalized_runtime_device = str(runtime_device or "").strip().lower()
        if normalized_runtime_device in {"cpu", "cuda"}:
            payload["runtime_device"] = normalized_runtime_device
        normalized_builtin_voice_id = str(builtin_voice_id or "").strip().lower()
        if normalized_builtin_voice_id:
            payload["builtin_voice_id"] = normalized_builtin_voice_id
        if reference_audio_bytes:
            payload["reference_audio_b64"] = base64.b64encode(reference_audio_bytes).decode("ascii")

        timeout = httpx.Timeout(connect=10.0, read=float(timeout_sec), write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url.rstrip('/')}/synthesize", json=payload)

        if response.status_code >= 400:
            detail = response.text.strip() or f"OpenVoice request failed ({response.status_code})"
            raise RuntimeError(detail)

        audio_bytes = response.content
        if not audio_bytes:
            raise RuntimeError("OpenVoice returned empty audio")
        media_type = response.headers.get("content-type", "audio/wav")
        return audio_bytes, media_type

    def _cosyvoice_runtime_config(self) -> tuple[str, str, int]:
        base_url = str(
            os.environ.get("COACH_COSYVOICE_BASE_URL", settings.COACH_COSYVOICE_BASE_URL)
            or settings.COACH_COSYVOICE_BASE_URL
        ).strip()
        model_id = str(
            os.environ.get("COACH_COSYVOICE_MODEL_ID", settings.COACH_COSYVOICE_MODEL_ID)
            or settings.COACH_COSYVOICE_MODEL_ID
        ).strip()
        timeout_raw = os.environ.get("COACH_COSYVOICE_TIMEOUT_SEC", settings.COACH_COSYVOICE_TIMEOUT_SEC)
        try:
            timeout_sec = max(5, int(timeout_raw))
        except (TypeError, ValueError):
            timeout_sec = int(settings.COACH_COSYVOICE_TIMEOUT_SEC)
        if not base_url:
            base_url = settings.COACH_COSYVOICE_BASE_URL
        if not model_id:
            model_id = settings.COACH_COSYVOICE_MODEL_ID
        return base_url, model_id, timeout_sec

    def _openvoice_runtime_config(self) -> tuple[str, str, int]:
        base_url = str(
            os.environ.get("COACH_OPENVOICE_BASE_URL", settings.COACH_OPENVOICE_BASE_URL)
            or settings.COACH_OPENVOICE_BASE_URL
        ).strip()
        model_id = str(
            os.environ.get("COACH_OPENVOICE_MODEL_ID", settings.COACH_OPENVOICE_MODEL_ID)
            or settings.COACH_OPENVOICE_MODEL_ID
        ).strip()
        timeout_raw = os.environ.get("COACH_OPENVOICE_TIMEOUT_SEC", settings.COACH_OPENVOICE_TIMEOUT_SEC)
        try:
            timeout_sec = max(5, int(timeout_raw))
        except (TypeError, ValueError):
            timeout_sec = int(settings.COACH_OPENVOICE_TIMEOUT_SEC)
        if not base_url:
            base_url = settings.COACH_OPENVOICE_BASE_URL
        if not model_id:
            model_id = settings.COACH_OPENVOICE_MODEL_ID
        return base_url, model_id, timeout_sec

    def _coach_runtime_mode(self, mode: Optional[str]) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in {"voice", "text", "idle"}:
            return normalized
        return "voice"

    def _languagetool_config(self) -> tuple[bool, str, int]:
        enabled = str(
            os.environ.get("COACH_LANGUAGETOOL_ENABLED", str(settings.COACH_LANGUAGETOOL_ENABLED))
            or str(settings.COACH_LANGUAGETOOL_ENABLED)
        ).strip().lower() in {"1", "true", "yes", "on"}
        base_url = str(
            os.environ.get("COACH_LANGUAGETOOL_BASE_URL", settings.COACH_LANGUAGETOOL_BASE_URL)
            or settings.COACH_LANGUAGETOOL_BASE_URL
        ).strip()
        timeout_raw = os.environ.get("COACH_LANGUAGETOOL_TIMEOUT_SEC", settings.COACH_LANGUAGETOOL_TIMEOUT_SEC)
        try:
            timeout_sec = max(2, int(timeout_raw))
        except (TypeError, ValueError):
            timeout_sec = int(settings.COACH_LANGUAGETOOL_TIMEOUT_SEC)
        return enabled, base_url, timeout_sec

    async def _languagetool_check(
        self,
        *,
        text: str,
        language: Optional[str],
    ) -> list[dict[str, Any]]:
        enabled, base_url, timeout_sec = self._languagetool_config()
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not enabled or not base_url or not normalized:
            return []

        language_code = (self._target_language_code(language) or "en").lower()
        lt_defaults = {
            "en": "en-US",
            "fr": "fr",
            "es": "es",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "ru": "ru",
            "ar": "ar",
            "ja": "ja",
            "ko": "ko",
            "zh": "zh-CN",
            "tr": "tr",
            "nl": "nl",
            "pl": "pl",
            "uk": "uk",
            "sv": "sv",
            "el": "el",
            "hi": "hi",
            "ur": "ur",
        }
        lt_language = lt_defaults.get(language_code, language_code)
        timeout = httpx.Timeout(connect=2.0, read=float(timeout_sec), write=5.0, pool=5.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{base_url.rstrip('/')}/v2/check",
                    data={"language": lt_language, "text": normalized},
                )
        except Exception as exc:
            logger.debug("LanguageTool unavailable: %s", exc)
            return []
        if response.status_code >= 400:
            logger.debug("LanguageTool returned status %s", response.status_code)
            return []

        payload: dict[str, Any] = {}
        try:
            parsed = response.json()
            if isinstance(parsed, Mapping):
                payload = dict(parsed)
        except Exception:
            return []

        matches = payload.get("matches")
        if not isinstance(matches, list):
            return []

        findings: list[dict[str, Any]] = []
        for match in matches[:5]:
            if not isinstance(match, Mapping):
                continue
            rule = match.get("rule") if isinstance(match.get("rule"), Mapping) else {}
            detail = str(match.get("message") or "").strip()
            if not detail:
                continue
            suggestion = ""
            replacements = match.get("replacements")
            if isinstance(replacements, list) and replacements:
                first = replacements[0]
                if isinstance(first, Mapping):
                    suggestion = str(first.get("value") or "").strip()
            if self._is_low_value_mistake(detail, suggestion=suggestion):
                continue
            findings.append(
                {
                    "category": "grammar",
                    "detail": detail,
                    "severity": "low",
                    "suggestion": suggestion or None,
                    "metadata": {
                        "source": "languagetool",
                        "rule_id": str(rule.get("id") or ""),
                        "issue_type": str(rule.get("issueType") or ""),
                    },
                }
            )
        return findings

    async def _languagetool_status(self) -> dict[str, Any]:
        enabled, base_url, timeout_sec = self._languagetool_config()
        if not enabled:
            return {
                "engine": "languagetool",
                "enabled": False,
                "ready": False,
                "state": "disabled",
                "detail": "LanguageTool is disabled.",
            }

        timeout = httpx.Timeout(connect=2.0, read=float(timeout_sec), write=5.0, pool=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url.rstrip('/')}/v2/languages")
            if response.status_code >= 400:
                raise RuntimeError(f"status={response.status_code}")
            return {
                "engine": "languagetool",
                "enabled": True,
                "ready": True,
                "state": "ready",
                "detail": "LanguageTool ready.",
            }
        except Exception as exc:
            return {
                "engine": "languagetool",
                "enabled": True,
                "ready": False,
                "state": "error",
                "detail": f"LanguageTool unavailable: {exc}",
            }

    async def _ollama_status(
        self,
        *,
        warm: bool,
        mode: str,
        preferred_model: Optional[str] = None,
        llm_device_preference: Optional[str] = None,
        tts_device_preference: Optional[str] = None,
    ) -> dict[str, Any]:
        profile = self._coach_hardware_profile()
        requested_model = str(preferred_model or "").strip()
        selected_model = str(requested_model or profile.get("llm_primary_model") or settings.HEAVY_MODEL)
        if mode == "idle":
            selected_model = settings.SMART_MODEL
        allocation = self._runtime_allocation(
            profile=profile,
            mode=mode,
            preferred_model=selected_model,
            llm_device_preference=llm_device_preference,
            tts_device_preference=tts_device_preference,
        )
        planned_llm_device = str(allocation.get("llm_device") or "cpu")
        timeout = httpx.Timeout(connect=3.0, read=15.0, write=10.0, pool=10.0)
        status: dict[str, Any] = {
            "engine": "ollama",
            "ready": False,
            "state": "error",
            "detail": "Ollama unavailable.",
            "selected_model": selected_model,
            "selection_reason": "requested_model" if requested_model and requested_model.lower() != "auto" else "profile_default",
            "available_models": [],
            "runtime_profile": profile.get("name"),
            "runtime_device": planned_llm_device,
            "planned_runtime_device": planned_llm_device,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                tags_response = await client.get(f"{self.base_url}/api/tags")
                tags_response.raise_for_status()
                tags_payload = tags_response.json() if tags_response.content else {}
                model_rows = tags_payload.get("models") if isinstance(tags_payload, Mapping) else []
                models = [
                    str(item.get("name") or "").strip()
                    for item in (model_rows or [])
                    if isinstance(item, Mapping) and str(item.get("name") or "").strip()
                ]
                status["available_models"] = models
                status["ready"] = bool(models)
                status["state"] = "ready" if models else "idle"
                status["detail"] = "Ollama ready." if models else "No Ollama models installed."
                if models and selected_model not in models:
                    fallback_model = models[0]
                    status["selection_reason"] = (
                        "requested_model_unavailable"
                        if requested_model and requested_model.lower() != "auto"
                        else "profile_model_unavailable"
                    )
                    status["detail"] = f"Using available Ollama model {fallback_model}."
                    selected_model = fallback_model
                    status["selected_model"] = selected_model

                if warm and selected_model:
                    keep_alive = str(
                        os.environ.get("COACH_OLLAMA_KEEPALIVE", settings.COACH_OLLAMA_KEEPALIVE)
                        or settings.COACH_OLLAMA_KEEPALIVE
                    ).strip() or "15m"
                    warm_payload = {
                        "model": selected_model,
                        "prompt": "hello",
                        "stream": False,
                        "keep_alive": keep_alive,
                        "options": {
                            "num_predict": 1,
                            "temperature": 0.1,
                        },
                    }
                    num_gpu = self._ollama_num_gpu_option(planned_llm_device)
                    if num_gpu is not None:
                        warm_payload["options"]["num_gpu"] = num_gpu
                    warm_response = await client.post(f"{self.base_url}/api/generate", json=warm_payload)
                    if warm_response.status_code >= 400:
                        status["state"] = "warming"
                        status["detail"] = f"Ollama warmup pending for {selected_model}."
                    else:
                        status["state"] = "ready"
                        status["detail"] = f"Ollama model warmed: {selected_model}"
        except Exception as exc:
            status["detail"] = f"Ollama unavailable: {exc}"
        return status

    def _asr_warmup_in_progress(self) -> bool:
        task = self._asr_warmup_task
        return bool(task is not None and not task.done())

    def _ensure_asr_warmup(self, *, preload_accurate: bool) -> bool:
        if self._whisper_model is not None:
            if not preload_accurate:
                return False
            accurate_model_name = self._accurate_whisper_model_ref()
            fast_model_name = self._fast_whisper_model_ref()
            if accurate_model_name == fast_model_name or self._whisper_accurate_model is not None:
                return False
        if self._asr_warmup_in_progress():
            return False

        async def runner() -> None:
            try:
                await self._get_whisper_model()
                if preload_accurate:
                    await self._get_whisper_accurate_model()
            except Exception as exc:
                logger.warning("ASR warmup task failed: %s", exc)

        task = asyncio.create_task(runner())
        self._asr_warmup_task = task

        def _clear_if_current(done_task: asyncio.Task[Any]) -> None:
            if self._asr_warmup_task is done_task:
                self._asr_warmup_task = None

        task.add_done_callback(_clear_if_current)
        return True

    async def _asr_status(self, *, warm: bool, auto_warm: bool = False) -> dict[str, Any]:
        profile = self._coach_hardware_profile()
        fast_model_name = self._fast_whisper_model_ref()
        accurate_model_name = self._accurate_whisper_model_ref()
        preload_accurate = bool(profile.get("asr_preload_accurate"))
        warmup_requested = False
        if warm:
            await self._get_whisper_model()
            if preload_accurate:
                await self._get_whisper_accurate_model()
            warmup_requested = True
        elif auto_warm:
            warmup_requested = self._ensure_asr_warmup(preload_accurate=preload_accurate)
        warmup_active = self._asr_warmup_in_progress()
        accurate_ready = self._whisper_accurate_model is not None
        if warm and accurate_model_name == fast_model_name:
            accurate_ready = self._whisper_model is not None
        ready = self._whisper_model is not None
        state = "ready" if ready else ("warming" if (warm or auto_warm or warmup_active) else "idle")
        if ready:
            detail = "ASR model ready."
        elif warmup_active:
            detail = "ASR model warming in background."
        elif warmup_requested or warm or auto_warm:
            detail = "ASR warmup requested."
        else:
            detail = "ASR model not loaded yet."
        return {
            "engine": "faster-whisper",
            "ready": ready,
            "state": state,
            "detail": detail,
            "model_id": fast_model_name,
            "runtime_device": self._whisper_runtime_device or "",
            "planned_runtime_device": str(profile.get("asr_device") or ""),
            "runtime_profile": str(profile.get("name") or ""),
            "fast_model_id": fast_model_name,
            "accurate_model_id": accurate_model_name,
            "fast_model_ready": ready,
            "accurate_model_ready": accurate_ready,
            "warmup_active": warmup_active,
        }

    async def get_runtime_status(
        self,
        *,
        warm: bool = False,
        mode: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[int] = None,
        preferred_model: Optional[str] = None,
        preferred_tts_provider: Optional[str] = None,
        llm_device_preference: Optional[str] = "auto",
        tts_device_preference: Optional[str] = "auto",
    ) -> dict[str, Any]:
        profile = self._coach_hardware_profile()
        runtime_mode = self._coach_runtime_mode(mode)
        want_voice = runtime_mode == "voice"
        persisted_llm_pref = "auto"
        persisted_tts_pref = "auto"
        normalized_session_id = str(session_id or "").strip()
        if normalized_session_id and user_id is not None:
            session = await self.get_session(session_id=normalized_session_id, user_id=int(user_id))
            if session is not None:
                persisted_llm_pref = self._normalize_device_preference(
                    getattr(session, "llm_device_preference", "auto")
                )
                persisted_tts_pref = self._normalize_device_preference(
                    getattr(session, "tts_device_preference", "auto")
                )
        effective_llm_pref = self._effective_device_preference(
            explicit_preference=llm_device_preference,
            persisted_preference=persisted_llm_pref,
        )
        effective_tts_pref = self._effective_device_preference(
            explicit_preference=tts_device_preference,
            persisted_preference=persisted_tts_pref,
        )
        tts_task = asyncio.create_task(
            self.get_tts_status(
                warm=(warm and want_voice),
                session_id=normalized_session_id or None,
                user_id=user_id,
                preferred_model=preferred_model,
                preferred_tts_provider=preferred_tts_provider,
                llm_device_preference=effective_llm_pref,
                tts_device_preference=effective_tts_pref,
            )
        )
        if want_voice and not warm:
            asr_task = asyncio.create_task(self._asr_status(warm=False, auto_warm=True))
        else:
            asr_task = asyncio.create_task(self._asr_status(warm=(warm and want_voice)))
        llm_task = asyncio.create_task(
            self._ollama_status(
                warm=warm,
                mode=runtime_mode,
                preferred_model=preferred_model,
                llm_device_preference=effective_llm_pref,
                tts_device_preference=effective_tts_pref,
            )
        )
        lt_task = asyncio.create_task(self._languagetool_status())
        tts_status, asr_status, llm_status, lt_status = await asyncio.gather(
            tts_task,
            asr_task,
            llm_task,
            lt_task,
        )
        allocation = self._runtime_allocation(
            profile=profile,
            mode=runtime_mode,
            preferred_model=preferred_model or str(llm_status.get("selected_model") or ""),
            llm_device_preference=effective_llm_pref,
            tts_device_preference=effective_tts_pref,
        )
        components = {
            "tts": tts_status,
            "asr": asr_status,
            "llm": llm_status,
            "languagetool": lt_status,
        }
        critical = ("llm",)
        if want_voice:
            critical = ("llm", "asr", "tts")
        ready = all(bool(components[name].get("ready")) for name in critical)
        state = "ready" if ready else "warming"
        return {
            "ok": True,
            "mode": runtime_mode,
            "ready": ready,
            "state": state,
            "allocation_policy": str(allocation.get("policy") or "auto_balance"),
            "allocation_reason": str(allocation.get("reason") or ""),
            "allocation_warnings": list(allocation.get("allocation_warnings") or []),
            "llm_device": str(allocation.get("llm_device") or ""),
            "tts_device": str(allocation.get("tts_device") or ""),
            "llm_device_preference": effective_llm_pref,
            "tts_device_preference": effective_tts_pref,
            "vram_snapshot": {
                "gpu_enabled": bool(profile.get("use_gpu")),
                "gpu_vram_gb": profile.get("gpu_vram_gb"),
                "high_vram_threshold_gb": profile.get("high_vram_threshold_gb"),
            },
            "runtime_profile": {
                "name": str(profile.get("name") or ""),
                "gpu_enabled": bool(profile.get("use_gpu")),
                "gpu_vram_gb": profile.get("gpu_vram_gb"),
                "high_vram_threshold_gb": profile.get("high_vram_threshold_gb"),
                "llm_primary_model": str(profile.get("llm_primary_model") or ""),
                "asr_device": str(profile.get("asr_device") or ""),
                "asr_fast_model": str(profile.get("asr_fast_model") or ""),
                "asr_accurate_model": str(profile.get("asr_accurate_model") or ""),
                "tts_device": str(profile.get("tts_device") or ""),
                "allocation_policy": str(allocation.get("policy") or "auto_balance"),
                "llm_device_preference": effective_llm_pref,
                "tts_device_preference": effective_tts_pref,
            },
            "components": components,
        }

    async def preload_runtime(self, *, mode: Optional[str] = None) -> dict[str, Any]:
        runtime_mode = self._coach_runtime_mode(mode)
        if runtime_mode == "idle":
            return await self.get_runtime_status(warm=False, mode=runtime_mode)
        return await self.get_runtime_status(warm=True, mode=runtime_mode)

    async def get_tts_status(
        self,
        *,
        warm: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[int] = None,
        preferred_model: Optional[str] = None,
        preferred_tts_provider: Optional[str] = None,
        llm_device_preference: Optional[str] = "auto",
        tts_device_preference: Optional[str] = "auto",
    ) -> dict[str, Any]:
        profile = self._coach_hardware_profile()
        persisted_llm_pref = "auto"
        persisted_tts_pref = "auto"
        normalized_session_id = str(session_id or "").strip()
        if normalized_session_id and user_id is not None:
            session = await self.get_session(session_id=normalized_session_id, user_id=int(user_id))
            if session is not None:
                persisted_llm_pref = self._normalize_device_preference(
                    getattr(session, "llm_device_preference", "auto")
                )
                persisted_tts_pref = self._normalize_device_preference(
                    getattr(session, "tts_device_preference", "auto")
                )
        effective_llm_pref = self._effective_device_preference(
            explicit_preference=llm_device_preference,
            persisted_preference=persisted_llm_pref,
        )
        effective_tts_pref = self._effective_device_preference(
            explicit_preference=tts_device_preference,
            persisted_preference=persisted_tts_pref,
        )
        allocation = self._runtime_allocation(
            profile=profile,
            mode="voice",
            preferred_model=preferred_model,
            llm_device_preference=effective_llm_pref,
            tts_device_preference=effective_tts_pref,
        )
        provider = str(
            preferred_tts_provider
            or os.environ.get("COACH_TTS_PROVIDER", settings.COACH_TTS_PROVIDER)
            or settings.COACH_TTS_PROVIDER
        ).strip().lower() or "espeak"
        if provider not in {"espeak", "cosyvoice", "openvoice"}:
            provider = "espeak"
        device_env_key = "COACH_COSYVOICE_DEVICE"
        configured_default = settings.COACH_COSYVOICE_DEVICE
        if provider == "openvoice":
            device_env_key = "COACH_OPENVOICE_DEVICE"
            configured_default = settings.COACH_OPENVOICE_DEVICE
        desired_device = self._resolve_runtime_device(
            env_key=device_env_key,
            configured_default=configured_default,
            profile_default=str(allocation.get("tts_device") or profile.get("tts_device") or "cpu"),
        )
        fallback_reason = ""
        if str(allocation.get("tts_device") or "").strip().lower() == "cpu":
            fallback_reason = str(allocation.get("reason") or "")
        if provider == "espeak":
            return {
                "ok": True,
                "engine": "espeak",
                "provider": provider,
                "ready": True,
                "state": "ready",
                "detail": "Local espeak voice is ready.",
                "model_id": "",
                "loaded_model_id": "",
                "warmup_active": False,
                "updated_at": None,
                "runtime_device": desired_device,
                "planned_runtime_device": desired_device,
                "fallback_reason": fallback_reason,
                "allocation_policy": str(allocation.get("policy") or "auto_balance"),
                "allocation_reason": str(allocation.get("reason") or ""),
                "allocation_warnings": list(allocation.get("allocation_warnings") or []),
                "llm_device_preference": effective_llm_pref,
                "tts_device_preference": effective_tts_pref,
            }

        engine_name = "cosyvoice"
        base_url, model_id, timeout_sec = self._cosyvoice_runtime_config()
        if provider == "openvoice":
            engine_name = "openvoice"
            base_url, model_id, timeout_sec = self._openvoice_runtime_config()
        timeout = httpx.Timeout(connect=5.0, read=min(float(timeout_sec), 15.0), write=10.0, pool=10.0)
        params = {"runtime_device": desired_device}
        if warm:
            params["warm"] = "true"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url.rstrip('/')}/status", params=params)
        except Exception as exc:
            return {
                "ok": False,
                "engine": engine_name,
                "provider": provider,
                "ready": False,
                "state": "error",
                "detail": f"Voice service unreachable: {exc}",
                "model_id": model_id,
                "loaded_model_id": "",
                "warmup_active": False,
                "updated_at": None,
                "runtime_device": desired_device,
                "planned_runtime_device": desired_device,
                "fallback_reason": fallback_reason or "service_unreachable",
                "allocation_policy": str(allocation.get("policy") or "auto_balance"),
                "allocation_reason": str(allocation.get("reason") or ""),
                "allocation_warnings": list(allocation.get("allocation_warnings") or []),
                "llm_device_preference": effective_llm_pref,
                "tts_device_preference": effective_tts_pref,
            }

        if response.status_code >= 400:
            detail = response.text.strip() or f"Voice status failed ({response.status_code})"
            return {
                "ok": False,
                "engine": engine_name,
                "provider": provider,
                "ready": False,
                "state": "error",
                "detail": detail,
                "model_id": model_id,
                "loaded_model_id": "",
                "warmup_active": False,
                "updated_at": None,
                "runtime_device": desired_device,
                "planned_runtime_device": desired_device,
                "fallback_reason": fallback_reason or "service_error",
                "allocation_policy": str(allocation.get("policy") or "auto_balance"),
                "allocation_reason": str(allocation.get("reason") or ""),
                "allocation_warnings": list(allocation.get("allocation_warnings") or []),
                "llm_device_preference": effective_llm_pref,
                "tts_device_preference": effective_tts_pref,
            }

        payload: dict[str, Any] = {}
        try:
            parsed = response.json()
            if isinstance(parsed, Mapping):
                payload = dict(parsed)
        except Exception:
            payload = {}

        ready = bool(payload.get("ready"))
        state = str(payload.get("state") or ("ready" if ready else "loading")).strip().lower()
        detail = str(payload.get("detail") or "").strip()
        if not detail:
            detail = "Voice model ready." if ready else "Voice model is preparing."
        runtime_device = str(payload.get("runtime_device") or desired_device)
        if desired_device == "cuda" and runtime_device != "cuda":
            fallback_reason = fallback_reason or "cuda_unavailable"
        return {
            "ok": bool(payload.get("ok", True)),
            "engine": str(payload.get("engine") or engine_name),
            "provider": provider,
            "ready": ready,
            "state": state,
            "detail": detail,
            "model_id": str(payload.get("model_id") or model_id),
            "loaded_model_id": str(payload.get("loaded_model_id") or ""),
            "warmup_active": bool(payload.get("warmup_active")),
            "updated_at": payload.get("updated_at"),
            "runtime_device": runtime_device,
            "planned_runtime_device": desired_device,
            "fallback_reason": fallback_reason,
            "allocation_policy": str(allocation.get("policy") or "auto_balance"),
            "allocation_reason": str(allocation.get("reason") or ""),
            "allocation_warnings": list(allocation.get("allocation_warnings") or []),
            "llm_device_preference": effective_llm_pref,
            "tts_device_preference": effective_tts_pref,
        }

    async def synthesize_reply_audio(
        self,
        *,
        text: str,
        language: Optional[str],
        voice_preset: Optional[str],
        persona_style: Optional[str] = None,
        tts_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        voice_mode: Optional[str] = "preset",
        voice_profile_id: Optional[str] = None,
        reference_clip_id: Optional[str] = None,
        builtin_voice_id: Optional[str] = None,
        session_id: Optional[str] = None,
        llm_device_preference: Optional[str] = "auto",
        tts_device_preference: Optional[str] = "auto",
        user_id: Optional[int] = None,
    ) -> tuple[bytes, str]:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            raise ValueError("Text is empty")
        spoken_text = self._sanitize_tts_text(normalized)
        if not spoken_text:
            raise ValueError("Text is empty")
        if spoken_text != normalized:
            logger.debug("Sanitized coach TTS text from %s chars to %s chars", len(normalized), len(spoken_text))

        profile = self._coach_hardware_profile()
        persisted_llm_pref = "auto"
        persisted_tts_pref = "auto"
        normalized_session_id = str(session_id or "").strip()
        if normalized_session_id and user_id is not None:
            session = await self.get_session(session_id=normalized_session_id, user_id=int(user_id))
            if session is not None:
                persisted_llm_pref = self._normalize_device_preference(
                    getattr(session, "llm_device_preference", "auto")
                )
                persisted_tts_pref = self._normalize_device_preference(
                    getattr(session, "tts_device_preference", "auto")
                )
        effective_llm_pref = self._effective_device_preference(
            explicit_preference=llm_device_preference,
            persisted_preference=persisted_llm_pref,
        )
        effective_tts_pref = self._effective_device_preference(
            explicit_preference=tts_device_preference,
            persisted_preference=persisted_tts_pref,
        )
        allocation = self._runtime_allocation(
            profile=profile,
            mode="voice",
            preferred_model=preferred_model,
            llm_device_preference=effective_llm_pref,
            tts_device_preference=effective_tts_pref,
        )
        clipped = spoken_text
        resolved_provider = str(
            tts_provider
            or os.environ.get("COACH_TTS_PROVIDER", settings.COACH_TTS_PROVIDER)
            or settings.COACH_TTS_PROVIDER
        ).strip().lower()
        if resolved_provider not in {"espeak", "cosyvoice", "openvoice"}:
            resolved_provider = "espeak"
        device_env_key = "COACH_COSYVOICE_DEVICE"
        configured_default = settings.COACH_COSYVOICE_DEVICE
        if resolved_provider == "openvoice":
            device_env_key = "COACH_OPENVOICE_DEVICE"
            configured_default = settings.COACH_OPENVOICE_DEVICE
        desired_tts_device = self._resolve_runtime_device(
            env_key=device_env_key,
            configured_default=configured_default,
            profile_default=str(allocation.get("tts_device") or profile.get("tts_device") or "cpu"),
        )
        resolved_mode = str(voice_mode or "preset").strip().lower() or "preset"
        if resolved_mode not in {"preset", "cloned_profile", "cloned_session"}:
            raise ValueError("Unsupported voice_mode")
        normalized_builtin_voice_id = str(builtin_voice_id or "").strip().lower()
        if normalized_builtin_voice_id and self._builtin_voice_record(normalized_builtin_voice_id) is None:
            raise ValueError(f"Built-in voice not found: {normalized_builtin_voice_id}")
        resolved_preset = str(voice_preset or "default").strip().lower()
        if "anby" in str(persona_style or "").lower() and resolved_preset == "default":
            resolved_preset = "anby"
        language_code = (self._target_language_code(language) or "en").lower()
        if (
            resolved_provider == "cosyvoice"
            and normalized_builtin_voice_id
            and language_code not in self.BUILTIN_CLONE_SAFE_LANGUAGE_CODES
        ):
            fallback_preset = self._builtin_voice_fallback_preset(
                builtin_voice_id=normalized_builtin_voice_id,
                current_preset=resolved_preset,
            )
            logger.info(
                "Builtin clone voice '%s' is not language-safe for %s; using local preset '%s' instead.",
                normalized_builtin_voice_id,
                language_code,
                fallback_preset,
            )
            resolved_provider = "espeak"
            resolved_preset = fallback_preset
            normalized_builtin_voice_id = ""

        if resolved_provider == "cosyvoice":
            if user_id is None:
                raise ValueError("user_id is required for cosyvoice synthesis")
            cosyvoice_mode = resolved_mode
            if normalized_builtin_voice_id and cosyvoice_mode == "preset":
                # Built-in library voices are driven through clone reference samples.
                cosyvoice_mode = "cloned_session"
            requires_clone_voice = bool(normalized_builtin_voice_id) or cosyvoice_mode in {"cloned_profile", "cloned_session"}
            reference_audio_bytes: Optional[bytes] = None
            if not normalized_builtin_voice_id:
                reference_audio_bytes = await self._load_reference_audio_bytes(
                    user_id=user_id,
                    voice_mode=cosyvoice_mode,
                    voice_profile_id=voice_profile_id,
                    reference_clip_id=reference_clip_id,
                    builtin_voice_id=normalized_builtin_voice_id,
                )
            try:
                return await self._synthesize_with_cosyvoice(
                    text=clipped,
                    language=language,
                    voice_preset=resolved_preset,
                    persona_style=persona_style,
                    voice_mode=cosyvoice_mode,
                    reference_audio_bytes=reference_audio_bytes,
                    builtin_voice_id=normalized_builtin_voice_id or None,
                    runtime_device=desired_tts_device,
                )
            except Exception:
                if requires_clone_voice:
                    raise RuntimeError(
                        "Selected clone voice is unavailable right now. CosyVoice cloning failed; local fallback is disabled for clone voices."
                    )
                allow_fallback = str(
                    os.environ.get(
                        "COACH_TTS_ALLOW_ESPEAK_FALLBACK",
                        str(settings.COACH_TTS_ALLOW_ESPEAK_FALLBACK),
                    )
                ).strip().lower() in {"1", "true", "yes", "on"}
                if not allow_fallback:
                    raise

        if resolved_provider == "openvoice":
            if user_id is None:
                raise ValueError("user_id is required for openvoice synthesis")
            openvoice_mode = resolved_mode
            if normalized_builtin_voice_id and openvoice_mode == "preset":
                openvoice_mode = "cloned_session"
            requires_clone_voice = bool(normalized_builtin_voice_id) or openvoice_mode in {"cloned_profile", "cloned_session"}
            reference_audio_bytes: Optional[bytes] = None
            if not normalized_builtin_voice_id:
                reference_audio_bytes = await self._load_reference_audio_bytes(
                    user_id=user_id,
                    voice_mode=openvoice_mode,
                    voice_profile_id=voice_profile_id,
                    reference_clip_id=reference_clip_id,
                    builtin_voice_id=normalized_builtin_voice_id,
                )
            try:
                return await self._synthesize_with_openvoice(
                    text=clipped,
                    language=language,
                    voice_preset=resolved_preset,
                    persona_style=persona_style,
                    voice_mode=openvoice_mode,
                    reference_audio_bytes=reference_audio_bytes,
                    builtin_voice_id=normalized_builtin_voice_id or None,
                    runtime_device=desired_tts_device,
                )
            except Exception:
                if requires_clone_voice:
                    raise RuntimeError(
                        "Selected clone voice is unavailable right now. OpenVoice cloning failed; local fallback is disabled for clone voices."
                    )
                allow_fallback = str(
                    os.environ.get(
                        "COACH_TTS_ALLOW_ESPEAK_FALLBACK",
                        str(settings.COACH_TTS_ALLOW_ESPEAK_FALLBACK),
                    )
                ).strip().lower() in {"1", "true", "yes", "on"}
                if not allow_fallback:
                    raise

        tts_binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not tts_binary:
            raise RuntimeError("Local TTS engine is unavailable. Install espeak-ng in the app container.")

        voice_name, rate, pitch = self._resolve_tts_voice(language=language, voice_preset=resolved_preset)
        audio_bytes = await asyncio.to_thread(
            self._synthesize_with_espeak,
            binary=tts_binary,
            text=clipped,
            voice_name=voice_name,
            rate=rate,
            pitch=pitch,
        )
        return audio_bytes, "audio/wav"

    async def list_turns(self, session_id: str, user_id: int) -> List[CoachTurn]:
        session_id = str(session_id)
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachTurn)
                .join(CoachSession)
                .where(CoachSession.id == session_id, CoachSession.user_id == user_id)
                .options(selectinload(CoachTurn.mistakes))
                .order_by(CoachTurn.turn_index.asc(), CoachTurn.created_at.asc())
            )
            return result.scalars().all() # type: ignore

    async def list_mistakes(self, session_id: str, user_id: int) -> List[CoachMistake]:
        session_id = str(session_id)
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachMistake)
                .where(
                    CoachMistake.session_id == session_id,
                    CoachMistake.user_id == user_id,
                )
                .order_by(CoachMistake.created_at.asc(), CoachMistake.id.asc())
            )
            return result.scalars().all() # type: ignore

    async def progress(self, user_id: int, session_id: str) -> dict:
        turns = await self.list_turns(session_id=session_id, user_id=user_id)
        mistakes = await self.list_mistakes(session_id=session_id, user_id=user_id)
        scores = [turn.score for turn in turns if isinstance(turn.score, int)]
        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        return {
            "session_id": str(session_id),
            "turn_count": len(turns),
            "mistake_count": len(mistakes),
            "average_score": avg_score,
            "latest_score": scores[-1] if scores else None,
        }

    async def end_session(self, user_id: int, session_id: str) -> Optional[dict]:
        session_id = str(session_id)
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CoachSession)
                .where(CoachSession.id == session_id, CoachSession.user_id == user_id)
                .options(selectinload(CoachSession.turns).selectinload(CoachTurn.mistakes))
            )
            coach_session = result.scalar_one_or_none()
            if coach_session is None:
                return None

            turns = list(coach_session.turns or [])
            mistakes = [mistake for turn in turns for mistake in turn.mistakes]
            scores = [turn.score for turn in turns if isinstance(turn.score, int)]
            scored_turn_count = len(scores)
            average_score = round(sum(scores) / len(scores), 2) if scores else None
            latest_score = scores[-1] if scores else None

            category_counts = Counter(
                str(mistake.category or "general").strip() or "general"
                for mistake in mistakes
            )
            top_categories = [
                category
                for category, _ in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)[:3]
            ]
            recent_feedback = [
                str(turn.explanation).strip()
                for turn in turns[-4:]
                if isinstance(turn.explanation, str) and turn.explanation.strip()
            ]

            summary_payload = self._fallback_end_session_summary(
                subject=coach_session.focus_area or coach_session.title,
                average_score=average_score,
                turn_count=len(turns),
                scored_turn_count=scored_turn_count,
                top_categories=top_categories,
                recent_feedback=recent_feedback,
            )

            coach_session.status = "completed"
            await session.commit()

        return {
            "session_id": session_id,
            "status": "completed",
            "subject": coach_session.focus_area or coach_session.title,
            "turn_count": len(turns),
            "scored_turn_count": scored_turn_count,
            "average_score": average_score,
            "latest_score": latest_score,
            "mistake_counts_by_category": dict(category_counts),
            **summary_payload,
        }

    def _fallback_end_session_summary(
        self,
        *,
        subject: Optional[str],
        average_score: Optional[float],
        turn_count: int,
        scored_turn_count: int,
        top_categories: Sequence[str],
        recent_feedback: Sequence[str],
    ) -> dict:
        topic = str(subject or "the selected topic").strip() or "the selected topic"
        score_label = "N/A" if average_score is None else f"{average_score:.1f}"
        if average_score is None:
            performance = "You completed the conversation. Keep speaking in full sentences for stronger scoring."
        elif average_score >= 85:
            performance = "Strong performance with clear answers and good flow."
        elif average_score >= 70:
            performance = "Solid base. Keep improving precision and detail."
        else:
            performance = "Good effort. Focus on clearer grammar and fuller answers."

        focus_text = (
            f"Main improvement area: {', '.join(top_categories)}."
            if top_categories
            else "No major recurring mistake category detected."
        )
        if scored_turn_count > 0:
            score_scope = f"Average score: {score_label} across {scored_turn_count} scored turns out of {turn_count} total turns."
        else:
            score_scope = f"No scored turns yet across {turn_count} total turns."
        feedback_summary = (
            f"Conversation complete on '{topic}'. "
            f"{score_scope} "
            f"{performance} {focus_text}"
        )
        strengths = [
            "You stayed engaged through multiple turns.",
            "You answered topic-focused prompts.",
        ]
        improvement_points = [
            f"Work on {top_categories[0]} in your next session." if top_categories else "Use more specific examples in your answers.",
            "Keep responses concise but complete.",
        ]
        if recent_feedback:
            improvement_points[1] = recent_feedback[-1]

        return {
            "feedback_summary": feedback_summary,
            "strengths": strengths,
            "improvement_points": improvement_points,
        }

    async def progress_summary(self, user_id: int) -> dict:
        async with AsyncSessionLocal() as session:
            total_sessions_result = await session.execute(
                select(func.count(CoachSession.id)).where(CoachSession.user_id == user_id)
            )
            total_turns_result = await session.execute(
                select(func.count(CoachTurn.id))
                .join(CoachSession, CoachTurn.session_id == CoachSession.id)
                .where(CoachSession.user_id == user_id)
            )
            total_mistakes_result = await session.execute(
                select(func.count(CoachMistake.id))
                .join(CoachSession, CoachMistake.session_id == CoachSession.id)
                .where(CoachSession.user_id == user_id)
            )
            active_sessions_result = await session.execute(
                select(func.count(CoachSession.id)).where(
                    CoachSession.user_id == user_id,
                    CoachSession.status == "active",
                )
            )
            category_rows = await session.execute(
                select(CoachMistake.category, func.count(CoachMistake.id))
                .join(CoachSession, CoachMistake.session_id == CoachSession.id)
                .where(CoachSession.user_id == user_id)
                .group_by(CoachMistake.category)
            )
            model_rows = await session.execute(
                select(CoachTurn.model_id, func.count(CoachTurn.id))
                .join(CoachSession, CoachTurn.session_id == CoachSession.id)
                .where(CoachSession.user_id == user_id)
                .group_by(CoachTurn.model_id)
            )
            latest_session_result = await session.execute(
                select(CoachSession)
                .where(CoachSession.user_id == user_id)
                .options(selectinload(CoachSession.turns).selectinload(CoachTurn.mistakes))
                .order_by(desc(CoachSession.created_at), desc(CoachSession.id))
            )
            latest_session = latest_session_result.scalars().first()

        mistake_counts_by_category = {
            str(category): int(count)
            for category, count in category_rows.all()
            if category is not None
        }
        turn_counts_by_model = {
            str(model_id) if model_id is not None else "unassigned": int(count)
            for model_id, count in model_rows.all()
        }

        latest_session_id = None
        latest_session_title = None
        latest_session_turns = 0
        if latest_session is not None:
            latest_session_id = latest_session.id
            latest_session_title = latest_session.title
            latest_session_turns = len(latest_session.turns)

        summary = {
            "user_id": user_id,
            "total_sessions": int(total_sessions_result.scalar_one() or 0),
            "total_turns": int(total_turns_result.scalar_one() or 0),
            "total_mistakes": int(total_mistakes_result.scalar_one() or 0),
            "mistake_counts_by_category": mistake_counts_by_category,
            "turn_counts_by_model": turn_counts_by_model,
            "active_sessions": int(active_sessions_result.scalar_one() or 0),
            "latest_session_id": latest_session_id,
            "latest_session_title": latest_session_title,
            "latest_session_turns": latest_session_turns,
        }
        logger.info(
            "Coach progress summary prepared for user_id=%s sessions=%s turns=%s mistakes=%s",
            user_id,
            summary["total_sessions"],
            summary["total_turns"],
            summary["total_mistakes"],
        )
        return summary

    async def _available_ollama_models(self) -> List[str]:
        timeout = httpx.Timeout(
            connect=self.OLLAMA_CONNECT_TIMEOUT_SEC,
            read=10.0,
            write=self.OLLAMA_WRITE_TIMEOUT_SEC,
            pool=self.OLLAMA_POOL_TIMEOUT_SEC,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return []
            payload = response.json()
            models = payload.get("models", [])
            names = [str(item.get("name", "")).strip() for item in models if item.get("name")]
            return [name for name in names if name]
        except Exception:
            return []

    def _whisper_runtime_options(self) -> tuple[str, str, bool, str]:
        profile = self._coach_hardware_profile()
        runtime_device = self._resolve_runtime_device(
            env_key="COACH_WHISPER_DEVICE",
            configured_default=settings.COACH_WHISPER_DEVICE,
            profile_default=str(profile.get("asr_device") or "cpu"),
        )

        compute_type = self._env_text("COACH_WHISPER_COMPUTE_TYPE", settings.COACH_WHISPER_COMPUTE_TYPE)
        if not compute_type:
            compute_type = str(profile.get("asr_compute_type") or ("float16" if runtime_device == "cuda" else "int8"))

        local_files_only = self._env_bool("COACH_WHISPER_LOCAL_FILES_ONLY", bool(settings.COACH_WHISPER_LOCAL_FILES_ONLY))
        download_root = self._env_text("COACH_WHISPER_DOWNLOAD_ROOT", settings.COACH_WHISPER_DOWNLOAD_ROOT)
        return runtime_device, compute_type, local_files_only, download_root

    async def _load_whisper_model(self, *, model_ref: str):
        from faster_whisper import WhisperModel

        runtime_device, compute_type, local_files_only, download_root = self._whisper_runtime_options()
        whisper_kwargs: Dict[str, Any] = {
            "device": runtime_device,
            "compute_type": compute_type,
            "local_files_only": local_files_only,
        }
        if download_root:
            whisper_kwargs["download_root"] = download_root

        try:
            model = WhisperModel(model_ref, **whisper_kwargs)
        except TypeError:
            # Compatibility fallback for older faster-whisper versions.
            whisper_kwargs.pop("local_files_only", None)
            model = WhisperModel(model_ref, **whisper_kwargs)

        logger.info(
            "Loaded faster-whisper model for coach transcription: model_ref=%s device=%s compute_type=%s local_files_only=%s download_root=%s",
            model_ref,
            runtime_device,
            compute_type,
            local_files_only,
            download_root or "<default>",
        )
        return model, runtime_device

    def _fast_whisper_model_ref(self) -> str:
        profile = self._coach_hardware_profile()
        fast_name = self._env_text("COACH_WHISPER_FAST_MODEL", settings.COACH_WHISPER_FAST_MODEL)
        legacy_name = self._env_text("COACH_WHISPER_MODEL", settings.COACH_WHISPER_MODEL)
        model_path = self._env_text("COACH_WHISPER_MODEL_PATH", settings.COACH_WHISPER_MODEL_PATH)
        if model_path:
            return model_path
        if fast_name and fast_name.lower() != "auto":
            return fast_name
        if legacy_name and legacy_name.lower() != "auto":
            return legacy_name
        return str(profile.get("asr_fast_model") or "medium")

    def _accurate_whisper_model_ref(self) -> str:
        profile = self._coach_hardware_profile()
        accurate_name = self._env_text("COACH_WHISPER_ACCURATE_MODEL", settings.COACH_WHISPER_ACCURATE_MODEL)
        if accurate_name and accurate_name.lower() != "auto":
            return accurate_name
        return str(profile.get("asr_accurate_model") or self._fast_whisper_model_ref())

    def _normalize_asr_preference(self, preferred_asr_model: Optional[str]) -> str:
        normalized = str(preferred_asr_model or "").strip().lower()
        if normalized in {"accurate", "quality", "high", "high_accuracy"}:
            return "accurate"
        if normalized in {"fast", "speed", "low_latency"}:
            return "fast"
        return "auto"

    async def _get_whisper_model(self):
        if self._whisper_model is not None:
            return self._whisper_model
        async with self._whisper_lock:
            if self._whisper_model is not None:
                return self._whisper_model
            try:
                model, runtime_device = await self._load_whisper_model(model_ref=self._fast_whisper_model_ref())
                self._whisper_model = model
                self._whisper_runtime_device = runtime_device
            except Exception as exc:
                logger.warning("Could not load faster-whisper model: %s", exc)
                self._whisper_model = None
                self._whisper_runtime_device = None
            return self._whisper_model

    async def _get_whisper_accurate_model(self):
        if self._whisper_accurate_model is not None:
            return self._whisper_accurate_model
        async with self._whisper_lock:
            if self._whisper_accurate_model is not None:
                return self._whisper_accurate_model
            try:
                model_ref = self._accurate_whisper_model_ref()
                if model_ref == self._fast_whisper_model_ref() and self._whisper_model is not None:
                    self._whisper_accurate_model = self._whisper_model
                    self._whisper_accurate_runtime_device = self._whisper_runtime_device
                else:
                    model, runtime_device = await self._load_whisper_model(model_ref=model_ref)
                    self._whisper_accurate_model = model
                    self._whisper_accurate_runtime_device = runtime_device
            except Exception as exc:
                logger.warning("Could not load accurate faster-whisper model: %s", exc)
                self._whisper_accurate_model = None
                self._whisper_accurate_runtime_device = None
            return self._whisper_accurate_model

    def _asr_confidence_band(
        self,
        *,
        avg_logprob: Optional[float],
        no_speech_prob: Optional[float],
        text: str,
    ) -> tuple[float, str]:
        score = 0.75
        if avg_logprob is not None:
            if avg_logprob < -1.3:
                score -= 0.45
            elif avg_logprob < -0.9:
                score -= 0.25
            elif avg_logprob < -0.6:
                score -= 0.1
        if no_speech_prob is not None:
            if no_speech_prob > 0.8:
                score -= 0.5
            elif no_speech_prob > 0.6:
                score -= 0.25
        if not self._is_transcript_evaluable(text):
            score -= 0.3
        score = max(0.0, min(1.0, score))
        if score >= 0.7:
            return score, "high"
        if score >= 0.45:
            return score, "medium"
        return score, "low"

    def _extract_segments_stats(self, segments: Sequence[Any], text: str) -> dict[str, Any]:
        avg_logprobs: list[float] = []
        no_speech_probs: list[float] = []
        for segment in segments:
            avg_logprob = getattr(segment, "avg_logprob", None)
            if isinstance(avg_logprob, (float, int)):
                avg_logprobs.append(float(avg_logprob))
            no_speech_prob = getattr(segment, "no_speech_prob", None)
            if isinstance(no_speech_prob, (float, int)):
                no_speech_probs.append(float(no_speech_prob))
        avg_logprob_value = (sum(avg_logprobs) / len(avg_logprobs)) if avg_logprobs else None
        no_speech_value = max(no_speech_probs) if no_speech_probs else None
        confidence_value, confidence_band = self._asr_confidence_band(
            avg_logprob=avg_logprob_value,
            no_speech_prob=no_speech_value,
            text=text,
        )
        return {
            "avg_logprob": avg_logprob_value,
            "no_speech_prob": no_speech_value,
            "confidence": confidence_value,
            "confidence_band": confidence_band,
        }

    async def _transcribe_with_model(
        self,
        *,
        whisper_model: Any,
        path: str,
        language: Optional[str],
    ) -> tuple[str, dict[str, Any]]:
        segments, _ = whisper_model.transcribe(path, language=language or None)
        normalized_segments = [segment for segment in segments]
        text = " ".join(
            segment.text.strip() for segment in normalized_segments if getattr(segment, "text", "").strip()
        )
        text = re.sub(r"\s+", " ", text).strip()
        stats = self._extract_segments_stats(normalized_segments, text=text)
        return text, stats

    async def _transcribe_audio_adaptive(
        self,
        audio_bytes: bytes,
        *,
        filename: Optional[str] = None,
        transcript_hint: Optional[str] = None,
        language: Optional[str] = None,
        preferred_asr_model: Optional[str] = None,
    ) -> dict[str, Any]:
        if transcript_hint and transcript_hint.strip():
            hinted = re.sub(r"\s+", " ", transcript_hint).strip()
            return {
                "text": hinted,
                "model": "hint",
                "retry_used": False,
                "selection": "hint",
                "confidence": 1.0,
                "confidence_band": "high",
                "avg_logprob": None,
                "no_speech_prob": None,
            }
        asr_selection = self._normalize_asr_preference(preferred_asr_model)
        fast_model_ref = self._fast_whisper_model_ref()
        accurate_model_ref = self._accurate_whisper_model_ref()
        if asr_selection == "accurate":
            whisper_model = await self._get_whisper_accurate_model()
            selected_model_ref = accurate_model_ref
        else:
            whisper_model = await self._get_whisper_model()
            selected_model_ref = fast_model_ref
        suffix = ".webm"
        if filename and "." in filename:
            suffix = f".{filename.rsplit('.', 1)[-1]}"
        if whisper_model is None:
            return {
                "text": f"captured {len(audio_bytes)} bytes of audio",
                "model": "none",
                "retry_used": False,
                "selection": asr_selection,
                "confidence": 0.0,
                "confidence_band": "low",
                "avg_logprob": None,
                "no_speech_prob": None,
            }

        profile = self._coach_hardware_profile()
        retry_default = bool(profile.get("asr_retry_enabled", True))
        retry_raw = self._env_text("COACH_WHISPER_ENABLE_ACCURATE_RETRY", str(settings.COACH_WHISPER_ENABLE_ACCURATE_RETRY))
        if retry_raw.lower() == "auto":
            retry_enabled = retry_default
        else:
            retry_enabled = retry_raw.lower() in {"1", "true", "yes", "on"}
        if asr_selection != "auto":
            retry_enabled = False
        threshold_default = float(profile.get("asr_retry_threshold", settings.COACH_WHISPER_RETRY_CONFIDENCE_THRESHOLD))
        threshold_raw = self._env_text("COACH_WHISPER_RETRY_CONFIDENCE_THRESHOLD", str(settings.COACH_WHISPER_RETRY_CONFIDENCE_THRESHOLD))
        try:
            retry_threshold = max(0.0, min(1.0, float(threshold_raw)))
        except (TypeError, ValueError):
            retry_threshold = threshold_default

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            text, stats = await self._transcribe_with_model(
                whisper_model=whisper_model,
                path=temp_path,
                language=language,
            )
            result = {
                "text": text or f"captured {len(audio_bytes)} bytes of audio",
                "model": selected_model_ref,
                "retry_used": False,
                "selection": asr_selection,
                **stats,
            }

            if retry_enabled and result["confidence"] < retry_threshold:
                accurate_model = await self._get_whisper_accurate_model()
                if accurate_model is not None:
                    retry_text, retry_stats = await self._transcribe_with_model(
                        whisper_model=accurate_model,
                        path=temp_path,
                        language=language,
                    )
                    retry_result = {
                        "text": retry_text or result["text"],
                        "model": accurate_model_ref,
                        "retry_used": True,
                        "selection": "auto",
                        **retry_stats,
                    }
                    if retry_result["confidence"] >= result["confidence"]:
                        result = retry_result
            return result
        except Exception as exc:
            error_text = str(exc).lower()
            cuda_runtime_error = any(marker in error_text for marker in ("libcublas", "cuda", "cudnn", "cublas"))
            if cuda_runtime_error and self._whisper_runtime_device == "cuda":
                logger.warning("Coach transcription CUDA runtime issue detected (%s). Retrying on CPU.", exc)
                self._whisper_model = None
                self._whisper_runtime_device = None
                os.environ["COACH_WHISPER_DEVICE"] = "cpu"
                cpu_model = await self._get_whisper_model()
                if cpu_model is not None and temp_path:
                    try:
                        text, stats = await self._transcribe_with_model(
                            whisper_model=cpu_model,
                            path=temp_path,
                            language=language,
                        )
                        return {
                            "text": text or f"captured {len(audio_bytes)} bytes of audio",
                            "model": selected_model_ref,
                            "retry_used": False,
                            "selection": asr_selection,
                            **stats,
                        }
                    except Exception as retry_exc:
                        logger.warning("Coach transcription CPU retry failed: %s", retry_exc)
            logger.warning("Coach transcription failed, using fallback transcript: %s", exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        return {
            "text": f"captured {len(audio_bytes)} bytes of audio",
            "model": "fallback-bytes",
            "retry_used": False,
            "selection": asr_selection,
            "confidence": 0.0,
            "confidence_band": "low",
            "avg_logprob": None,
            "no_speech_prob": None,
        }

    async def _transcribe_audio(
        self,
        audio_bytes: bytes,
        *,
        filename: Optional[str] = None,
        transcript_hint: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        payload = await self._transcribe_audio_adaptive(
            audio_bytes=audio_bytes,
            filename=filename,
            transcript_hint=transcript_hint,
            language=language,
        )
        return str(payload.get("text") or "").strip() or f"captured {len(audio_bytes)} bytes of audio"

    def _fallback_coaching(
        self,
        transcript: str,
        *,
        target_language: Optional[str] = None,
        native_language: Optional[str] = None,
        learner_level: Optional[str] = None,
        focus_area: Optional[str] = None,
        persona_style: Optional[str] = None,
    ) -> dict:
        word_count = len(transcript.split())
        normalized_level = self._normalize_learner_level(learner_level)
        mistakes = []
        if word_count < 5:
            mistakes.append(
                {
                    "category": "fluency",
                    "detail": "Response is too short for practice depth.",
                    "severity": "medium",
                    "suggestion": "Add one more sentence with a concrete detail.",
                }
            )
        if not mistakes:
            mistakes.append(
                {
                    "category": "general",
                    "detail": "No major issues detected in this turn.",
                    "severity": "low",
                    "suggestion": "Keep speaking with varied sentence structure.",
                }
            )
        if word_count >= 14:
            score = 78
        elif word_count >= 8:
            score = 70
        else:
            score = 55
        correction = mistakes[0]["suggestion"] if mistakes and mistakes[0]["category"] != "general" else None
        explanation = "Target: keep your answer precise and practical."
        native_name = self._resolve_supported_language_name(native_language) or ""
        if native_name and native_name.lower() != str(target_language or "").strip().lower():
            explanation = f"{explanation}\n{native_name}: keep expanding with concrete examples."
        topic = str(focus_area or "").strip()
        if topic:
            follow_up_question = f"Can you give one concrete example about {topic}?"
        else:
            follow_up_question = "Can you answer again with one concrete example?"
        persona_hint = str(persona_style or "").strip()
        persona_prefix = "Acknowledged. " if persona_hint else ""
        if normalized_level in {"ignorant", "novice"} and native_name:
            follow_up_question = (
                f"You can answer in {native_name} if needed. What is one basic sentence you want to say?"
            )
        return {
            "reply": f"{persona_prefix}Great effort. Let's refine this response: {transcript}",
            "follow_up_question": follow_up_question,
            "score": score,
            "correction": correction,
            "explanation": explanation,
            "response_translation": "",
            "mistakes": mistakes,
        }

    def _has_actionable_mistake(self, mistakes: Sequence[Any]) -> bool:
        for item in mistakes:
            data = self._coerce_mapping(item)
            category = str(data.get("category") or "").strip().lower()
            detail = str(data.get("detail") or data.get("message") or "").strip()
            if detail and category not in {"general", "audio"}:
                return True
        return False

    def _prepend_explicit_recast(
        self,
        *,
        reply: str,
        correction: Optional[str],
        mistakes: Sequence[Any],
        confidence_band: str = "high",
    ) -> str:
        normalized_reply = re.sub(r"\s+", " ", str(reply or "")).strip()
        normalized_correction = re.sub(r"\s+", " ", str(correction or "")).strip()
        if not normalized_correction:
            return normalized_reply
        correction_lower = normalized_correction.lower()
        if correction_lower.startswith(("use ", "add ", "remove ", "say ", "try ", "avoid ", "replace ", "change ")):
            return normalized_reply
        if str(confidence_band or "high").strip().lower() == "low":
            return normalized_reply
        if not self._has_actionable_mistake(mistakes):
            return normalized_reply
        if normalized_correction.lower() in normalized_reply.lower():
            return normalized_reply
        recast_prefix = f"You mean: {normalized_correction}"
        if recast_prefix[-1:] not in {".", "!", "?"}:
            recast_prefix = f"{recast_prefix}."
        if not normalized_reply:
            return recast_prefix
        return f"{recast_prefix} {normalized_reply}".strip()

    def _extract_json_object(self, text: str) -> Optional[dict]:
        text = text.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    async def _coach_fast_with_model(
        self,
        *,
        model: str,
        transcript: str,
        target_language: str,
        native_language: Optional[str] = None,
        learner_level: str = "medium",
        focus_area: Optional[str] = None,
        recent_turns: Optional[Sequence[CoachTurn]] = None,
        persona_style: Optional[str] = None,
        llm_device_preference: Optional[str] = None,
        asr_confidence_band: str = "high",
    ) -> tuple[dict[str, Any], int]:
        subject = str(focus_area or "").strip() or "General conversation"
        persona = str(persona_style or "").strip() or "Default coach"
        confidence_band = str(asr_confidence_band or "high").strip().lower()
        normalized_level = self._normalize_learner_level(learner_level)
        level_instructions = self._coaching_level_instructions(normalized_level)
        recent_context: List[str] = []
        for turn in list(recent_turns or [])[-2:]:
            learner = str(turn.transcript or "").strip()
            coach_reply = str(turn.reply or "").strip()
            if learner:
                recent_context.append(f"Learner: {learner}")
            if coach_reply:
                recent_context.append(f"Coach: {coach_reply}")
        context_block = "\n".join(recent_context) if recent_context else "No prior turns."
        needs_translation = self._should_attach_translation(
            target_language=target_language,
            native_language=native_language,
        )
        system_prompt = (
            "You are an empathetic language coach in live voice mode. Return JSON only.\n"
            "Output fields:\n"
            "1) reply: one short response sentence.\n"
            "2) follow_up_question: one short question that keeps the learner speaking.\n"
            "3) quick_recast: optional short correction/recast.\n"
            "4) needs_confirmation: boolean.\n"
            "5) confirmation_text: optional question when transcription confidence is low.\n"
            "6) response_translation: optional native-language translation of the final coach message.\n"
            "Rules:\n"
            "- If there is an obvious grammar mistake and confidence is not low, include a direct recast correction.\n"
            "- If confidence is low, avoid strict correction and ask confirmation first.\n"
            "- If the learner asks a direct question, answer it first before continuing the drill.\n"
            "- Do not claim you did not understand unless confidence is low.\n"
            "- Keep tone positive and calming; if learner is confused, reduce complexity.\n"
            "- For beginner learners, allow native-language replies and gently bridge to target language.\n"
            "- Do not repeat the same coach sentence from the recent context.\n"
            "- Do not invent story context not present in transcript or selected subject.\n"
            "- If Persona instructions define a character identity, stay in that identity and never rename yourself.\n"
            "- Be strict on language accuracy without judging intent. Never accuse the learner of avoidance, laziness, or bad faith.\n"
            "- Keep messages concise and natural."
        )
        user_prompt = (
            f"Target language: {target_language}\n"
            f"Native language: {native_language or 'Unknown'}\n"
            f"Learner level: {normalized_level}\n"
            f"Subject: {subject}\n"
            f"Persona instructions: {persona}\n"
            f"Level instructions: {level_instructions}\n"
            f"Need native translation in response_translation: {str(needs_translation).lower()}\n"
            f"ASR confidence band: {confidence_band}\n"
            f"Recent turns:\n{context_block}\n"
            f"Learner transcript: {transcript}\n"
            "Return strictly valid JSON."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_ctx": settings.OLLAMA_NUM_CTX,
                "num_predict": 90,
            },
        }
        num_gpu = self._ollama_num_gpu_option(llm_device_preference)
        if num_gpu is not None:
            payload["options"]["num_gpu"] = num_gpu
        timeout = httpx.Timeout(
            connect=self.OLLAMA_CONNECT_TIMEOUT_SEC,
            read=max(10.0, float(settings.OLLAMA_CHAT_TIMEOUT_SEC)),
            write=self.OLLAMA_WRITE_TIMEOUT_SEC,
            pool=self.OLLAMA_POOL_TIMEOUT_SEC,
        )
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        data = response.json()
        content = (
            data.get("message", {}).get("content")
            if isinstance(data.get("message"), dict)
            else data.get("response")
        )
        parsed = self._extract_json_object(str(content or ""))
        if not parsed:
            raise ValueError("Coach fast response was not valid JSON")
        reply = str(parsed.get("reply") or "").strip()
        follow_up_question = str(parsed.get("follow_up_question") or "").strip()
        quick_recast = str(parsed.get("quick_recast") or "").strip()
        needs_confirmation = bool(parsed.get("needs_confirmation"))
        confirmation_text = str(parsed.get("confirmation_text") or "").strip()
        response_translation = str(parsed.get("response_translation") or "").strip()
        if not reply:
            reply = "Good start. Keep going."
        return {
            "reply": reply,
            "follow_up_question": follow_up_question,
            "quick_recast": quick_recast,
            "needs_confirmation": needs_confirmation,
            "confirmation_text": confirmation_text,
            "response_translation": response_translation,
        }, elapsed_ms

    async def _coach_with_model(
        self,
        *,
        model: str,
        transcript: str,
        target_language: str,
        native_language: Optional[str],
        cefr_level: str,
        focus_area: Optional[str] = None,
        recent_turns: Optional[Sequence[CoachTurn]] = None,
        persona_style: Optional[str] = None,
        llm_device_preference: Optional[str] = None,
    ) -> tuple[dict, int]:
        subject = str(focus_area or "").strip()
        persona = str(persona_style or "").strip()
        normalized_level = self._normalize_learner_level(cefr_level)
        level_instructions = self._coaching_level_instructions(normalized_level)
        needs_translation = self._should_attach_translation(
            target_language=target_language,
            native_language=native_language,
        )
        recent_context: List[str] = []
        for turn in list(recent_turns or [])[-3:]:
            learner = str(turn.transcript or "").strip()
            coach_reply = str(turn.reply or "").strip()
            if learner:
                recent_context.append(f"Learner: {learner}")
            if coach_reply:
                recent_context.append(f"Coach: {coach_reply}")
        context_block = "\n".join(recent_context) if recent_context else "No prior turns."

        system_prompt = (
            "You are an adaptive language coach running a live conversation drill. Return JSON only.\n"
            "Requirements:\n"
            "1) Always include fields: reply (string), follow_up_question (string), score (0-100 integer), mistakes (array).\n"
            "2) follow_up_question must be one short question that keeps the learner talking.\n"
            "3) The follow_up_question must stay on the selected subject when provided.\n"
            "4) Avoid repeating the exact previous coach question.\n"
            "5) Include correction and explanation only if needed.\n"
            "6) explanation must be bilingual: target language first, native language second.\n"
            "7) mistakes must include category, detail, severity, suggestion.\n"
            "8) Keep a consistent speaking style based on Persona instructions when provided.\n"
            "9) When learner grammar is clearly wrong, begin reply with a direct recast correction in target language.\n"
            "10) If Persona instructions define a character identity, stay in that identity and never rename yourself.\n"
            "11) Do not invent facts or scenario details that the learner did not say.\n"
            "12) Keep the learner comfortable and positive; do not shame or mock.\n"
            "13) If learner is confused, simplify and steer gently.\n"
            "14) Add field response_translation (string). If native language is supported and different from target, translate your full coach message (reply + follow_up_question) into native language.\n"
            "15) If learner asks a direct question, answer it directly first, then ask one follow-up.\n"
            "16) Do not repeatedly say you did not understand unless the learner input is empty.\n"
            "17) Stay respectful and non-judgmental. Be critical about language output, never about the learner's intent or effort.\n"
        )
        user_prompt = (
            f"Target language: {target_language}\n"
            f"Native language: {native_language or 'Unknown'}\n"
            f"Learner level: {normalized_level}\n"
            f"Level instructions: {level_instructions}\n"
            f"Subject: {subject or 'General conversation'}\n"
            f"Persona instructions: {persona or 'Default coach'}\n"
            f"Need native translation in response_translation: {str(needs_translation).lower()}\n"
            f"Recent turns:\n{context_block}\n"
            f"Transcript: {transcript}\n"
            "Return strictly valid JSON."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_ctx": settings.OLLAMA_NUM_CTX,
            },
        }
        num_gpu = self._ollama_num_gpu_option(llm_device_preference)
        if num_gpu is not None:
            payload["options"]["num_gpu"] = num_gpu

        timeout = httpx.Timeout(
            connect=self.OLLAMA_CONNECT_TIMEOUT_SEC,
            read=max(10.0, float(settings.OLLAMA_CHAT_TIMEOUT_SEC)),
            write=self.OLLAMA_WRITE_TIMEOUT_SEC,
            pool=self.OLLAMA_POOL_TIMEOUT_SEC,
        )
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        data = response.json()
        content = (
            data.get("message", {}).get("content")
            if isinstance(data.get("message"), dict)
            else data.get("response")
        )
        parsed = self._extract_json_object(str(content or ""))
        if not parsed:
            raise ValueError("Coach model response was not valid JSON")

        reply = str(parsed.get("reply") or "").strip()
        score_raw = parsed.get("score")
        try:
            score = int(score_raw) # type: ignore
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))
        mistakes = parsed.get("mistakes")
        if not isinstance(mistakes, list):
            mistakes = []

        result = {
            "reply": reply or f"Let's improve this: {transcript}",
            "follow_up_question": str(parsed.get("follow_up_question") or "").strip(),
            "score": score,
            "correction": parsed.get("correction"),
            "explanation": parsed.get("explanation"),
            "response_translation": str(parsed.get("response_translation") or "").strip(),
            "mistakes": mistakes,
        }
        return result, elapsed_ms

    async def stream_turn(
        self,
        *,
        user_id: int,
        session_id: str,
        audio,
        transcript_hint: Optional[str] = None,
        preferred_model: Optional[str] = None,
        preferred_asr_model: Optional[str] = None,
        llm_device_preference: Optional[str] = None,
        persona_style: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        session = await self.get_session(session_id=str(session_id), user_id=user_id)
        if session is None:
            raise ValueError("Session not found")
        effective_llm_device_preference = self._effective_device_preference(
            explicit_preference=llm_device_preference,
            persisted_preference=getattr(session, "llm_device_preference", "auto"),
        )

        audio_bytes = await audio.read(self.COACH_MAX_AUDIO_BYTES + 1)
        if not audio_bytes:
            raise ValueError("Uploaded audio file is empty")
        if len(audio_bytes) > self.COACH_MAX_AUDIO_BYTES:
            raise ValueError("Audio file is too large")
        await audio.seek(0)

        transcription_payload = await self._transcribe_audio_adaptive(
            audio_bytes=audio_bytes,
            filename=getattr(audio, "filename", None),
            transcript_hint=transcript_hint,
            language=self._target_language_code(session.target_language),
            preferred_asr_model=preferred_asr_model,
        )
        transcript = str(transcription_payload.get("text") or "").strip()
        asr_confidence_band = str(transcription_payload.get("confidence_band") or "low").strip().lower()
        turn_id = str(uuid.uuid4())

        partial_words: List[str] = []
        for token in transcript.split():
            partial_words.append(token)
            yield {
                "type": "stt_partial",
                "turn_id": turn_id,
                "text": " ".join(partial_words),
            }

        yield {
            "type": "stt_final",
            "turn_id": turn_id,
            "text": transcript,
            "asr_model": str(transcription_payload.get("model") or ""),
            "asr_selection": str(transcription_payload.get("selection") or "auto"),
            "asr_retry_used": bool(transcription_payload.get("retry_used")),
            "asr_confidence_band": asr_confidence_band,
            "asr_confidence": transcription_payload.get("confidence"),
        }

        if not self._is_transcript_evaluable(transcript):
            retry_reply = (
                "I couldn't catch a clear spoken answer. Click the mic and respond again with one full sentence."
            )
            saved_turn = await self.save_turn_with_mistakes(
                session_id=str(session_id),
                user_id=user_id,
                transcript=transcript,
                reply=retry_reply,
                score=None,
                explanation="No clear speech detected. This turn was not scored.",
                model_id="asr-no-speech",
                mistakes=[
                    {
                        "category": "audio",
                        "detail": "No clear speech detected in this recording.",
                        "severity": "low",
                        "suggestion": "Speak closer to the microphone and answer in a full sentence.",
                    }
                ],
            )
            if saved_turn is None:
                raise ValueError("Failed to persist coach turn")

            yield {
                "type": "coach_reply",
                "turn_id": turn_id,
                "text": saved_turn.reply,
                "question": "",
            }
            yield {
                "type": "feedback",
                "turn_id": turn_id,
                "summary": saved_turn.explanation or "No clear speech detected. This turn was not scored.",
                "mistakes": [
                    {
                        "category": mistake.category,
                        "detail": mistake.detail,
                        "severity": mistake.severity,
                        "suggestion": mistake.suggestion,
                    }
                    for mistake in saved_turn.mistakes
                ],
            }
            yield {
                "type": "score",
                "turn_id": turn_id,
                "value": None,
                "model_id": "asr-no-speech",
            }
            return

        available_models = await self._available_ollama_models()
        candidates = self.get_quality_first_model_order()
        if available_models:
            allowed = set(available_models)
            candidates = [candidate for candidate in candidates if candidate in allowed] or candidates

        requested_model = str(preferred_model or "").strip()
        respect_user_model_choice = False
        if requested_model:
            if requested_model in candidates:
                respect_user_model_choice = True
                candidates = [requested_model] + [candidate for candidate in candidates if candidate != requested_model]
            else:
                yield {
                    "type": "model_fallback",
                    "turn_id": turn_id,
                    "requested_model": requested_model,
                    "selected_model": candidates[0],
                    "reason": "requested_model_unavailable",
                }

        selected_model = candidates[0]
        fast_coaching_result: Optional[dict[str, Any]] = None
        full_coaching_result: Optional[dict[str, Any]] = None
        fast_latency_ms: Optional[int] = None
        full_latency_ms: Optional[int] = None
        last_coach_reply = ""
        if session.turns:
            last_coach_reply = str(session.turns[-1].reply or "").strip()
        two_pass_enabled = bool(
            str(
                os.environ.get("COACH_ENABLE_TWO_PASS_VOICE", str(settings.COACH_ENABLE_TWO_PASS_VOICE))
                or str(settings.COACH_ENABLE_TWO_PASS_VOICE)
            ).strip().lower() in {"1", "true", "yes", "on"}
        )
        for idx, model in enumerate(candidates):
            try:
                if two_pass_enabled and fast_coaching_result is None:
                    fast_result, observed_fast_latency = await self._coach_fast_with_model(
                        model=model,
                        transcript=transcript,
                        target_language=session.target_language,
                        native_language=session.native_language,
                        learner_level=session.cefr_level,
                        focus_area=session.focus_area,
                        recent_turns=session.turns,
                        persona_style=persona_style,
                        llm_device_preference=effective_llm_device_preference,
                        asr_confidence_band=asr_confidence_band,
                    )
                    selected_model = model
                    fast_coaching_result = fast_result
                    fast_latency_ms = observed_fast_latency
                    fast_follow_up = str(fast_result.get("follow_up_question") or "").strip()
                    fast_reply = str(fast_result.get("reply") or "").strip()
                    quick_recast = str(fast_result.get("quick_recast") or "").strip()
                    needs_confirmation = bool(fast_result.get("needs_confirmation"))
                    confirmation_text = str(fast_result.get("confirmation_text") or "").strip()
                    if needs_confirmation and confirmation_text:
                        fast_reply = confirmation_text
                    elif quick_recast and asr_confidence_band != "low":
                        fast_reply = f"{quick_recast} {fast_reply}".strip()
                    fast_output = fast_reply
                    if fast_follow_up and fast_follow_up not in fast_reply:
                        fast_output = f"{fast_reply.rstrip()} {fast_follow_up}".strip()
                    fast_output = self._apply_direct_question_guard(
                        transcript=transcript,
                        candidate_reply=fast_output,
                        target_language=session.target_language,
                        persona_style=persona_style,
                    )
                    fast_output = self._prevent_repeat_loop(
                        candidate_reply=fast_output,
                        last_coach_reply=last_coach_reply,
                        target_language=session.target_language,
                        native_language=session.native_language,
                        learner_level=session.cefr_level,
                        focus_area=session.focus_area,
                        learner_transcript=transcript,
                    )
                    fast_output = self._sanitize_coach_tone(fast_output)
                    fast_output = self._append_native_translation(
                        message=fast_output,
                        translation=str(fast_result.get("response_translation") or "").strip(),
                        target_language=session.target_language,
                        native_language=session.native_language,
                    )
                    if fast_output:
                        yield {
                            "type": "coach_reply",
                            "turn_id": turn_id,
                            "text": fast_output,
                            "question": fast_follow_up,
                        }

                result, observed_latency = await self._coach_with_model(
                    model=model,
                    transcript=transcript,
                    target_language=session.target_language,
                    native_language=session.native_language,
                    cefr_level=session.cefr_level,
                    focus_area=session.focus_area,
                    recent_turns=session.turns,
                    persona_style=persona_style,
                    llm_device_preference=effective_llm_device_preference,
                )
                if (not two_pass_enabled) and observed_latency > self.default_latency_budget_ms and idx + 1 < len(candidates):
                    if respect_user_model_choice and model == requested_model:
                        logger.info(
                            "Keeping user-selected coach model '%s' despite high latency (%sms > %sms)",
                            model,
                            observed_latency,
                            self.default_latency_budget_ms,
                        )
                        selected_model = model
                        full_coaching_result = result
                        full_latency_ms = observed_latency
                        break
                    yield {
                        "type": "model_fallback",
                        "turn_id": turn_id,
                        "requested_model": model,
                        "selected_model": candidates[idx + 1],
                        "reason": "latency_budget_exceeded",
                        "observed_latency_ms": observed_latency,
                    }
                    continue
                selected_model = model
                full_coaching_result = result
                full_latency_ms = observed_latency
                break
            except Exception as exc:
                logger.warning("Coach model '%s' failed: %s", model, exc)
                if idx + 1 < len(candidates):
                    yield {
                        "type": "model_fallback",
                        "turn_id": turn_id,
                        "requested_model": model,
                        "selected_model": candidates[idx + 1],
                        "reason": "model_call_failed",
                    }
                    continue

        if full_coaching_result is None:
            full_coaching_result = self._fallback_coaching(
                transcript,
                target_language=session.target_language,
                native_language=session.native_language,
                learner_level=session.cefr_level,
                focus_area=session.focus_area,
                persona_style=persona_style,
            )
            selected_model = "fallback-heuristic"
            full_latency_ms = None

        follow_up_question = str((fast_coaching_result or {}).get("follow_up_question") or "").strip()
        reply_text = str((fast_coaching_result or {}).get("reply") or "").strip()
        full_mistakes = list(full_coaching_result.get("mistakes") or [])
        if not reply_text:
            follow_up_question = str(full_coaching_result.get("follow_up_question") or "").strip()
            reply_text = str(full_coaching_result.get("reply") or "").strip()
        final_reply = reply_text
        if follow_up_question and follow_up_question not in reply_text:
            final_reply = f"{reply_text.rstrip()} {follow_up_question}".strip()
        final_reply = self._prepend_explicit_recast(
            reply=final_reply,
            correction=str(full_coaching_result.get("correction") or "").strip() or None,
            mistakes=full_mistakes,
            confidence_band=asr_confidence_band,
        )
        final_reply = self._apply_direct_question_guard(
            transcript=transcript,
            candidate_reply=final_reply,
            target_language=session.target_language,
            persona_style=persona_style,
        )
        final_reply = self._prevent_repeat_loop(
            candidate_reply=final_reply,
            last_coach_reply=last_coach_reply,
            target_language=session.target_language,
            native_language=session.native_language,
            learner_level=session.cefr_level,
            focus_area=session.focus_area,
            learner_transcript=transcript,
        )
        final_reply = self._sanitize_coach_tone(final_reply)
        final_reply = self._append_native_translation(
            message=final_reply,
            translation=str(
                full_coaching_result.get("response_translation")
                or (fast_coaching_result or {}).get("response_translation")
                or ""
            ).strip(),
            target_language=session.target_language,
            native_language=session.native_language,
        )

        if not fast_coaching_result:
            yield {
                "type": "coach_reply",
                "turn_id": turn_id,
                "text": final_reply,
                "question": follow_up_question,
            }

        combined_latency_ms: Optional[int] = None
        if fast_latency_ms is not None or full_latency_ms is not None:
            combined_latency_ms = int((fast_latency_ms or 0) + (full_latency_ms or 0))

        score_raw = full_coaching_result.get("score")
        score_value: Optional[int]
        if score_raw is None or asr_confidence_band == "low":
            score_value = None
        else:
            try:
                score_value = max(0, min(100, int(score_raw)))
            except (TypeError, ValueError):
                score_value = None

        forced_uncertain_mistake = None
        if asr_confidence_band == "low":
            forced_uncertain_mistake = {
                "category": "audio",
                "detail": "Low ASR confidence. Transcript may be inaccurate.",
                "severity": "low",
                "suggestion": "Please repeat that sentence clearly or type it in text chat.",
            }

        explanation_value = full_coaching_result.get("explanation")
        if asr_confidence_band == "low":
            base_explanation = str(explanation_value or "").strip()
            explanation_value = (
                f"{base_explanation}\nLow ASR confidence: score was withheld.".strip()
                if base_explanation
                else "Low ASR confidence: score was withheld."
            )

        saved_turn = await self.save_turn_with_mistakes(
            session_id=str(session_id),
            user_id=user_id,
            transcript=transcript,
            reply=final_reply,
            score=score_value,
            correction=full_coaching_result.get("correction"),
            explanation=explanation_value,
            model_id=selected_model,
            latency_ms=combined_latency_ms,
            mistakes=(
                full_mistakes
                + ([forced_uncertain_mistake] if forced_uncertain_mistake else [])
            ),
        )
        if saved_turn is None:
            raise ValueError("Failed to persist coach turn")

        feedback_payload = {
            "summary": (saved_turn.explanation or "").strip() or "Feedback generated.",
            "mistakes": [
                {
                    "category": mistake.category,
                    "detail": mistake.detail,
                    "severity": mistake.severity,
                    "suggestion": mistake.suggestion,
                }
                for mistake in saved_turn.mistakes
            ],
        }
        if saved_turn.correction:
            feedback_payload["correction"] = saved_turn.correction

        yield {
            "type": "feedback",
            "turn_id": turn_id,
            **feedback_payload,
        }

        yield {
            "type": "score",
            "turn_id": turn_id,
            "value": saved_turn.score,
            "model_id": selected_model,
        }

    async def process_text_turn(
        self,
        *,
        user_id: int,
        session_id: str,
        text: str,
        preferred_model: Optional[str] = None,
        llm_device_preference: Optional[str] = None,
        persona_style: Optional[str] = None,
    ) -> dict[str, Any]:
        session = await self.get_session(session_id=str(session_id), user_id=user_id)
        if session is None:
            raise ValueError("Session not found")
        effective_llm_device_preference = self._effective_device_preference(
            explicit_preference=llm_device_preference,
            persisted_preference=getattr(session, "llm_device_preference", "auto"),
        )

        transcript = re.sub(r"\s+", " ", str(text or "")).strip()
        if not transcript:
            raise ValueError("Text is empty")
        if not self._is_transcript_evaluable(transcript):
            raise ValueError("Please write at least one clear sentence.")

        available_models = await self._available_ollama_models()
        candidates = self.get_quality_first_model_order()
        if available_models:
            allowed = set(available_models)
            candidates = [candidate for candidate in candidates if candidate in allowed] or candidates
        if preferred_model and preferred_model.strip():
            normalized_model = preferred_model.strip()
            if normalized_model in candidates:
                candidates = [normalized_model] + [candidate for candidate in candidates if candidate != normalized_model]

        selected_model = candidates[0]
        last_coach_reply = ""
        if session.turns:
            last_coach_reply = str(session.turns[-1].reply or "").strip()
        coaching_result: Optional[dict[str, Any]] = None
        latency_ms: Optional[int] = None
        for model in candidates:
            try:
                result, observed_latency = await self._coach_with_model(
                    model=model,
                    transcript=transcript,
                    target_language=session.target_language,
                    native_language=session.native_language,
                    cefr_level=session.cefr_level,
                    focus_area=session.focus_area,
                    recent_turns=session.turns,
                    persona_style=persona_style,
                    llm_device_preference=effective_llm_device_preference,
                )
                selected_model = model
                coaching_result = result
                latency_ms = observed_latency
                break
            except Exception as exc:
                logger.warning("Coach text model '%s' failed: %s", model, exc)
                continue

        if coaching_result is None:
            coaching_result = self._fallback_coaching(
                transcript,
                target_language=session.target_language,
                native_language=session.native_language,
                learner_level=session.cefr_level,
                focus_area=session.focus_area,
                persona_style=persona_style,
            )
            selected_model = "fallback-heuristic"
            latency_ms = None

        lt_findings: list[dict[str, Any]] = []
        try:
            lt_findings = await self._languagetool_check(text=transcript, language=session.target_language)
        except Exception:
            lt_findings = []

        merged_mistakes = list(coaching_result.get("mistakes") or [])
        merged_mistakes.extend(lt_findings)

        follow_up_question = str(coaching_result.get("follow_up_question") or "").strip()
        reply_text = str(coaching_result.get("reply") or "").strip()
        final_reply = reply_text
        if follow_up_question and follow_up_question not in reply_text:
            final_reply = f"{reply_text.rstrip()} {follow_up_question}".strip()
        final_reply = self._prepend_explicit_recast(
            reply=final_reply,
            correction=str(coaching_result.get("correction") or "").strip() or None,
            mistakes=merged_mistakes,
            confidence_band="high",
        )
        final_reply = self._apply_direct_question_guard(
            transcript=transcript,
            candidate_reply=final_reply,
            target_language=session.target_language,
            persona_style=persona_style,
        )
        final_reply = self._prevent_repeat_loop(
            candidate_reply=final_reply,
            last_coach_reply=last_coach_reply,
            target_language=session.target_language,
            native_language=session.native_language,
            learner_level=session.cefr_level,
            focus_area=session.focus_area,
            learner_transcript=transcript,
        )
        final_reply = self._sanitize_coach_tone(final_reply)
        final_reply = self._append_native_translation(
            message=final_reply,
            translation=str(coaching_result.get("response_translation") or "").strip(),
            target_language=session.target_language,
            native_language=session.native_language,
        )
        explanation = str(coaching_result.get("explanation") or "").strip()
        if lt_findings:
            lt_hint = f"LanguageTool flagged {len(lt_findings)} additional rule-based issue(s)."
            explanation = f"{explanation}\n{lt_hint}".strip()

        score_raw = coaching_result.get("score")
        try:
            score_value = max(0, min(100, int(score_raw)))
        except (TypeError, ValueError):
            score_value = None

        saved_turn = await self.save_turn_with_mistakes(
            session_id=str(session_id),
            user_id=user_id,
            transcript=transcript,
            reply=final_reply,
            score=score_value,
            correction=coaching_result.get("correction"),
            explanation=explanation,
            model_id=selected_model,
            latency_ms=latency_ms,
            mistakes=merged_mistakes,
        )
        if saved_turn is None:
            raise ValueError("Failed to persist coach turn")
        return self._serialize_turn(saved_turn)


coach_service = CoachService()
