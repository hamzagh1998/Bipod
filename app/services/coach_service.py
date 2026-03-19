import asyncio
import base64
from collections import Counter
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
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import desc, func, select
from sqlalchemy.orm import selectinload

from app.core.config import settings
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

    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL
        self.default_latency_budget_ms = 15000.0
        self._whisper_model = None
        self._whisper_lock = asyncio.Lock()
        self._whisper_runtime_device: Optional[str] = None
        self._voice_cipher: Optional[Fernet] = None
        self._voice_cipher_key: Optional[str] = None

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
        return self._normalize_model_candidates(
            [
                settings.HEAVY_MODEL,
                settings.SMART_MODEL,
                settings.MEDIUM_MODEL,
                settings.LIGHT_MODEL,
            ]
        )

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

    def _session_counts(self, session: CoachSession) -> tuple[int, int]:
        turn_count = len(session.turns)
        mistake_count = sum(len(turn.mistakes) for turn in session.turns)
        return turn_count, mistake_count

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
    ) -> CoachSession:
        session_id = str(uuid.uuid4())
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
                target_language=(target_language or "English").strip() or "English",
                native_language=native_language.strip() if isinstance(native_language, str) else native_language,
                cefr_level=(cefr_level or "A2").strip() or "A2",
                audio_retention_opt_in=bool(audio_retention_opt_in),
                focus_area=focus_area.strip() if isinstance(focus_area, str) else focus_area,
                model_id=model_id.strip() if isinstance(model_id, str) else model_id,
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
                    "voice_profile_id": coach_session.voice_profile_id,
                    "status": coach_session.status,
                    "created_at": coach_session.created_at,
                    "updated_at": coach_session.updated_at,
                    "turn_count": turn_count,
                    "mistake_count": mistake_count,
                }
            )
        return response

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

                mistake = CoachMistake(
                    session_id=session_id,
                    turn_id=coach_turn.id,
                    user_id=user_id,
                    category=str(data.get("category") or "general").strip() or "general",
                    detail=detail,
                    severity=str(data.get("severity") or "medium").strip() or "medium",
                    suggestion=(
                        str(data["suggestion"]).strip()
                        if data.get("suggestion") is not None
                        else None
                    ),
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

        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", normalized)
        if len(words) < 2:
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
        if len(meaningful_words) < 2:
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
    ) -> tuple[bytes, str]:
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

        payload: dict[str, Any] = {
            "text": text,
            "language": language or "English",
            "voice_preset": voice_preset or "default",
            "persona_style": persona_style or "",
            "voice_mode": voice_mode,
            "model_id": model_id,
        }
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

    async def synthesize_reply_audio(
        self,
        *,
        text: str,
        language: Optional[str],
        voice_preset: Optional[str],
        persona_style: Optional[str] = None,
        tts_provider: Optional[str] = None,
        voice_mode: Optional[str] = "preset",
        voice_profile_id: Optional[str] = None,
        reference_clip_id: Optional[str] = None,
        builtin_voice_id: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> tuple[bytes, str]:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            raise ValueError("Text is empty")

        clipped = normalized[:2000]
        resolved_provider = str(
            tts_provider
            or os.environ.get("COACH_TTS_PROVIDER", settings.COACH_TTS_PROVIDER)
            or settings.COACH_TTS_PROVIDER
        ).strip().lower()
        resolved_mode = str(voice_mode or "preset").strip().lower() or "preset"
        if resolved_mode not in {"preset", "cloned_profile", "cloned_session"}:
            raise ValueError("Unsupported voice_mode")
        normalized_builtin_voice_id = str(builtin_voice_id or "").strip().lower()
        if normalized_builtin_voice_id and self._builtin_voice_record(normalized_builtin_voice_id) is None:
            raise ValueError(f"Built-in voice not found: {normalized_builtin_voice_id}")
        resolved_preset = str(voice_preset or "default").strip().lower()
        if "anby" in str(persona_style or "").lower() and resolved_preset == "default":
            resolved_preset = "anby"

        if resolved_provider == "cosyvoice":
            if user_id is None:
                raise ValueError("user_id is required for cosyvoice synthesis")
            cosyvoice_mode = resolved_mode
            if normalized_builtin_voice_id and cosyvoice_mode == "preset":
                # Built-in library voices are driven through clone reference samples.
                cosyvoice_mode = "cloned_session"
            requires_clone_voice = bool(normalized_builtin_voice_id) or cosyvoice_mode in {"cloned_profile", "cloned_session"}
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
            return result.scalars().all()

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
            return result.scalars().all()

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

    async def _get_whisper_model(self):
        if self._whisper_model is not None:
            return self._whisper_model
        async with self._whisper_lock:
            if self._whisper_model is not None:
                return self._whisper_model
            try:
                from faster_whisper import WhisperModel

                configured_device = os.environ.get(
                    "COACH_WHISPER_DEVICE",
                    settings.COACH_WHISPER_DEVICE,
                ).strip().lower()
                if configured_device not in {"auto", "cpu", "cuda"}:
                    configured_device = "auto"
                runtime_device = "cuda" if configured_device == "cuda" or (
                    configured_device == "auto" and settings.USE_GPU
                ) else "cpu"

                compute_type = os.environ.get(
                    "COACH_WHISPER_COMPUTE_TYPE",
                    settings.COACH_WHISPER_COMPUTE_TYPE,
                ).strip()
                if not compute_type:
                    compute_type = "float16" if runtime_device == "cuda" else "int8"
                model_name = os.environ.get("COACH_WHISPER_MODEL", settings.COACH_WHISPER_MODEL).strip()
                model_path = os.environ.get("COACH_WHISPER_MODEL_PATH", settings.COACH_WHISPER_MODEL_PATH).strip()
                model_ref = model_path or model_name or "small"

                local_only_raw = os.environ.get(
                    "COACH_WHISPER_LOCAL_FILES_ONLY",
                    str(settings.COACH_WHISPER_LOCAL_FILES_ONLY),
                ).strip().lower()
                local_files_only = local_only_raw in {"1", "true", "yes", "on"}

                download_root = os.environ.get(
                    "COACH_WHISPER_DOWNLOAD_ROOT",
                    settings.COACH_WHISPER_DOWNLOAD_ROOT,
                ).strip()

                whisper_kwargs: Dict[str, Any] = {
                    "device": runtime_device,
                    "compute_type": compute_type,
                    "local_files_only": local_files_only,
                }
                if download_root:
                    whisper_kwargs["download_root"] = download_root

                try:
                    self._whisper_model = WhisperModel(model_ref, **whisper_kwargs)
                except TypeError:
                    # Compatibility fallback for older faster-whisper versions.
                    whisper_kwargs.pop("local_files_only", None)
                    self._whisper_model = WhisperModel(model_ref, **whisper_kwargs)

                self._whisper_runtime_device = runtime_device
                logger.info(
                    "Loaded faster-whisper model for coach transcription: model_ref=%s device=%s compute_type=%s local_files_only=%s download_root=%s",
                    model_ref,
                    runtime_device,
                    compute_type,
                    local_files_only,
                    download_root or "<default>",
                )
            except Exception as exc:
                logger.warning("Could not load faster-whisper model: %s", exc)
                self._whisper_model = None
                self._whisper_runtime_device = None
            return self._whisper_model

    async def _transcribe_audio(
        self,
        audio_bytes: bytes,
        *,
        filename: Optional[str] = None,
        transcript_hint: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        if transcript_hint and transcript_hint.strip():
            return re.sub(r"\s+", " ", transcript_hint).strip()

        whisper_model = await self._get_whisper_model()
        suffix = ".webm"
        if filename and "." in filename:
            suffix = f".{filename.rsplit('.', 1)[-1]}"
        if whisper_model is None:
            return f"captured {len(audio_bytes)} bytes of audio"

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            segments, _ = whisper_model.transcribe(temp_path, language=language or None)
            text = " ".join(segment.text.strip() for segment in segments if segment.text and segment.text.strip())
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                return text
        except Exception as exc:
            error_text = str(exc).lower()
            cuda_runtime_error = any(
                marker in error_text for marker in ("libcublas", "cuda", "cudnn", "cublas")
            )
            if cuda_runtime_error and self._whisper_runtime_device == "cuda":
                logger.warning(
                    "Coach transcription CUDA runtime issue detected (%s). Retrying on CPU.",
                    exc,
                )
                self._whisper_model = None
                self._whisper_runtime_device = None
                os.environ["COACH_WHISPER_DEVICE"] = "cpu"
                cpu_model = await self._get_whisper_model()
                if cpu_model is not None:
                    try:
                        segments, _ = cpu_model.transcribe(temp_path, language=language or None)
                        text = " ".join(
                            segment.text.strip() for segment in segments if segment.text and segment.text.strip()
                        )
                        text = re.sub(r"\s+", " ", text).strip()
                        if text:
                            return text
                    except Exception as retry_exc:
                        logger.warning("Coach transcription CPU retry failed: %s", retry_exc)
            logger.warning("Coach transcription failed, using fallback transcript: %s", exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        return f"captured {len(audio_bytes)} bytes of audio"

    def _fallback_coaching(
        self,
        transcript: str,
        *,
        focus_area: Optional[str] = None,
        persona_style: Optional[str] = None,
    ) -> dict:
        word_count = len(transcript.split())
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
        score = 80 if word_count >= 5 else 68
        correction = mistakes[0]["suggestion"] if mistakes and mistakes[0]["category"] != "general" else None
        explanation = (
            "Target: keep your answer precise.\nNative: keep expanding with concrete examples."
        )
        topic = str(focus_area or "").strip()
        if topic:
            follow_up_question = f"Can you give one concrete example about {topic}?"
        else:
            follow_up_question = "Can you answer again with one concrete example?"
        persona_hint = str(persona_style or "").strip()
        persona_prefix = "Acknowledged. " if persona_hint else ""
        return {
            "reply": f"{persona_prefix}Great effort. Let's refine this response: {transcript}",
            "follow_up_question": follow_up_question,
            "score": score,
            "correction": correction,
            "explanation": explanation,
            "mistakes": mistakes,
        }

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
    ) -> tuple[dict, int]:
        subject = str(focus_area or "").strip()
        persona = str(persona_style or "").strip()
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
            "You are a strict language coach running a live conversation drill. Return JSON only.\n"
            "Requirements:\n"
            "1) Always include fields: reply (string), follow_up_question (string), score (0-100 integer), mistakes (array).\n"
            "2) follow_up_question must be one short question that keeps the learner talking in the target language.\n"
            "3) The follow_up_question must stay on the selected subject when provided.\n"
            "4) Avoid repeating the exact previous coach question.\n"
            "5) Include correction and explanation only if needed.\n"
            "6) explanation must be bilingual: target language first, native language second.\n"
            "7) mistakes must include category, detail, severity, suggestion.\n"
            "8) Keep a consistent speaking style based on Persona instructions when provided.\n"
        )
        user_prompt = (
            f"Target language: {target_language}\n"
            f"Native language: {native_language or 'Unknown'}\n"
            f"CEFR level: {cefr_level}\n"
            f"Subject: {subject or 'General conversation'}\n"
            f"Persona instructions: {persona or 'Default coach'}\n"
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
            score = int(score_raw)
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
        persona_style: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        session = await self.get_session(session_id=str(session_id), user_id=user_id)
        if session is None:
            raise ValueError("Session not found")

        audio_bytes = await audio.read(self.COACH_MAX_AUDIO_BYTES + 1)
        if not audio_bytes:
            raise ValueError("Uploaded audio file is empty")
        if len(audio_bytes) > self.COACH_MAX_AUDIO_BYTES:
            raise ValueError("Audio file is too large")
        await audio.seek(0)

        transcript = await self._transcribe_audio(
            audio_bytes=audio_bytes,
            filename=getattr(audio, "filename", None),
            transcript_hint=transcript_hint,
            language=self._target_language_code(session.target_language),
        )
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
        }

        if not self._is_transcript_evaluable(transcript):
            retry_reply = (
                "I couldn't catch a clear spoken answer. Please hold the button and respond with one full sentence."
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
            }
            return

        available_models = await self._available_ollama_models()
        candidates = self.get_quality_first_model_order()
        if available_models:
            allowed = set(available_models)
            candidates = [candidate for candidate in candidates if candidate in allowed] or candidates

        if preferred_model and preferred_model.strip():
            preferred_model = preferred_model.strip()
            if preferred_model in candidates:
                candidates = [preferred_model] + [candidate for candidate in candidates if candidate != preferred_model]
            else:
                yield {
                    "type": "model_fallback",
                    "turn_id": turn_id,
                    "requested_model": preferred_model,
                    "selected_model": candidates[0],
                    "reason": "requested_model_unavailable",
                }

        selected_model = candidates[0]
        coaching_result = None
        latency_ms = None
        for idx, model in enumerate(candidates):
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
                )
                if observed_latency > self.default_latency_budget_ms and idx + 1 < len(candidates):
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
                coaching_result = result
                latency_ms = observed_latency
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

        if coaching_result is None:
            coaching_result = self._fallback_coaching(
                transcript,
                focus_area=session.focus_area,
                persona_style=persona_style,
            )
            selected_model = "fallback-heuristic"
            latency_ms = None

        follow_up_question = str(coaching_result.get("follow_up_question") or "").strip()
        reply_text = str(coaching_result.get("reply") or "")
        final_reply = reply_text
        if follow_up_question and follow_up_question not in reply_text:
            final_reply = f"{reply_text.rstrip()} {follow_up_question}".strip()

        saved_turn = await self.save_turn_with_mistakes(
            session_id=str(session_id),
            user_id=user_id,
            transcript=transcript,
            reply=final_reply,
            score=int(coaching_result.get("score") or 0),
            correction=coaching_result.get("correction"),
            explanation=coaching_result.get("explanation"),
            model_id=selected_model,
            latency_ms=latency_ms,
            mistakes=coaching_result.get("mistakes") or [],
        )
        if saved_turn is None:
            raise ValueError("Failed to persist coach turn")

        yield {
            "type": "coach_reply",
            "turn_id": turn_id,
            "text": saved_turn.reply,
            "question": follow_up_question,
        }

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
            "value": saved_turn.score if saved_turn.score is not None else 0,
        }


coach_service = CoachService()
