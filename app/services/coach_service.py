import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Mapping, Optional, Sequence

import httpx
from sqlalchemy import desc, func, select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.logger import get_logger
from app.db.database import AsyncSessionLocal
from app.db.models import CoachMistake, CoachSession, CoachTurn

logger = get_logger("bipod.services.coach")

LatencyProbe = Callable[[str], Optional[float]]
LatencyFallback = Callable[[List[str]], Optional[str]]


class CoachService:
    OLLAMA_CONNECT_TIMEOUT_SEC = 10.0
    OLLAMA_WRITE_TIMEOUT_SEC = 30.0
    OLLAMA_POOL_TIMEOUT_SEC = 30.0
    COACH_MAX_AUDIO_BYTES = 20 * 1024 * 1024

    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL
        self.default_latency_budget_ms = 15000.0
        self._whisper_model = None
        self._whisper_lock = asyncio.Lock()

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
    ) -> CoachSession:
        session_id = str(uuid.uuid4())
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

    async def save_turn_with_mistakes(
        self,
        session_id: str,
        user_id: int,
        transcript: str,
        reply: str,
        score: int,
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

                compute_type = "float16" if settings.USE_GPU else "int8"
                model_name = os.environ.get("COACH_WHISPER_MODEL", "small")
                self._whisper_model = WhisperModel(model_name, device="cuda" if settings.USE_GPU else "cpu", compute_type=compute_type)
                logger.info("Loaded faster-whisper model for coach transcription: %s", model_name)
            except Exception as exc:
                logger.warning("Could not load faster-whisper model: %s", exc)
                self._whisper_model = None
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
            logger.warning("Coach transcription failed, using fallback transcript: %s", exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        return f"captured {len(audio_bytes)} bytes of audio"

    def _fallback_coaching(self, transcript: str) -> dict:
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
        return {
            "reply": f"Great effort. Let's refine this response: {transcript}",
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
    ) -> tuple[dict, int]:
        system_prompt = (
            "You are a strict language coach. Return JSON only.\n"
            "Requirements:\n"
            "1) Always include fields: reply (string), score (0-100 integer), mistakes (array).\n"
            "2) Include correction and explanation only if needed.\n"
            "3) explanation must be bilingual: target language first, native language second.\n"
            "4) mistakes must include category, detail, severity, suggestion.\n"
        )
        user_prompt = (
            f"Target language: {target_language}\n"
            f"Native language: {native_language or 'Unknown'}\n"
            f"CEFR level: {cefr_level}\n"
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
            language=session.target_language[:2].lower() if session.target_language else None,
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
            coaching_result = self._fallback_coaching(transcript)
            selected_model = "fallback-heuristic"
            latency_ms = None

        saved_turn = await self.save_turn_with_mistakes(
            session_id=str(session_id),
            user_id=user_id,
            transcript=transcript,
            reply=str(coaching_result.get("reply") or ""),
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
