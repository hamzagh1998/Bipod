from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger("bipod.openvoice")

_IMPORT_ERROR = ""
try:
    import requests
    import torch
    from melo.api import TTS
    from openvoice.api import ToneColorConverter
except Exception as exc:  # pragma: no cover - runtime dependency guard
    requests = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    TTS = None  # type: ignore[assignment]
    ToneColorConverter = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(exc)


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: Optional[str] = Field(default="English")
    voice_preset: Optional[str] = Field(default="default")
    persona_style: Optional[str] = Field(default=None)
    voice_mode: str = Field(default="preset")
    model_id: Optional[str] = Field(default=None)
    reference_audio_b64: Optional[str] = Field(default=None)
    builtin_voice_id: Optional[str] = Field(default=None)
    runtime_device: Optional[str] = Field(default=None)


class OpenVoiceRuntime:
    LANGUAGE_TO_MELO = {
        "ar": "EN",
        "arabic": "EN",
        "de": "EN",
        "german": "EN",
        "el": "EN",
        "greek": "EN",
        "en": "EN_NEWEST",
        "english": "EN_NEWEST",
        "es": "ES",
        "spanish": "ES",
        "fr": "FR",
        "french": "FR",
        "hi": "EN",
        "hindi": "EN",
        "it": "EN",
        "italian": "EN",
        "ja": "JP",
        "japanese": "JP",
        "ko": "KR",
        "korean": "KR",
        "nl": "EN",
        "dutch": "EN",
        "pl": "EN",
        "polish": "EN",
        "pt": "ES",
        "portuguese": "ES",
        "ru": "EN",
        "russian": "EN",
        "sv": "EN",
        "swedish": "EN",
        "tr": "EN",
        "turkish": "EN",
        "uk": "EN",
        "ukrainian": "EN",
        "ur": "EN",
        "urdu": "EN",
        "zh": "ZH",
        "chinese": "ZH",
    }

    def __init__(self) -> None:
        self._init_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._warmup_thread: Optional[threading.Thread] = None
        self._status_state = "idle"
        self._status_detail = "OpenVoice model is idle."
        self._status_updated_at = time.time()
        self._status_model_id = ""
        self._converter: Any = None
        self._melo_models: dict[tuple[str, str], Any] = {}
        self._model_id = ""
        self._model_device = ""
        self._checkpoints_dir: Optional[Path] = None
        self._last_init_error = ""
        self._builtin_voice_cache: dict[str, bytes] = {}

    def _set_status(self, *, state: str, detail: str, model_id: Optional[str]) -> None:
        with self._status_lock:
            self._status_state = str(state)
            self._status_detail = str(detail)
            self._status_updated_at = time.time()
            self._status_model_id = str(model_id or "")

    def status_snapshot(self, *, requested_model_id: Optional[str]) -> dict[str, Any]:
        with self._status_lock:
            state = str(self._status_state)
            detail = str(self._status_detail)
            updated_at = float(self._status_updated_at)
            status_model_id = str(self._status_model_id or requested_model_id or "")
            warmup_active = bool(self._warmup_thread and self._warmup_thread.is_alive())
        ready = bool(self._converter is not None and self._model_id)
        if ready and state != "ready":
            state = "ready"
            detail = "OpenVoice model ready."
        return {
            "ok": True,
            "service": "openvoice-sidecar",
            "engine": "openvoice",
            "ready": ready,
            "state": state,
            "detail": detail,
            "requested_model_id": str(requested_model_id or ""),
            "model_id": status_model_id,
            "loaded_model_id": str(self._model_id or ""),
            "runtime_device": str(self._model_device or ""),
            "warmup_active": warmup_active,
            "updated_at": updated_at,
        }

    def _detect_gpu_available(self) -> bool:
        if torch is None:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _resolve_runtime_device(self, requested_device: Optional[str]) -> str:
        runtime_device = str(requested_device or "").strip().lower()
        if runtime_device in {"cpu", "cuda"}:
            if runtime_device == "cuda" and not self._detect_gpu_available():
                return "cpu"
            return runtime_device

        configured = str(os.environ.get("COACH_OPENVOICE_DEVICE") or "auto").strip().lower()
        if configured in {"cpu", "cuda"}:
            if configured == "cuda" and not self._detect_gpu_available():
                return "cpu"
            return configured
        return "cuda" if self._detect_gpu_available() else "cpu"

    def _language_key(self, language: Optional[str]) -> str:
        raw = str(language or "").strip().lower()
        if raw in self.LANGUAGE_TO_MELO:
            return self.LANGUAGE_TO_MELO[raw]
        if len(raw) == 2 and raw in self.LANGUAGE_TO_MELO:
            return self.LANGUAGE_TO_MELO[raw]
        return "EN_NEWEST"

    def _download_root(self) -> Path:
        configured = str(
            os.environ.get("COACH_OPENVOICE_DOWNLOAD_ROOT")
            or os.environ.get("COACH_COSYVOICE_DOWNLOAD_ROOT")
            or "/app/data/openvoice"
        ).strip()
        root = Path(configured or "/app/data/openvoice")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _discover_checkpoints_dir(self, root: Path) -> Optional[Path]:
        direct = root / "checkpoints_v2"
        if (direct / "converter" / "config.json").exists():
            return direct
        for config_path in root.glob("**/converter/config.json"):
            candidate = config_path.parent.parent
            if (candidate / "base_speakers" / "ses").exists():
                return candidate
        return None

    def _checkpoints_url(self) -> str:
        return str(
            os.environ.get("COACH_OPENVOICE_CHECKPOINTS_URL")
            or "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"
        ).strip()

    def _local_files_only(self) -> bool:
        return str(os.environ.get("COACH_OPENVOICE_LOCAL_FILES_ONLY", "false")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _ensure_checkpoints(self) -> Path:
        root = self._download_root()
        existing = self._discover_checkpoints_dir(root)
        if existing is not None:
            return existing
        if self._local_files_only():
            raise RuntimeError(
                "OpenVoice checkpoints are missing and COACH_OPENVOICE_LOCAL_FILES_ONLY=true. "
                "Disable local-only mode or preload checkpoints."
            )
        if requests is None:
            raise RuntimeError(f"OpenVoice downloader is unavailable: {_IMPORT_ERROR}")

        checkpoints_url = self._checkpoints_url()
        if not checkpoints_url:
            raise RuntimeError("OpenVoice checkpoint URL is empty.")

        archive_path = root / "checkpoints_v2.zip"
        self._set_status(
            state="downloading",
            detail="Downloading OpenVoice checkpoints...",
            model_id="openvoice-v2",
        )
        with requests.get(checkpoints_url, stream=True, timeout=1800) as response:
            response.raise_for_status()
            with open(archive_path, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    file_handle.write(chunk)

        self._set_status(
            state="loading",
            detail="Extracting OpenVoice checkpoints...",
            model_id="openvoice-v2",
        )
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(root)
        extracted = self._discover_checkpoints_dir(root)
        if extracted is None:
            raise RuntimeError("OpenVoice checkpoints downloaded but required files were not found.")
        return extracted

    def _pick_speaker(self, model: Any, voice_preset: str) -> tuple[int, str]:
        mapping = getattr(getattr(model, "hps", None), "data", None)
        spk2id = getattr(mapping, "spk2id", {}) if mapping is not None else {}
        if not isinstance(spk2id, dict) or not spk2id:
            raise RuntimeError("OpenVoice base TTS speaker map is unavailable.")
        keys = list(spk2id.keys())
        preset = str(voice_preset or "default").strip().lower()
        selected_key = ""
        if preset == "male":
            selected_key = next((key for key in keys if "male" in key.lower()), "")
        elif preset in {"female", "anby"}:
            selected_key = next((key for key in keys if "female" in key.lower()), "")
        if not selected_key:
            selected_key = next((key for key in keys if "default" in key.lower()), keys[0])
        speaker_id = int(spk2id[selected_key])
        speaker_key = selected_key.lower().replace("_", "-")
        return speaker_id, speaker_key

    def _source_se_path(self, speaker_key: str) -> Path:
        checkpoints = self._checkpoints_dir or self._ensure_checkpoints()
        candidate = checkpoints / "base_speakers" / "ses" / f"{speaker_key}.pth"
        if candidate.exists():
            return candidate
        fallback = checkpoints / "base_speakers" / "ses" / "en-newest.pth"
        if fallback.exists():
            return fallback
        raise RuntimeError(f"OpenVoice base speaker embedding is missing for {speaker_key}.")

    def _voice_library_samples_dir(self) -> Path:
        root = Path(
            str(os.environ.get("COACH_VOICE_LIBRARY_DIR") or "/app/data/coach_voice_library").strip()
            or "/app/data/coach_voice_library"
        )
        return root / "clone samples"

    def _load_builtin_reference_bytes(self, voice_id: str) -> bytes:
        normalized = str(voice_id or "").strip().lower()
        if not normalized:
            raise RuntimeError("Built-in voice id is required.")
        if normalized in self._builtin_voice_cache:
            return self._builtin_voice_cache[normalized]

        sample_dir = self._voice_library_samples_dir()
        if not sample_dir.exists():
            raise RuntimeError("Built-in voice library directory is missing.")
        selected_path: Optional[Path] = None
        for file_path in sample_dir.iterdir():
            if not file_path.is_file():
                continue
            lower_name = file_path.name.lower()
            token = f"[{normalized}]"
            if token in lower_name or normalized in lower_name:
                selected_path = file_path
                break
        if selected_path is None:
            raise RuntimeError(f"Built-in voice sample is missing: {normalized}")
        payload = selected_path.read_bytes()
        if not payload:
            raise RuntimeError(f"Built-in voice sample is empty: {normalized}")
        self._builtin_voice_cache[normalized] = payload
        return payload

    def _load_or_create_melo(self, language_key: str, runtime_device: str) -> Any:
        cache_key = (language_key, runtime_device)
        if cache_key in self._melo_models:
            return self._melo_models[cache_key]
        model = TTS(language=language_key, device=runtime_device)
        self._melo_models[cache_key] = model
        return model

    def _init_model(self, model_id: Optional[str], runtime_device: Optional[str]) -> bool:
        target_model_id = str(model_id or os.environ.get("COACH_OPENVOICE_MODEL_ID") or "openvoice-v2").strip()
        target_device = self._resolve_runtime_device(runtime_device)

        if self._converter is not None and self._model_id == target_model_id and self._model_device == target_device:
            return True
        with self._init_lock:
            if self._converter is not None and self._model_id == target_model_id and self._model_device == target_device:
                return True
            if ToneColorConverter is None or TTS is None or torch is None:
                self._converter = None
                self._melo_models = {}
                self._model_id = ""
                self._model_device = ""
                self._checkpoints_dir = None
                self._last_init_error = _IMPORT_ERROR or "OpenVoice dependencies are unavailable."
                self._set_status(
                    state="error",
                    detail=f"OpenVoice dependencies are unavailable: {self._last_init_error}",
                    model_id=target_model_id,
                )
                return False
            try:
                self._set_status(state="loading", detail="Loading OpenVoice model...", model_id=target_model_id)
                checkpoints = self._ensure_checkpoints()
                converter_config = checkpoints / "converter" / "config.json"
                converter_ckpt = checkpoints / "converter" / "checkpoint.pth"
                if not converter_config.exists() or not converter_ckpt.exists():
                    raise RuntimeError("OpenVoice converter checkpoint files are missing.")

                converter = ToneColorConverter(str(converter_config), device=target_device, enable_watermark=False)
                converter.load_ckpt(str(converter_ckpt))
                self._converter = converter
                self._melo_models = {}
                self._model_id = target_model_id
                self._model_device = target_device
                self._checkpoints_dir = checkpoints
                self._last_init_error = ""
                self._set_status(
                    state="ready",
                    detail="OpenVoice model ready.",
                    model_id=target_model_id,
                )
                return True
            except Exception as exc:
                self._converter = None
                self._melo_models = {}
                self._model_id = ""
                self._model_device = ""
                self._checkpoints_dir = None
                self._last_init_error = str(exc)
                self._set_status(
                    state="error",
                    detail=f"OpenVoice model load failed: {exc}",
                    model_id=target_model_id,
                )
                logger.exception("OpenVoice model load failed")
                return False

    def ensure_warmup(self, model_id: Optional[str], runtime_device: Optional[str]) -> None:
        with self._warmup_lock:
            if self._warmup_thread and self._warmup_thread.is_alive():
                return

            def _runner() -> None:
                self._init_model(model_id, runtime_device)

            thread = threading.Thread(target=_runner, name="openvoice-warmup", daemon=True)
            self._warmup_thread = thread
            thread.start()

    def _guess_audio_extension(self, payload: bytes) -> str:
        if payload.startswith(b"RIFF"):
            return ".wav"
        if payload.startswith(b"ID3") or payload[:2] == b"\xff\xfb":
            return ".mp3"
        if payload.startswith(b"OggS"):
            return ".ogg"
        if b"ftyp" in payload[:32]:
            return ".m4a"
        return ".wav"

    def _write_reference_file(self, reference_audio_bytes: bytes, workdir: Path) -> Path:
        guessed_ext = self._guess_audio_extension(reference_audio_bytes)
        raw_path = workdir / f"reference{guessed_ext}"
        raw_path.write_bytes(reference_audio_bytes)
        if guessed_ext == ".wav":
            return raw_path

        wav_path = workdir / "reference.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_path), "-ac", "1", "-ar", "24000", str(wav_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return wav_path
        except Exception:
            return raw_path

    def synthesize(
        self,
        *,
        text: str,
        language: Optional[str],
        voice_mode: str,
        voice_preset: str,
        model_id: Optional[str],
        reference_audio_bytes: Optional[bytes],
        runtime_device: Optional[str] = None,
    ) -> bytes:
        clone_mode = voice_mode in {"cloned_profile", "cloned_session"}
        if not self._init_model(model_id, runtime_device):
            detail = self._last_init_error or "OpenVoice model is unavailable."
            if clone_mode:
                raise RuntimeError(detail)
            raise RuntimeError(detail)

        if self._converter is None:
            raise RuntimeError("OpenVoice converter is not initialized.")

        runtime_device_resolved = str(self._model_device or self._resolve_runtime_device(runtime_device)).strip() or "cpu"
        language_key = self._language_key(language)
        model = self._load_or_create_melo(language_key=language_key, runtime_device=runtime_device_resolved)
        speaker_id, speaker_key = self._pick_speaker(model=model, voice_preset=voice_preset)

        with tempfile.TemporaryDirectory(prefix="openvoice-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            src_path = tmp_root / "source.wav"
            model.tts_to_file(
                text,
                speaker_id,
                str(src_path),
                speed=1.0,
                quiet=True,
            )

            if not clone_mode or not reference_audio_bytes:
                payload = src_path.read_bytes()
                if not payload:
                    raise RuntimeError("OpenVoice base synthesis returned empty audio.")
                return payload

            reference_path = self._write_reference_file(reference_audio_bytes=reference_audio_bytes, workdir=tmp_root)
            try:
                target_se, _ = self._converter.extract_se([str(reference_path)], se_save_path=str(tmp_root / "target_se.pt"))
            except Exception as exc:
                raise RuntimeError(f"OpenVoice target speaker embedding failed: {exc}") from exc

            source_se_path = self._source_se_path(speaker_key=speaker_key)
            source_se = torch.load(str(source_se_path), map_location=runtime_device_resolved)
            out_path = tmp_root / "output.wav"
            self._converter.convert(
                audio_src_path=str(src_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(out_path),
                message="@Bipod",
            )
            payload = out_path.read_bytes()
            if not payload:
                raise RuntimeError("OpenVoice conversion returned empty audio.")
            return payload


app = FastAPI(title="Bipod OpenVoice Sidecar")
runtime = OpenVoiceRuntime()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "openvoice-sidecar"}


@app.get("/status")
def status(warm: bool = False, runtime_device: Optional[str] = None) -> dict[str, Any]:
    model_id = str(os.environ.get("COACH_OPENVOICE_MODEL_ID") or "openvoice-v2").strip() or "openvoice-v2"
    if warm:
        runtime.ensure_warmup(model_id, runtime_device)
    snapshot = runtime.status_snapshot(requested_model_id=model_id)
    resolved_device = runtime._resolve_runtime_device(runtime_device)
    snapshot["runtime_device"] = str(snapshot.get("runtime_device") or resolved_device)
    return snapshot


@app.post("/synthesize")
def synthesize(payload: SynthesizeRequest):
    text = str(payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    voice_mode = str(payload.voice_mode or "preset").strip().lower()
    if voice_mode not in {"preset", "cloned_profile", "cloned_session"}:
        raise HTTPException(status_code=400, detail="Unsupported voice mode")

    reference_bytes: Optional[bytes] = None
    builtin_voice_id = str(payload.builtin_voice_id or "").strip().lower()
    if builtin_voice_id:
        try:
            reference_bytes = runtime._load_builtin_reference_bytes(builtin_voice_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if payload.reference_audio_b64:
        try:
            reference_bytes = base64.b64decode(payload.reference_audio_b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid reference audio payload") from exc

    try:
        audio = runtime.synthesize(
            text=text,
            language=payload.language,
            voice_mode=voice_mode,
            voice_preset=str(payload.voice_preset or "default"),
            model_id=payload.model_id,
            reference_audio_bytes=reference_bytes,
            runtime_device=payload.runtime_device,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content=audio, media_type="audio/wav", headers={"X-TTS-Engine": "openvoice"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5002)
