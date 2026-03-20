from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger("bipod.cosyvoice")


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


class CosyVoiceRuntime:
    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: Optional[str] = None
        self._model_device: Optional[str] = None
        self._last_init_error: str = ""
        self._init_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._warmup_thread: Optional[threading.Thread] = None
        self._status_state = "idle"
        self._status_detail = "Voice model is idle."
        self._status_updated_at = time.time()
        self._status_model_id: Optional[str] = None
        self._builtin_voice_cache: dict[str, bytes] = {}

    def _set_status(self, *, state: str, detail: str, model_id: Optional[str]) -> None:
        with self._status_lock:
            self._status_state = str(state)
            self._status_detail = str(detail)
            self._status_updated_at = time.time()
            self._status_model_id = model_id

    def status_snapshot(self, *, requested_model_id: Optional[str]) -> dict[str, Any]:
        with self._status_lock:
            state = str(self._status_state)
            detail = str(self._status_detail)
            updated_at = float(self._status_updated_at)
            status_model_id = self._status_model_id
            warmup_active = bool(self._warmup_thread and self._warmup_thread.is_alive())
        loaded_model_id = str(self._model_id or "")
        loaded_runtime_device = str(self._model_device or "")
        ready = bool(self._model is not None and loaded_model_id)
        if ready and state != "ready":
            state = "ready"
            detail = "Voice model ready."
        return {
            "ok": True,
            "service": "cosyvoice-sidecar",
            "engine": "cosyvoice",
            "ready": ready,
            "state": state,
            "detail": detail,
            "requested_model_id": str(requested_model_id or ""),
            "model_id": str(status_model_id or requested_model_id or ""),
            "loaded_model_id": loaded_model_id,
            "runtime_device": loaded_runtime_device,
            "warmup_active": warmup_active,
            "updated_at": updated_at,
        }

    def _detect_gpu_available(self) -> bool:
        try:
            probe = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
            return bool(str(probe.stdout or "").strip())
        except Exception:
            return False

    def _detect_gpu_vram_gb(self) -> float:
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            )
            values_mb = [float(line.strip()) for line in str(probe.stdout or "").splitlines() if line.strip()]
            if not values_mb:
                return 0.0
            return max(values_mb) / 1024.0
        except Exception:
            return 0.0

    def _resolve_runtime_device(self, requested_device: Optional[str]) -> str:
        runtime_device = str(requested_device or "").strip().lower()
        if runtime_device in {"cpu", "cuda"}:
            return runtime_device

        configured = str(os.environ.get("COACH_COSYVOICE_DEVICE") or "auto").strip().lower()
        if configured in {"cpu", "cuda"}:
            return configured

        threshold_raw = str(os.environ.get("COACH_HIGH_VRAM_THRESHOLD_GB") or "16").strip()
        try:
            high_vram_threshold = max(8.0, float(threshold_raw))
        except ValueError:
            high_vram_threshold = 16.0
        if self._detect_gpu_available() and self._detect_gpu_vram_gb() >= high_vram_threshold:
            return "cuda"
        return "cpu"

    def ensure_warmup(self, requested_model_id: Optional[str], requested_device: Optional[str] = None) -> bool:
        model_id = str(requested_model_id or os.environ.get("COACH_COSYVOICE_MODEL_ID") or "").strip() or "iic/CosyVoice-300M"
        runtime_device = self._resolve_runtime_device(requested_device)
        if self._model is not None and self._model_id == model_id and self._model_device == runtime_device:
            self._set_status(state="ready", detail="Voice model ready.", model_id=model_id)
            return False
        with self._warmup_lock:
            if self._warmup_thread and self._warmup_thread.is_alive():
                return False
            self._set_status(
                state="warming",
                detail=f"Warming voice model ({model_id}) on {runtime_device}. First synthesis may take a while.",
                model_id=model_id,
            )
            thread = threading.Thread(target=self._init_model, args=(model_id, runtime_device), daemon=True)
            self._warmup_thread = thread
            thread.start()
            return True

    def _to_wav_bytes(self, samples: np.ndarray, sample_rate: int) -> bytes:
        normalized = np.asarray(samples, dtype=np.float32).flatten()
        clipped = np.clip(normalized, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(int(sample_rate))
            writer.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def _extract_audio(self, item: Any) -> tuple[np.ndarray, int]:
        sample_rate = 22050
        payload = item
        if isinstance(item, dict):
            sample_rate = int(item.get("sample_rate") or item.get("sampling_rate") or sample_rate)
            for key in ("tts_speech", "audio", "speech", "wav"):
                if key in item:
                    payload = item[key]
                    break
        if isinstance(payload, (tuple, list)) and payload:
            payload = payload[0]

        if hasattr(payload, "detach"):
            payload = payload.detach().cpu().numpy() # type: ignore
        payload = np.asarray(payload, dtype=np.float32)
        return payload, sample_rate

    def _merge_outputs(self, outputs: list[Any]) -> tuple[np.ndarray, int]:
        chunks: list[np.ndarray] = []
        sample_rate: Optional[int] = None
        for item in outputs:
            samples, chunk_sample_rate = self._extract_audio(item)
            normalized = np.asarray(samples, dtype=np.float32).flatten()
            if not normalized.size:
                continue
            if sample_rate is None:
                sample_rate = int(chunk_sample_rate)
            elif int(chunk_sample_rate) != int(sample_rate):
                logger.debug(
                    "Skipping CosyVoice chunk with mismatched sample_rate=%s (expected=%s)",
                    chunk_sample_rate,
                    sample_rate,
                )
                continue
            chunks.append(normalized)
        if not chunks:
            raise RuntimeError("CosyVoice inference returned empty audio chunks.")
        if len(chunks) == 1:
            return chunks[0], int(sample_rate or 22050)
        return np.concatenate(chunks), int(sample_rate or 22050)

    def _init_model(self, requested_model_id: Optional[str], runtime_device: Optional[str] = None) -> bool:
        with self._init_lock:
            model_id = str(requested_model_id or os.environ.get("COACH_COSYVOICE_MODEL_ID") or "").strip()
            if not model_id:
                model_id = "iic/CosyVoice-300M"
            resolved_device = self._resolve_runtime_device(runtime_device)
            if self._model is not None and self._model_id == model_id and self._model_device == resolved_device:
                self._set_status(state="ready", detail="Voice model ready.", model_id=model_id)
                return True

            model_source = model_id
            if not Path(model_id).exists():
                self._set_status(state="downloading", detail=f"Downloading voice model assets for {model_id}.", model_id=model_id)
                local_only = str(os.environ.get("COACH_COSYVOICE_LOCAL_FILES_ONLY", "false")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                cache_dir = str(os.environ.get("COACH_COSYVOICE_DOWNLOAD_ROOT", "/app/data/modelscope")).strip()
                allow_patterns = [
                    "cosyvoice*.yaml",
                    "llm.pt",
                    "flow.pt",
                    "hift.pt",
                    "campplus.onnx",
                    "speech_tokenizer*.onnx",
                    "spk2info.pt",
                    "CosyVoice-BlankEN/*",
                    "tokenizer*",
                    "*.json",
                    "*.model",
                    "*.txt",
                ]
                ignore_patterns = [
                    "*.zip",
                    "*.plan",
                    "*.engine",
                    "*.onnx.data",
                    "*.tar",
                    "*.tensorrt*",
                ]
                try:
                    from modelscope import snapshot_download  # type: ignore
                    model_source = snapshot_download(
                        model_id,
                        cache_dir=cache_dir,
                        local_files_only=local_only,
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                    )
                except Exception as exc:
                    self._last_init_error = f"CosyVoice model download failed ({model_id}): {exc}"
                    self._set_status(state="error", detail=self._last_init_error, model_id=model_id)
                    logger.exception("CosyVoice model download failed for model_id=%s", model_id)
                    self._model = None
                    self._model_id = None
                    self._model_device = None
                    return False

            self._set_status(state="loading", detail=f"Loading voice model {model_id}.", model_id=model_id)
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice  # type: ignore
            except Exception as exc:
                self._last_init_error = f"CosyVoice import failed: {exc}"
                self._set_status(state="error", detail=self._last_init_error, model_id=model_id)
                logger.exception("CosyVoice import failed for model_id=%s", model_id)
                self._model = None
                self._model_id = None
                self._model_device = None
                return False

            kwargs = {"device": resolved_device}

            # CosyVoice constructor signatures vary by release; keep this permissive.
            try:
                self._model = CosyVoice(model_source, **kwargs)
            except TypeError:
                try:
                    self._model = CosyVoice(model_source)
                except Exception as exc:
                    self._last_init_error = f"CosyVoice model load failed ({model_id}): {exc}"
                    self._set_status(state="error", detail=self._last_init_error, model_id=model_id)
                    logger.exception("CosyVoice model load failed for model_id=%s", model_id)
                    self._model = None
                    self._model_id = None
                    self._model_device = None
                    return False
            except Exception as exc:
                self._last_init_error = f"CosyVoice model load failed ({model_id}): {exc}"
                self._set_status(state="error", detail=self._last_init_error, model_id=model_id)
                logger.exception("CosyVoice model load failed for model_id=%s", model_id)
                self._model = None
                self._model_id = None
                self._model_device = None
                return False

            self._last_init_error = ""
            self._model_id = model_id
            self._model_device = resolved_device
            self._set_status(state="ready", detail="Voice model ready.", model_id=model_id)
            return True

    def _convert_reference_to_wav(self, raw_bytes: bytes) -> Path:
        input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        input_path.write(raw_bytes)
        input_path.flush()
        input_path.close()
        output_path.close()
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path.name,
                    "-ac",
                    "1",
                    "-ar",
                    "22050",
                    output_path.name,
                ],
                check=True,
                capture_output=True,
            )
        except Exception:
            # Fall back to raw bytes if ffmpeg cannot decode the payload.
            Path(output_path.name).unlink(missing_ok=True)
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            output_path.write(raw_bytes)
            output_path.flush()
            output_path.close()
        finally:
            Path(input_path.name).unlink(missing_ok=True)
        return Path(output_path.name)

    def _voice_library_samples_dir(self) -> Path:
        root = str(os.environ.get("COACH_VOICE_LIBRARY_DIR") or "/app/data/coach_voice_library").strip()
        return Path(root) / "clone samples"

    def _load_builtin_reference_bytes(self, voice_id: str) -> bytes:
        normalized = str(voice_id or "").strip().lower()
        if not normalized:
            raise RuntimeError("Built-in voice id is required")
        cached = self._builtin_voice_cache.get(normalized)
        if cached:
            return cached

        sample_dir = self._voice_library_samples_dir()
        if not sample_dir.exists():
            raise RuntimeError("Voice library samples directory is missing.")

        matched_path: Optional[Path] = None
        for file_path in sample_dir.iterdir():
            if not file_path.is_file():
                continue
            lower_name = file_path.name.lower()
            if f"[{normalized}]" in lower_name or normalized in lower_name:
                matched_path = file_path
                break
        if matched_path is None:
            raise RuntimeError(f"Built-in voice sample not found for: {normalized}")
        payload = matched_path.read_bytes()
        if not payload:
            raise RuntimeError(f"Built-in voice sample is empty for: {normalized}")
        self._builtin_voice_cache[normalized] = payload
        return payload

    def _espeak_fallback(self, text: str, voice_preset: str) -> bytes:
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if not binary:
            raise RuntimeError("CosyVoice is unavailable and espeak fallback is not installed.")
        preset = str(voice_preset or "default").strip().lower()
        voice = "en-us"
        if preset == "male":
            voice = "en-us+m3"
        elif preset in {"female", "anby"}:
            voice = "en-us+f3"

        output_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                output_path = temp_file.name
            subprocess.run(
                [binary, "-v", voice, "-s", "160", "-w", output_path, text],
                check=True,
                capture_output=True,
                text=True,
            )
            return Path(output_path).read_bytes()
        finally:
            if output_path:
                try:
                    Path(output_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def synthesize(
        self,
        *,
        text: str,
        voice_mode: str,
        voice_preset: str,
        model_id: Optional[str],
        reference_audio_bytes: Optional[bytes],
        runtime_device: Optional[str] = None,
    ) -> bytes:
        clone_mode = voice_mode in {"cloned_profile", "cloned_session"}
        if not self._init_model(model_id, runtime_device):
            if clone_mode and reference_audio_bytes:
                detail = self._last_init_error or "CosyVoice clone model is unavailable in sidecar."
                raise RuntimeError(detail)
            return self._espeak_fallback(text=text, voice_preset=voice_preset)

        reference_path: Optional[Path] = None
        try:
            if reference_audio_bytes:
                reference_path = self._convert_reference_to_wav(reference_audio_bytes)

            outputs = None
            if voice_mode in {"cloned_profile", "cloned_session"} and reference_path:
                # Prefer cross-lingual cloning for reference-audio voices. It does not
                # rely on prompt transcript text and is more stable for custom samples.
                if hasattr(self._model, "inference_cross_lingual"):
                    cross_lingual_kwargs = [
                        {"tts_text": text, "prompt_wav": str(reference_path), "stream": False},
                        {"text": text, "prompt_wav": str(reference_path), "stream": False},
                    ]
                    for kwargs in cross_lingual_kwargs:
                        try:
                            outputs = list(self._model.inference_cross_lingual(**kwargs))
                            if outputs:
                                break
                        except Exception:
                            logger.debug("CosyVoice cross-lingual kwargs failed: %s", kwargs)
                            continue

                # Keep zero-shot as a compatibility fallback.
                if outputs is None and hasattr(self._model, "inference_zero_shot"):
                    prompt_text = str(os.environ.get("COACH_COSYVOICE_PROMPT_TEXT") or "").strip()
                    zero_shot_kwargs = []
                    if prompt_text:
                        zero_shot_kwargs.extend(
                            [
                                {
                                    "tts_text": text,
                                    "prompt_text": prompt_text,
                                    "prompt_wav": str(reference_path),
                                    "stream": False,
                                },
                                {
                                    "text": text,
                                    "prompt_text": prompt_text,
                                    "prompt_wav": str(reference_path),
                                    "stream": False,
                                },
                            ]
                        )
                    zero_shot_kwargs.extend(
                        [
                            {
                                "tts_text": text,
                                "prompt_text": "",
                                "prompt_wav": str(reference_path),
                                "stream": False,
                            },
                            {
                                "text": text,
                                "prompt_text": "",
                                "prompt_wav": str(reference_path),
                                "stream": False,
                            },
                        ]
                    )
                    for kwargs in zero_shot_kwargs:
                        try:
                            outputs = list(self._model.inference_zero_shot(**kwargs))
                            if outputs:
                                break
                        except Exception:
                            logger.debug("CosyVoice zero-shot kwargs failed: %s", kwargs)
                            continue

            if outputs is None and hasattr(self._model, "inference_sft"):
                trial_kwargs = [
                    {"tts_text": text, "spk_id": "default", "stream": False},
                    {"text": text, "spk_id": "default", "stream": False},
                ]
                for kwargs in trial_kwargs:
                    try:
                        outputs = list(self._model.inference_sft(**kwargs))
                        if outputs:
                            break
                    except Exception:
                        logger.debug("CosyVoice SFT kwargs failed: %s", kwargs)
                        continue

            if not outputs:
                raise RuntimeError("CosyVoice inference returned no audio.")

            samples, sample_rate = self._merge_outputs(outputs)
            return self._to_wav_bytes(samples=samples, sample_rate=sample_rate)
        finally:
            if reference_path is not None:
                try:
                    reference_path.unlink(missing_ok=True)
                except OSError:
                    pass


app = FastAPI(title="Bipod CosyVoice Sidecar")
runtime = CosyVoiceRuntime()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "cosyvoice-sidecar"}


@app.get("/status")
def status(warm: bool = False, runtime_device: Optional[str] = None) -> dict[str, Any]:
    model_id = str(os.environ.get("COACH_COSYVOICE_MODEL_ID") or "").strip() or "iic/CosyVoice-300M"
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

    reference_bytes = None
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

    return Response(content=audio, media_type="audio/wav", headers={"X-TTS-Engine": "cosyvoice"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
