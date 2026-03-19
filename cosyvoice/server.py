from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
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


class CosyVoiceRuntime:
    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: Optional[str] = None

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

    def _init_model(self, requested_model_id: Optional[str]) -> bool:
        model_id = str(requested_model_id or os.environ.get("COACH_COSYVOICE_MODEL_ID") or "").strip()
        if not model_id:
            model_id = "FunAudioLLM/CosyVoice3-0.5B"
        if self._model is not None and self._model_id == model_id:
            return True

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice  # type: ignore
        except Exception:
            self._model = None
            self._model_id = None
            return False

        kwargs = {}
        device = str(os.environ.get("COACH_COSYVOICE_DEVICE", "auto")).strip().lower()
        if device in {"cpu", "cuda"}:
            kwargs["device"] = device

        # CosyVoice constructor signatures vary by release; keep this permissive.
        try:
            self._model = CosyVoice(model_id, **kwargs)
        except TypeError:
            self._model = CosyVoice(model_id)
        self._model_id = model_id
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
    ) -> bytes:
        clone_mode = voice_mode in {"cloned_profile", "cloned_session"}
        if not self._init_model(model_id):
            if clone_mode and reference_audio_bytes:
                raise RuntimeError("CosyVoice clone model is unavailable in sidecar.")
            return self._espeak_fallback(text=text, voice_preset=voice_preset)

        reference_path: Optional[Path] = None
        try:
            if reference_audio_bytes:
                reference_path = self._convert_reference_to_wav(reference_audio_bytes)

            outputs = None
            if voice_mode in {"cloned_profile", "cloned_session"} and reference_path and hasattr(self._model, "inference_zero_shot"):
                trial_kwargs = [
                    {"tts_text": text, "prompt_text": "", "prompt_wav": str(reference_path), "stream": False},
                    {"text": text, "prompt_text": "", "prompt_wav": str(reference_path), "stream": False},
                ]
                for kwargs in trial_kwargs:
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

            samples, sample_rate = self._extract_audio(outputs[0])
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


@app.post("/synthesize")
def synthesize(payload: SynthesizeRequest):
    text = str(payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    voice_mode = str(payload.voice_mode or "preset").strip().lower()
    if voice_mode not in {"preset", "cloned_profile", "cloned_session"}:
        raise HTTPException(status_code=400, detail="Unsupported voice mode")

    reference_bytes = None
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
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content=audio, media_type="audio/wav", headers={"X-TTS-Engine": "cosyvoice"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
