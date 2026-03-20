#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CACHE_DIR="${ROOT_DIR}/data/preload_cache/coach"
STATE_FILE="${CACHE_DIR}/done.steps"
mkdir -p "${CACHE_DIR}"
touch "${STATE_FILE}"

FORCE_PRELOAD_RAW="${COACH_PRELOAD_FORCE:-0}"
FORCE_PRELOAD="0"
case "${FORCE_PRELOAD_RAW,,}" in
  1|true|yes|on) FORCE_PRELOAD="1" ;;
esac

step_key() {
  printf "%s" "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

step_done() {
  local key="$1"
  if [[ "${FORCE_PRELOAD}" == "1" ]]; then
    return 1
  fi
  grep -Fxq "${key}" "${STATE_FILE}"
}

mark_step_done() {
  local key="$1"
  if ! grep -Fxq "${key}" "${STATE_FILE}"; then
    echo "${key}" >> "${STATE_FILE}"
  fi
}

echo "[coach-preload] Building coach images (installs runtime packages)..."
docker compose build bipod-app cosyvoice openvoice

detect_gpu_vram_gb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return 0
  fi
  local mb
  mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{print $1}')"
  if [[ -z "${mb}" ]]; then
    echo "0"
    return 0
  fi
  awk -v mb="${mb}" 'BEGIN { printf "%.2f", mb/1024 }'
}

HIGH_VRAM_THRESHOLD_GB="${COACH_HIGH_VRAM_THRESHOLD_GB:-16}"
GPU_VRAM_GB="$(detect_gpu_vram_gb)"
RUNTIME_PROFILE="${COACH_RUNTIME_PROFILE:-auto}"

detect_runtime_profile() {
  local requested="$1"
  local vram="$2"
  local threshold="$3"
  local has_gpu="0"
  awk -v v="$vram" 'BEGIN { exit !(v > 0.01) }' && has_gpu="1"

  case "${requested}" in
    cpu|gpu_constrained|gpu_full)
      echo "${requested}"
      return 0
      ;;
  esac

  if [[ "${has_gpu}" != "1" ]]; then
    echo "cpu"
    return 0
  fi

  awk -v v="$vram" 'BEGIN { exit !(v < 6.0) }' && {
    echo "cpu"
    return 0
  }
  awk -v v="$vram" -v t="$threshold" 'BEGIN { exit !(v >= t) }' && {
    echo "gpu_full"
    return 0
  }
  echo "gpu_constrained"
}

RUNTIME_PROFILE="$(detect_runtime_profile "${RUNTIME_PROFILE}" "${GPU_VRAM_GB}" "${HIGH_VRAM_THRESHOLD_GB}")"

HEAVY_MODEL="${HEAVY_MODEL:-qwen3:8b}"
SMART_MODEL="${SMART_MODEL:-qwen2.5:7b}"
MEDIUM_MODEL="${MEDIUM_MODEL:-llama3.2:3b}"
LIGHT_MODEL="${LIGHT_MODEL:-llama3.2:1b}"

if [[ "${COACH_WHISPER_FAST_MODEL:-auto}" == "auto" || -z "${COACH_WHISPER_FAST_MODEL:-}" ]]; then
  FAST_ASR_MODEL="medium"
  awk -v have="${GPU_VRAM_GB}" -v need="${HIGH_VRAM_THRESHOLD_GB}" 'BEGIN { exit !(have >= need) }' \
    && FAST_ASR_MODEL="large-v3"
else
  FAST_ASR_MODEL="${COACH_WHISPER_FAST_MODEL}"
fi

if [[ "${COACH_WHISPER_ACCURATE_MODEL:-auto}" == "auto" || -z "${COACH_WHISPER_ACCURATE_MODEL:-}" ]]; then
  ACCURATE_ASR_MODEL="large-v3"
else
  ACCURATE_ASR_MODEL="${COACH_WHISPER_ACCURATE_MODEL}"
fi

if [[ "${RUNTIME_PROFILE}" == "cpu" ]]; then
  OLLAMA_MODELS=("${MEDIUM_MODEL}" "${SMART_MODEL}" "${LIGHT_MODEL}" "${HEAVY_MODEL}")
else
  OLLAMA_MODELS=("${HEAVY_MODEL}" "${SMART_MODEL}" "${MEDIUM_MODEL}" "${LIGHT_MODEL}")
fi

echo "[coach-preload] Detected GPU VRAM: ${GPU_VRAM_GB} GB (high-tier threshold: ${HIGH_VRAM_THRESHOLD_GB} GB, profile: ${RUNTIME_PROFILE})"

echo "[coach-preload] Starting Ollama and LanguageTool..."
docker compose up -d ollama languagetool

echo "[coach-preload] Waiting for Ollama..."
for _ in $(seq 1 60); do
  if docker exec bipod_ollama ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "[coach-preload] Pulling Ollama tutor models for profile ${RUNTIME_PROFILE}..."
ollama_model_installed() {
  local model="$1"
  docker exec bipod_ollama ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${model}"
}

for model in "${OLLAMA_MODELS[@]}"; do
  step="ollama_pull_$(step_key "${model}")"
  if ollama_model_installed "${model}"; then
    echo "[coach-preload] ollama model already present: ${model}"
    mark_step_done "${step}"
    continue
  fi
  if step_done "${step}"; then
    echo "[coach-preload] skipping ollama pull (cached step): ${model}"
    continue
  fi
  echo "[coach-preload] ollama pull ${model}"
  docker exec bipod_ollama ollama pull "${model}"
  mark_step_done "${step}"
done

echo "[coach-preload] Warming Ollama models..."
for model in "${OLLAMA_MODELS[@]}"; do
  curl -sS "http://localhost:11434/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${model}\",\"prompt\":\"hello\",\"stream\":false,\"options\":{\"num_predict\":1,\"temperature\":0.1}}" \
    >/dev/null || true
done

echo "[coach-preload] Pulling faster-whisper models (${FAST_ASR_MODEL}, ${ACCURATE_ASR_MODEL}) into /app/data/huggingface/hub ..."
declare -A asr_seen=()
for model_name in "${FAST_ASR_MODEL}" "${ACCURATE_ASR_MODEL}"; do
  if [[ -n "${asr_seen[${model_name}]:-}" ]]; then
    continue
  fi
  asr_seen["${model_name}"]=1
  step="asr_model_$(step_key "${model_name}")"
  if step_done "${step}"; then
    echo "[coach-preload] skipping ASR model (cached step): ${model_name}"
    continue
  fi
  docker compose run --rm \
    -e HF_HUB_OFFLINE=0 \
    -e COACH_WHISPER_LOCAL_FILES_ONLY=false \
    -e COACH_WHISPER_MODEL_NAME="${model_name}" \
    bipod-app \
    python - <<'PY'
from faster_whisper import WhisperModel
import os

model_name = os.environ.get("COACH_WHISPER_MODEL_NAME", "medium").strip() or "medium"
WhisperModel(
    model_name,
    device="cpu",
    compute_type="int8",
    download_root="/app/data/huggingface/hub",
    local_files_only=False,
)
print(f"[coach-preload] faster-whisper model downloaded: {model_name}")
PY
  mark_step_done "${step}"
done

echo "[coach-preload] Pulling CosyVoice model assets into /app/data/modelscope ..."
COSYVOICE_MODEL_ID="${COACH_COSYVOICE_MODEL_ID:-iic/CosyVoice-300M}"
cosy_step="cosyvoice_assets_$(step_key "${COSYVOICE_MODEL_ID}")"
if step_done "${cosy_step}"; then
  echo "[coach-preload] skipping CosyVoice assets (cached step): ${COSYVOICE_MODEL_ID}"
else
  docker compose run --rm \
    -e HF_HUB_OFFLINE=0 \
    -e COACH_COSYVOICE_LOCAL_FILES_ONLY=false \
    -e COACH_COSYVOICE_MODEL_ID="${COSYVOICE_MODEL_ID}" \
    cosyvoice \
    python - <<'PY'
from modelscope import snapshot_download
import os

model_id = os.environ.get("COACH_COSYVOICE_MODEL_ID", "iic/CosyVoice-300M").strip() or "iic/CosyVoice-300M"
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

path = snapshot_download(
    model_id,
    cache_dir="/app/data/modelscope",
    local_files_only=False,
    allow_patterns=allow_patterns,
    ignore_patterns=ignore_patterns,
)
print(f"[coach-preload] CosyVoice model cached at: {path}")
PY
  mark_step_done "${cosy_step}"
fi

echo "[coach-preload] Starting services with preloaded artifacts..."
docker compose up -d bipod-app cosyvoice openvoice languagetool ollama

echo "[coach-preload] Triggering CosyVoice warmup..."
curl -sS "http://localhost:5001/status?warm=true" >/dev/null || true

echo "[coach-preload] Triggering OpenVoice warmup..."
curl -sS "http://localhost:5002/status?warm=true" >/dev/null || true

echo "[coach-preload] Checking LanguageTool readiness..."
LT_OK="0"
for _ in $(seq 1 30); do
  if curl -fsS "http://localhost:8010/v2/languages" >/dev/null; then
    LT_OK="1"
    break
  fi
  sleep 1
done
if [[ "${LT_OK}" == "1" ]]; then
  echo "[coach-preload] LanguageTool ready."
else
  echo "[coach-preload] WARNING: LanguageTool not ready yet (service may still be warming)."
fi

echo "[coach-preload] Done. Coach stack (LLM + ASR + TTS + LT) has been preloaded for profile ${RUNTIME_PROFILE}."
echo "[coach-preload] Resume cache state: ${STATE_FILE} (set COACH_PRELOAD_FORCE=1 to ignore cache markers)"
