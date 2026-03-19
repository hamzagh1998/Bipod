# 🦵 Bipod: Weightless Intelligence

Bipod is a self-sovereign AI companion designed to be free from the gravity of the cloud. It is a local-first, hardware-agnostic system that scales from high-end workstations to Raspberry Pis.

## 🌌 Project Philosophy

- **Locality is Law:** Data never leaves your machine unless explicitly requested.
- **Hardware Agnostic:** Runs on NVIDIA GPUs or falls back to optimized CPU inference.
- **True Agency:** A system entity that interacts with files, cameras, and microphones.

---

## 🛠️ Hardware Setup: NVIDIA GPU Configuration

To leverage GPU acceleration for inference (Ollama) and audio processing (Faster-Whisper), you must configure the NVIDIA Container Toolkit.

### 🍎 Fedora (Verified)

1. **Install the Toolkit:**

   ```bash
   sudo dnf install -y nvidia-container-toolkit
   ```

2. **Configure Docker Runtime:**

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

3. **Restart Docker:**

```bash
sudo systemctl restart docker
```

### 🐧 Ubuntu / Debian

1. **Setup the Repository:**

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

2. **Install the Toolkit:**

   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

3. **Configure Docker Runtime:**

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

4. **Restart Docker:**

```bash
sudo systemctl restart docker
```

### ✅ Verification

Test if Docker can access the GPU:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Getting Started

Bipod uses a **Sidecar Pattern**, separating the Inference Server (Ollama) from the Logic Server (FastAPI).

1. **Clone the repository.**
2. **Launch the stack:**

   ```bash
   docker compose up -d
   ```

3. **Check the logs:**

   ```bash
   docker compose logs -f
   ```

## 🧩 Coach Frontend Build (Local Bundle)

The `/coach` page now uses a locally bundled React build (no CDN React/Babel at runtime).

```bash
# Install JS build dependencies once
npm install

# Build coach bundle
npm run build:coach

# Optional: rebuild automatically while editing
npm run build:coach:watch
```

When building `bipod-app` via Docker, the coach bundle is now built automatically in the image (`docker/Dockerfile.app`).
`docker-compose.yaml` intentionally does not bind-mount `./frontend` so container runs always use the prebuilt bundle from the image.

## 🎙️ Coach Preload (Recommended)

The first time you open `/coach`, Faster-Whisper and CosyVoice may download model assets.  
To avoid first-run stalls, preload coach packages/models manually:

```bash
./scripts/preload-coach-models.sh
```

What this script does:

- builds `bipod-app` and `cosyvoice` images (runtime packages ready)
- starts `ollama` and `languagetool`
- auto-selects runtime profile (`cpu`, `gpu_constrained`, `gpu_full`) from GPU VRAM (or `COACH_RUNTIME_PROFILE` override)
- pulls and warms Ollama tutor model chain for the active profile
- auto-detects GPU VRAM and picks ASR preload defaults:
  - constrained profile (`< 16GB VRAM`): fast=`medium`, accurate=`large-v3`
  - full GPU profile (`>= 16GB VRAM`): fast=`large-v3`, accurate=`large-v3`
- downloads Faster-Whisper fast/accurate models to `data/huggingface/hub`
- downloads required CosyVoice assets to `data/modelscope`
- starts `bipod-app` + `cosyvoice` (plus `ollama` + `languagetool`)
- warms CosyVoice status endpoint to reduce first-turn TTS delay
- verifies LanguageTool readiness endpoint (`/v2/languages`)

You can override the automatic picks:

- `COACH_RUNTIME_PROFILE=cpu|gpu_constrained|gpu_full|auto`
- `COACH_HIGH_VRAM_THRESHOLD_GB=16`
- `COACH_WHISPER_FAST_MODEL=...`
- `COACH_WHISPER_ACCURATE_MODEL=...`
- `HEAVY_MODEL`, `SMART_MODEL`, `MEDIUM_MODEL`, `LIGHT_MODEL` (for Ollama pulls)

Threshold guidance:

- `COACH_HIGH_VRAM_THRESHOLD_GB` controls when `auto` switches to `gpu_full`.
- Example: setting it to `6` means a 6GB GPU is treated as `gpu_full` (aggressive: ASR/TTS on CUDA).
- For most 6-8GB cards, keep threshold at `16` or force `COACH_RUNTIME_PROFILE=gpu_constrained`.
- Recommended for 6-8GB:

```bash
COACH_RUNTIME_PROFILE=auto COACH_HIGH_VRAM_THRESHOLD_GB=16 ./scripts/preload-coach-models.sh
```

Resume behavior:

- Download steps are cached in `data/preload_cache/coach/done.steps`.
- If the script is interrupted, rerun it and it will skip completed pull/download steps.
- Set `COACH_PRELOAD_FORCE=1` to ignore cache markers and force all download steps again.

## ⚡ Coach Runtime Preload (In-App)

Coach now exposes runtime warmup endpoints so the UI can preheat models based on mode:

- `POST /api/v1/coach/runtime/preload` with `{ "mode": "voice" | "text" | "idle" }`
- `GET /api/v1/coach/runtime/status?warm=true&mode=voice`

Current preload behavior:

- `voice` mode warms Ollama + ASR (fast model) + TTS
- `text` mode warms Ollama (and optional LanguageTool sidecar)
- `idle` mode only reports status, no warmup

Runtime status now returns an inferred coach profile (`cpu`, `gpu_constrained`, `gpu_full`) and the selected ASR/TTS device strategy so UI/services can stay predictable across hardware tiers.

## 🌐 Coach Language Selection

Coach setup now consumes backend-supported languages from:

- `GET /api/v1/coach/languages/supported`

Only selectable languages are shown in the setup screen, and the selected language is used for ASR hinting, coaching/corrections, and TTS output.

## 🧠 Required Models

Bipod uses different Ollama models for each brain tier. Pull the models below into the `bipod_ollama` container before using those tiers.

Run these commands to install the supported brain and utility models:

```bash
# Smart Tier (Higher intelligence & reliable tools)
docker exec -it bipod_ollama ollama pull qwen2.5:7b

# Heavy Tier (Creative baseline / default GPU-heavy option)
docker exec -it bipod_ollama ollama pull qwen3:8b

# Medium Tier (Standard CPU)
docker exec -it bipod_ollama ollama pull llama3.2:3b

# Light Tier (Edge / Low Resource)
docker exec -it bipod_ollama ollama pull llama3.2:1b

# Vision Capabilities (Image Analysis)
docker exec -it bipod_ollama ollama pull moondream

# Embedding Service (Long-Term Memory/RAG)
docker exec -it bipod_ollama ollama pull nomic-embed-text
```

## 🎨 Imagine Studio & Image/video Generation

Bipod features a professional-grade **Imagine Studio** for high-quality, local image and video generation.

### ✨ Features

- **Standalone Page**: Dedicated workspace for creation.
- **Flux.1-schnell**: Top-tier photorealism and complex prompt following.
- **SDXL Lightning**: High-speed, high-quality generation for daily tasks.
- **Hardware Aware**: Dynamic resolution and capability scaling based on GPU VRAM.
- **Batch Processing**: Generate multiple variations simultaneously.

### 🚀 Preloading Models (Recommended)

To avoid long wait times during your first session, pre-download the model suite (~45 GB total).

**Important: Flux.1-schnell requires Hugging Face authentication.**

1. **Accept the License:**
   Go to [huggingface.co/black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), scroll to the license section, and click **"Agree and access repository"**. (It's instant).

2. **Login to Hugging Face:**
   Run this in your terminal to authenticate (requires a Read access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

   ```bash
   docker exec -it bipod_imagine huggingface-cli login
   ```

3. **Preload everything:**

   ```bash
   docker exec -it bipod_imagine python preload.py
   ```

_Once complete, Bipod can generate images entirely offline._

### ⚙️ Hardware Optimization

Bipod automatically detects your GPU and scales the engine:

| Tier       | Max Quality | Recommended VRAM | Optimization Strategy      |
| :--------- | :---------- | :--------------- | :------------------------- |
| **Ultra**  | 2048x2048   | 24GB+            | No Offload                 |
| **High**   | 1536x1024   | 12-16GB          | Model Offload (Flux Ready) |
| **Medium** | 1024x1024   | 8-10GB           | Model Offload              |
| **Low**    | 512x512     | <6GB             | Aggressive Offload (Tiled) |

_Bipod ensures stability by applying tiling and slicing optimizations for lower-tier hardware._

## 🔋 Edge Device Support (Raspberry Pi)

Bipod is designed to scale down. On devices without a GPU, it gracefully falls back to:

- **Quantized GGUF models** for CPU inference.
- **Moondream** (Efficient Mode) for vision tasks.

---

> _"Intelligence should not require a subscription."_
