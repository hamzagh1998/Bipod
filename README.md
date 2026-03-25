# 🤖 Bipod: Weightless Intelligence

Bipod is a self-sovereign AI companion designed to be free from the cloud. It is a local-first, hardware-agnostic system that scales from high-end workstations to Raspberry Pis.

## 🌌 Project Philosophy

- **Locality is Law:** Data never leaves your machine unless explicitly requested.
- **Hardware Agnostic:** Runs on NVIDIA GPUs or falls back to optimized CPU inference.
- **True Agency:** A system entity that interacts with files, cameras, and microphones.

---

## ✨ Features

- **Conversational AI** — Multi-turn chat with tool-calling, powered by Ollama LLMs (Qwen, Llama, Moondream)
- **Intent Routing** — Automatic classification of user intent (web search, image generation, file operations, vision analysis, coding, system info) with rule-first + semantic fallback
- **Image Generation & Upscaling** — Local Stable Diffusion (SDXL Lightning, Flux.1-schnell, Juggernaut XL, Realistic Vision, Tiny SD) with Swin2SR and Real-ESRGAN upscaling
- **Imagine Studio** — Dedicated workspace for project-based image creation with prompt improvement, batch generation, and organization
- **Language Coach** — Speaking practice in 18+ languages with real-time ASR (Faster-Whisper), pronunciation feedback, grammar correction (LanguageTool), and mistake tracking
- **Voice & TTS** — Text-to-speech via CosyVoice (4 built-in voices) or OpenVoice (voice cloning), with ESpeak fallback
- **Semantic Long-Term Memory** — FAISS vector database with `nomic-embed-text` embeddings for persistent context across conversations
- **Vision Analysis** — Image understanding via Moondream model
- **File Operations** — Sandboxed host filesystem access (read, write, search, move, delete) with path validation
- **Web Search** — DuckDuckGo integration for real-time information retrieval
- **Authentication** — JWT-based user authentication with BCrypt password hashing
- **Hardware-Aware** — Automatic model selection and context window sizing based on detected GPU/CPU/ARM64 capabilities

---

## 🏗️ Architecture

Bipod uses a **Sidecar Pattern** with six containerized services:

| Service | Container | Port | Purpose | GPU |
| :--- | :--- | :--- | :--- | :--- |
| **bipod-app** | `bipod_brain` | 4444 | FastAPI logic server (brain, chat, coach, studio, auth) | Optional |
| **ollama** | `bipod_ollama` | 11434 | LLM inference (Qwen, Llama, Moondream, nomic-embed-text) | Yes |
| **imagine** | `bipod_imagine` | 3333 | Image generation (diffusers) & upscaling | Yes |
| **cosyvoice** | `bipod_cosyvoice` | 5001 | TTS synthesis (CosyVoice-300M) | Optional |
| **openvoice** | `bipod_openvoice` | 5002 | Voice cloning (OpenVoice v2, Melo TTS) | Optional |
| **languagetool** | `bipod_languagetool` | 8010 | Grammar checking | No |

All services share the `./data` volume for models, vectors, generated files, and uploads. Services communicate over an internal `bipod-network` Docker network.

---

## 🧰 Tech Stack

| Area | Technologies |
| :--- | :--- |
| **Backend** | FastAPI, Pydantic, SQLAlchemy (async), aiosqlite, SQLite |
| **AI / LLM** | Ollama, LangChain, semantic-router, FAISS (vector search) |
| **Image Gen** | diffusers, transformers, Swin2SR, Real-ESRGAN, Hugging Face Hub |
| **Speech** | Faster-Whisper (ASR), CosyVoice, OpenVoice / Melo TTS, ESpeak |
| **Auth** | python-jose (JWT/HS256), passlib + BCrypt |
| **Search** | duckduckgo-search |
| **Frontend** | Vanilla HTML/CSS/JS (Chat, Studio), React bundle (Coach), Marked.js, highlight.js |
| **Infrastructure** | Docker Compose, NVIDIA Container Toolkit |

---

## 📁 Project Structure

```
app/                  # FastAPI backend
├── api/              #   Route handlers & Pydantic schemas
├── core/             #   Config (hardware detection, env vars) & logging
├── db/               #   SQLAlchemy models & database setup
├── services/         #   Domain logic
│   ├── brain/        #     Brain sub-services (answer composer, context builder, router, tool orchestrator)
│   ├── brain_service.py    Central intelligence with tool calling
│   ├── coach_service.py    Language coaching pipeline (ASR + TTS + LLM)
│   ├── intent_router.py    Intent classification & routing
│   ├── memory_service.py   Conversation persistence (SQLite)
│   ├── vector_service.py   Semantic long-term memory (FAISS)
│   ├── studio_service.py   Image project management
│   ├── file_service.py     Host filesystem operations (sandboxed)
│   ├── vision_service.py   Image analysis (Moondream)
│   ├── audio_service.py    Audio processing
│   └── auth_service.py     JWT authentication
└── agents/           #   Agent definitions
frontend/             # Static client assets
├── index.html        #   Chat interface
├── studio.html       #   Imagine Studio
├── coach.html        #   Language Coach (React app loader)
├── js/ & css/        #   Client scripts & styles
imagine/              # Image generation sidecar (FastAPI, port 3333)
cosyvoice/            # CosyVoice TTS sidecar (FastAPI, port 5001)
openvoice_sidecar/    # OpenVoice TTS sidecar (FastAPI, port 5002)
tests/                # Pytest test suite
docker/               # Dockerfiles for each service
scripts/              # Build & preload scripts
data/                 # Runtime artifacts (models, vectors, DB, uploads, generated files)
```

---

## 🧠 Brain & Intelligence

### Model Tiers

Bipod automatically selects a brain model based on hardware, or you can override via environment variables:

| Tier | Model | Use Case | Requirement |
| :--- | :--- | :--- | :--- |
| **Smart** | `qwen2.5:7b` | Tool calling, precision tasks | 8GB+ VRAM |
| **Heavy** | `qwen3:8b` | Creative, high-intelligence tasks | 8GB+ VRAM/RAM |
| **Medium** | `llama3.2:3b` | Standard PC / CPU fallback | 4GB+ RAM |
| **Light** | `llama3.2:1b` | Raspberry Pi / edge devices | 2GB+ RAM |
| **Vision** | `moondream` | Image analysis | Minimal |
| **Embedding** | `nomic-embed-text` | Vector embeddings (long-term memory) | Minimal |

### Tool Calling

The brain orchestrates tools iteratively — when a user request requires action (web search, file operation, image generation, vision analysis, system info), the intent router classifies the request and the tool orchestrator manages multi-step tool/model turns with hallucination detection.

### Context Management

- **Recent history**: Last 10 messages kept in full
- **History summarization**: Triggered at 14 messages; older messages are compressed via LLM summary
- **Long-term memory**: Top 3 relevant semantic memories retrieved per turn from the FAISS vector store
- **Attachments**: Images and PDFs included in context (up to 12,000 chars per attachment)

### Semantic Long-Term Memory

Bipod maintains a per-user FAISS vector index for persistent context across conversations:

- **Embedding model**: `nomic-embed-text` via Ollama
- **Chunk size**: Up to 6,000 characters (~1,500 tokens) with overlapping chunks
- **Storage**: Local FAISS index files in `data/vector/`
- **Retrieval**: Top-K semantic similarity search on each turn

---

## 🛠️ Hardware Setup: NVIDIA GPU Configuration

To leverage GPU acceleration for inference (Ollama) and audio processing (Faster-Whisper), you must configure the NVIDIA Container Toolkit.

### 👒 Fedora (Verified)

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

### Docker (Recommended)

1. **Clone the repository.**
2. **Launch the stack:**

   ```bash
   docker compose up -d
   ```

3. **Check the logs:**

   ```bash
   docker compose logs -f
   ```

4. **Pull required models** (see [Required Models](#-required-models) below).
5. **Open** `http://localhost:4444` in your browser.

### Local Development (Without Docker)

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama** locally (see [ollama.com](https://ollama.com)):

   ```bash
   ollama serve
   ```

3. **Run the app:**

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 4444 --reload
   ```

   The app automatically detects whether it's inside Docker or local and adjusts service URLs accordingly (`localhost` vs container hostnames).

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

If Hugging Face models are private/gated (or to avoid anonymous rate limits), set `HF_TOKEN` in `.env` before running preload.

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
- `COACH_COSYVOICE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121|cpu`
- `COACH_OPENVOICE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121|cpu`
- `COACH_COSYVOICE_LOCAL_AFTER_CACHE=true|false` (default: `true`)

Notes on TTS Torch wheels:

- TTS images now default to CUDA wheels (`cu121`) so the same container can run on GPU (when available) or CPU (fallback or manual CPU selection).
- If you explicitly want CPU-only wheels, set:
  - `COACH_COSYVOICE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`
  - `COACH_OPENVOICE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`
- With `COACH_COSYVOICE_LOCAL_AFTER_CACHE=true`, CosyVoice downloads missing assets once, then uses cached local files on later warmups/loads.

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
- Coach UI uses periodic status polling without warm triggers; GPU TTS keepalive uses guarded low-frequency warm pings (every 45s) only during active voice sessions.

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

Bipod features a professional-grade **Imagine Studio** for high-quality, local image generation.

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

## 🎙️ Voice & TTS

Bipod supports three TTS providers, selected via `COACH_TTS_PROVIDER`:

| Provider | Model | Voice Cloning | Languages | GPU Required |
| :--- | :--- | :--- | :--- | :--- |
| **CosyVoice** | CosyVoice-300M | Yes (reference audio) | Multi-language | Optional |
| **OpenVoice** | OpenVoice v2 + Melo TTS | Yes (tone color conversion) | 18+ languages | Optional |
| **ESpeak** | — | No | Broad | No |

**CosyVoice built-in voices**: Anby, BMO, Goku, Gute

**OpenVoice supported languages**: Arabic, German, Greek, English, Spanish, French, Hindi, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Russian, Swedish, Turkish, Ukrainian, Urdu, Chinese

---

## 📡 API Reference

All endpoints are under `/api/v1`. The app also serves interactive docs at `/docs` (Swagger) and `/redoc`.

### Health & System

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/health` | System status check |
| GET | `/system/config` | Hardware capabilities & available models |

### Authentication

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/auth/signup` | Register user (returns JWT) |
| POST | `/auth/login` | Authenticate user (returns JWT) |
| GET | `/auth/me` | Get current user info |

### Chat

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/chat` | Single-turn chat (blocking) |
| POST | `/chat/stream` | Streaming chat with progress events |
| POST | `/clear` | Clear memory for a conversation |

### Conversations

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/conversations` | List user conversations |
| POST | `/conversations` | Create new conversation |
| GET | `/conversations/{id}/messages` | Get conversation history |
| PATCH | `/conversations/{id}` | Update (title, archive, password) |
| POST | `/conversations/{id}/unlock` | Unlock archived conversation |
| DELETE | `/conversations/{id}` | Delete conversation |

### Image Generation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/generate` | Generate image (proxies to Imagine service) |
| POST | `/upscale` | Upscale image (Swin2SR or Real-ESRGAN) |

### Studio (Image Projects)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/studio/projects` | List projects |
| POST | `/studio/projects` | Create project |
| DELETE | `/studio/projects/{id}` | Delete project |
| GET | `/studio/projects/{id}/images` | List project images |
| DELETE | `/studio/projects/{id}/images/{image_id}` | Delete image |
| POST | `/studio/prompt-improve` | Improve image prompt via LLM |

<details>
<summary><strong>Coach (Language Learning) — 15+ endpoints</strong></summary>

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| POST | `/coach/sessions` | Create coaching session |
| GET | `/coach/sessions` | List sessions |
| GET | `/coach/sessions/{id}` | Get session details |
| DELETE | `/coach/sessions/{id}` | Delete session |
| PATCH | `/coach/sessions/{id}/settings` | Update session settings |
| GET | `/coach/sessions/{id}/turns` | List recorded turns |
| GET | `/coach/sessions/{id}/mistakes` | List mistakes with feedback |
| GET | `/coach/sessions/{id}/progress` | Session progress |
| POST | `/coach/sessions/{id}/end` | End & summarize session |
| GET | `/coach/progress` | Overall user progress |
| POST | `/coach/turns/stream` | Audio streaming turn (multipart) |
| POST | `/coach/turns/text` | Text-based coaching turn |
| POST | `/coach/tts` | Text-to-speech synthesis |
| GET | `/coach/tts/status` | TTS service status |
| GET | `/coach/runtime/status` | Runtime status (LLM + TTS) |
| POST | `/coach/runtime/preload` | Preload models into memory |
| GET | `/coach/languages/supported` | List supported languages |
| POST | `/coach/voices/reference` | Upload voice sample for cloning |

</details>

---

## ⚙️ Environment Variables

Configure via `.env` file or environment. Key variables:

<details>
<summary><strong>Full environment variable reference</strong></summary>

### Hardware (Auto-Detected)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HARDWARE_TARGET` | auto | `arm64` or `amd64` |
| `USE_GPU` | auto | NVIDIA GPU detected via `nvidia-smi` |
| `GPU_VRAM` | auto | Largest visible GPU VRAM in GB |

### Brain Models

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SMART_MODEL` | `qwen2.5:7b` | Tool calling, precision |
| `HEAVY_MODEL` | `qwen3:8b` | Intelligence, creativity |
| `MEDIUM_MODEL` | `llama3.2:3b` | Standard CPU fallback |
| `LIGHT_MODEL` | `llama3.2:1b` | Edge / low resource |
| `VISION_MODEL` | `moondream` | Image analysis |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Vector embeddings |

### Ollama Runtime

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_NUM_CTX` | auto (2048–32768) | Context window, sized to GPU VRAM |
| `OLLAMA_TEMPERATURE` | `0.3` | Generation temperature |
| `RECENT_HISTORY_MESSAGES` | `10` | Messages kept in full context |
| `HISTORY_SUMMARY_TRIGGER` | `14` | Message count that triggers summarization |
| `MAX_MEMORY_ITEMS` | `3` | Long-term memory results per turn |

### Routing

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ROUTER_USE_SEMANTIC_FALLBACK` | `true` | Enable semantic intent fallback |
| `ROUTER_SEMANTIC_THRESHOLD` | `0.6` | Minimum similarity for semantic match |
| `ROUTER_MARGIN_THRESHOLD` | `0.08` | Margin between top-2 intent scores |

### Coach (ASR & TTS)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `COACH_WHISPER_MODEL` | `auto` | Faster-Whisper model |
| `COACH_TTS_PROVIDER` | `espeak` | `espeak`, `cosyvoice`, or `openvoice` |
| `COACH_RUNTIME_PROFILE` | `auto` | `auto`, `cpu`, `gpu_constrained`, `gpu_full` |
| `COACH_HIGH_VRAM_THRESHOLD_GB` | `16` | VRAM threshold for `gpu_full` profile |
| `COACH_COSYVOICE_MODEL_ID` | `iic/CosyVoice-300M` | CosyVoice model |
| `COACH_OPENVOICE_MODEL_ID` | `openvoice-v2` | OpenVoice model |

### Imagine

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | — | Hugging Face token (for gated models like Flux) |
| `OFFLINE_MODE` | `true` | Use cached models only |

### Auth

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SECRET_KEY` | built-in | JWT signing key (change in production) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` | Token expiry (7 days) |

</details>

---

## 🧪 Testing

Tests live under `tests/` and run with pytest:

```bash
pytest -q
```

Test coverage includes:

| Test File | Coverage Area |
| :--- | :--- |
| `test_answer_composer.py` | Response sanitization |
| `test_brain_handoff.py` | Tool orchestration logic |
| `test_brain_image_generation_contract.py` | Image generation API contract |
| `test_brain_service_file_handoff.py` | File operation handling |
| `test_brain_service_search_contract.py` | Web search & memory retrieval |
| `test_chat_stream_api.py` | Streaming chat events |
| `test_coach_api.py` | Coach endpoint contracts |
| `test_coach_models.py` | Coach model loading |
| `test_coach_service.py` | Coach service logic |
| `test_config_runtime_defaults.py` | Hardware auto-detection & config defaults |
| `test_db_schema_patches.py` | Database schema migrations |
| `test_router_service.py` | Intent routing classification |

---

## 🤝 Contributing

- Follow **Conventional Commits**: `feat:`, `fix:`, `chore:`, `docs:`
- Use 4-space indentation, explicit type hints, and `async`/`await` for I/O paths
- Keep imports ordered: standard library, third-party, local modules
- Use structured logging via `app/core/logger.py` (`get_logger(...)`) — no `print`
- Run `pytest -q` before opening a PR
- See [AGENTS.md](AGENTS.md) for full coding conventions and PR guidelines

---

> _"Intelligence should not require a subscription."_
