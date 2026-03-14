# AGENTS.md — Bipod Development Guide

> "Intelligence should not require a subscription."

This file provides context for AI agents operating in this repository.

---

## 1. Project Overview

**Bipod** is a self-sovereign AI companion—a local-first, hardware-agnostic system that runs entirely on the user's machine. It separates inference (Ollama) from logic (FastAPI) using a sidecar pattern.

### Key Directives
- **Locality is Law**: Data never leaves the host unless explicitly requested
- **Hardware Agnostic**: Must scale from high-end GPUs to Raspberry Pi 5
- **True Agency**: Not a chatbot—an entity that interacts with files, cameras, microphones

### Technology Stack
- **Backend**: FastAPI, Python 3.13+, SQLAlchemy (SQLite), Pydantic
- **AI/LLM**: Ollama (local inference), LangChain, Faster-Whisper
- **Storage**: SQLite (memory), FAISS (vector embeddings)
- **Image Gen**: Stable Diffusion, Flux.1-schnell
- **Container**: Docker + Docker Compose

---

## 2. Build, Run & Test Commands

### Development (Local)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI app with auto-reload
uvicorn app.main:app --host 0.0.0.0 --port 4444 --reload

# Verify Docker build for all architectures
docker compose build --no-cache
```

### Docker Deployment
```bash
# Start all services (Ollama, Imagine, Bipod app)
docker compose up -d

# View logs
docker compose logs -f

# Pull required AI models into Ollama
docker exec -it bipod_ollama ollama pull qwen2.5:7b
docker exec -it bipod_ollama ollama pull llama3.1:8b
docker exec -it bipod_ollama ollama pull llama3.2:3b
docker exec -it bipod_ollama ollama pull moondream
docker exec -it bipod_ollama ollama pull nomic-embed-text

# Preload image generation models (requires HF token)
docker exec -it bipod_imagine python preload.py
```

### Testing
- **No pytest framework configured yet** — tests should be added in a `tests/` directory
- Manual testing via: `curl http://localhost:4444/api/v1/health`

---

## 3. Code Style Guidelines

### General Principles
- **Async-First**: Use `async`/`await` for all I/O-bound operations (HTTP calls, file I/O, DB)
- **Strict Type Hints**: Always use type annotations—see `app/core/config.py` and `app/db/models.py`
- **No Suppression**: Never use `as any`, `@ts-ignore`, or bare `except:`

### Imports (Order)
```python
# 1. Standard library
import os
import asyncio
from typing import List, Optional

# 2. Third-party packages
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import Column

# 3. Local application imports
from app.core.config import settings
from app.core.logger import get_logger
from app.services.brain_service import brain_service
```

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Files | snake_case | `brain_service.py`, `config.py` |
| Classes | PascalCase | `BrainService`, `Settings` |
| Functions/Variables | snake_case | `get_logger()`, `active_model` |
| Constants | SCREAMING_SNAKE_CASE | `OLLAMA_BASE_URL`, `ACTIVE_MODEL` |
| DB Tables | snake_case (plural) | `users`, `conversations`, `messages` |

### Error Handling
```python
# DO: Log and handle gracefully
try:
    result = await client.post(url, json=payload)
    result.raise_for_status()
except httpx.HTTPStatusError as e:
    logger.error(f"Brain failure (HTTP): {e}")
    return f"Error: {e.response.status_code}"
except Exception as e:
    logger.error(f"Brain failure: {e}")
    return f"My thoughts are fragmented: {str(e)}"

# DON'T: Empty catch or suppression
try:
    await risky_thing()
except:
    pass  # NEVER
```

### Logging
- Use the custom logger from `app/core/logger.py`:
```python
from app.core.logger import get_logger
logger = get_logger("bipod.module_name")
```
- Log levels: `logger.info()` for flow, `logger.warning()` for degraded behavior, `logger.error()` for failures

### Pydantic Models
- Use `BaseModel` for request/response schemas
- Use `from_attributes = True` for ORM compatibility
```python
class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True
```

### Database Models (SQLAlchemy)
- Use modern SQLAlchemy 2.0 style with `Mapped[]` and `mapped_column`:
```python
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True)
```

---

## 4. Project Structure

```
Bipod/
├── app/                      # Main FastAPI application
│   ├── api/
│   │   ├── __init__.py       # API routes
│   │   └── schemas.py        # Pydantic models
│   ├── core/
│   │   ├── config.py         # Settings (Pydantic)
│   │   └── logger.py         # Colored logging
│   ├── db/
│   │   ├── database.py       # SQLAlchemy setup
│   │   └── models.py         # ORM models
│   ├── services/
│   │   ├── brain_service.py  # LLM orchestration
│   │   ├── memory_service.py # SQLite operations
│   │   ├── vector_service.py # FAISS embeddings
│   │   ├── file_service.py   # File operations
│   │   ├── auth_service.py   # JWT auth
│   │   └── ...
│   └── main.py               # FastAPI entry point
├── frontend/                 # Static HTML/CSS/JS
│   ├── index.html
│   ├── studio.html
│   ├── style.css
│   └── js/
├── imagine/                  # Image generation service
│   ├── main.py
│   └── preload.py
├── docker/
│   ├── Dockerfile.app
│   └── Dockerfile.imagine
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

---

## 5. Hardware Awareness

The app auto-detects hardware at startup. See `app/core/config.py`:

| Variable | Detection | Behavior |
|----------|-----------|----------|
| `USE_GPU` | Runs `nvidia-smi` | Enables GPU-accelerated inference |
| `GPU_VRAM` | Queries NVIDIA driver | Selects appropriate model tier |
| `HARDWARE_TARGET` | `platform.machine()` | Switches between amd64/arm64 |
| `ACTIVE_MODEL` | Combines above | Picks: qwen2.5:7b, llama3.1:8b, llama3.2:3b, or llama3.2:1b |

---

## 6. Environment Variables

Create a `.env` file in the root:

```bash
# Required for image generation (HuggingFace)
HF_TOKEN=hf_xxxxxxxxxxxx

# Optional
OFFLINE_MODE=true
PYTHON_JIT=on
```

---

## 7. Port Reference

| Service | Internal Port | External (Docker) |
|---------|--------------|-------------------|
| Bipod App | 4444 | localhost:4444 |
| Ollama | 11434 | localhost:11434 |
| Imagine | 3333 | localhost:3333 |

---

## 8. Common Development Tasks

### Adding a New API Endpoint
1. Define schema in `app/api/schemas.py`
2. Add route in `app/api/__init__.py`
3. Use dependency injection for auth: `user_id: int = Depends(auth_service.get_current_user)`

### Adding a New Service
1. Create `app/services/new_service.py`
2. Import and instantiate in `app/api/__init__.py` or `app/main.py`
3. Use `get_logger("bipod.new_service")` for logging

### Modifying Database Schema
1. Edit `app/db/models.py`
2. Delete existing SQLite file: `rm -f data/memory/bipod_memory.db`
3. Restart app—the DB recreates automatically (development only)

---

## 9. Existing Agent Rules

This project has `.agent/rules/rules.md` with additional architecture standards. Key points:

- Containerization required for all features
- Sidecar pattern: Ollama separate from FastAPI
- Dynamic hardware detection required
- Use Python 3.14+ with JIT (`PYTHON_JIT=on`)

---

## 10. Quick Reference

```bash
# Full stack startup
docker compose up -d

# Rebuild everything
docker compose build --no-cache

# View all logs
docker compose logs -f

# Restart a single service
docker compose restart bipod-app

# Access container shell
docker exec -it bipod_brain sh
```
