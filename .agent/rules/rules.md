---
trigger: always_on
---

## 1. Project Philosophy: "Weightless Intelligence"

The core goal of **Bipod** is to provide an AI companion that is free from the gravity of the cloud.

- **Locality is Law:** If it can be done locally, it _must_ be done locally. Data never leaves the host machine unless explicitly requested by the user.
- **Hardware Agnostic:** Bipod must run on a high-end NVIDIA workstation or a Raspberry Pi 5. It scales its "brain" size, not its architecture.
- **True Agency:** Bipod is not a chatbot; it is a system entity. It interacts with files, cameras, and microphones to understand and manipulate its environment.

---

## 2. Core Directives (Architecture Standards)

To maintain "AI" status, all development must follow these technical constraints:

### **A. Containerization & Portability**

- **Standard:** Everything must live in a Docker container.
- **Multi-Arch Support:** Dockerfiles must use multi-arch base images (e.g., `python:3.14-slim`) to support both `amd64` (PC) and `arm64` (Pi) out of the box.
- **The Sidecar Pattern:** Keep the **Inference Server** (Ollama) separate from the **Logic Server** (FastAPI). This allows for modular upgrades and independent scaling.

### **B. Hardware Awareness**

- **Dynamic Optimization:** The code must detect its environment. If a GPU is present, use it for Faster-Whisper and Ollama. If not, fallback to CPU quantization (GGUF) gracefully without crashing.
- **Resource Throttling:** On edge devices (Pi), the agent must switch to "Efficient Mode" (Moondream).

### **C. Modern Python (3.14+) Standards**

- **Async-First:** Use `asyncio` for all I/O bound tasks (Webcam, Mic, LLM calls) to prevent the "Robot Stutter."
- **JIT Enabled:** Leverage the Python 3.14 JIT compiler (`PYTHON_JIT=on`) for high-performance logic processing.
- **Typed Python:** Use strict type hinting to ensure the codebase remains maintainable as a developer-led project.

---

## 3. Best Practices for Bipod Development

| Category     | Rule                        | Reason                                                                                         |
| :----------- | :-------------------------- | :--------------------------------------------------------------------------------------------- |
| **Vision**   | **Triggered, not Streamed** | Continuous video streaming kills CPU. Bipod should only "look" when it needs to or when asked. |
| **Audio**    | **VAD Integration**         | Use Voice Activity Detection (VAD) to ignore silence and background noise.                     |
| **Memory**   | **Contextual Persistence**  | Use a local SQLite instance for long-term memory. Don't rely on volatile RAM.                  |
| **Security** | **Explicit Shell Access**   | Bipod cannot run system-altering commands without "Human-in-the-loop" confirmation.            |
| **Safety**   | **The Sandbox**             | Bipod only has access to volumes explicitly mounted in `docker-compose.yml`.                   |

---

## 4. What We Are Achieving

We are building a **Self-Sovereign Companion**.

1.  **Privacy:** Your conversations and camera feed are yours alone.
2.  **Longevity:** If the internet goes down, Bipod still wakes up.
3.  **Extensibility:** Because it's an "AI" agent, it can be moved from a desktop PC to a physical rover or a handheld device with minimal code changes.

> "Intelligence should not require a subscription."

---

### **Quick Start for Devs**

To maintain the integrity of Bipod, always run the linter and ensure the Docker build succeeds for both `linux/amd64` and `linux/arm64`.

```bash
# Check if the brain is still light
docker compose build --no-cache
```
