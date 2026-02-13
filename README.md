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

## 🧠 Required Models

Bipod uses different models based on the selected tier and capabilities. You must pull these models into the Ollama container for them to work.

Run the following command to install the recommended brain and utility models:

```bash
# Recommended Brain (Higher intelligence & reliable tools)
docker exec -it bipod_ollama ollama pull qwen2.5:7b

# Alternative Heavy Brain (General baseline)
docker exec -it bipod_ollama ollama pull llama3.1:8b

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
   python3 -c "from huggingface_hub import login; login()"
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
