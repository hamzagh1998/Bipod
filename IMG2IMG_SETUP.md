# Image-to-Image Generation Setup 🖼️➡️🎨

You can now upload an image and ask Bipod to modifying it (e.g., _"Make this look like a painting"_).

**How it works:**

1. Upload: Attach an image to the chat.
2. Prompt: Ask Bipod to transform it.
3. Process: Bipod checks the image, saves it temporarily to `data/uploads/`, and sends it to the Imagine service which switches to `Img2Img` mode.

**Action Required:**
Because I updated the `imagine` service code (internal dependencies), you **MUST rebuild** the container:

```bash
docker compose up -d --build
```

## NEW: Imagine Model Selector ⚙️

You can now choose your preferred model in the **Brain Settings** (click the **+ icon** next to the chat input).

- Stable Diffusion: Best quality (Requires NVIDIA GPU).
- TinySD: Fast and lightweight (Optimized for CPU/Edge devices).

Bipod automatically detects your hardware; if no GPU is found, the Stable Diffusion option will be disabled to prevent crashes.

Enjoy your locally generated masterpieces! 🎨✨
