import torch
import logging
import os
import sys
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    StableVideoDiffusionPipeline,
)
from huggingface_hub import snapshot_download, try_to_load_from_cache

# Configure logging to both file and stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/preload.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("bipod.preload")

# Ensure environment is consistent
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
HF_HOME = os.environ.get("HF_HOME", "/app/models")

# ---------------------------------------------------------------------------
# Model Registry
#
# type field controls how a model is downloaded and verified:
#
#   sd          — SD 1.5 pipeline, loaded via StableDiffusionPipeline
#   sdxl        — SDXL pipeline, loaded via AutoPipelineForText2Image
#   svd         — Stable Video Diffusion, loaded via StableVideoDiffusionPipeline
#   vae         — Standalone VAE, snapshot only (no pipeline load needed)
#   transformer — Standalone transformer model, snapshot only
# ---------------------------------------------------------------------------

MODELS = [
    # ------------------------------------------------------------------
    # Image Models (SD 1.5)
    # ------------------------------------------------------------------
    {"id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",       "type": "sd"},
    {"id": "stabilityai/sd-vae-ft-mse",                      "type": "vae"},

    # ------------------------------------------------------------------
    # Image Models (SDXL Suite)
    # ------------------------------------------------------------------
    {"id": "ByteDance/SDXL-Lightning",                       "type": "sdxl"},
    {"id": "stabilityai/sdxl-turbo",                         "type": "sdxl"},
    {"id": "stabilityai/stable-diffusion-xl-base-1.0",       "type": "sdxl"},
    {"id": "madebyollin/sdxl-vae-fp16-fix",                  "type": "vae"},

    # ------------------------------------------------------------------
    # Image Models (Lightweight)
    # ------------------------------------------------------------------
    {"id": "segmind/tiny-sd",                                "type": "sd"},

    # ------------------------------------------------------------------
    # Image Upscaler
    # ------------------------------------------------------------------
    {"id": "caidas/swin2SR-classical-sr-x2-64",              "type": "transformer"},

    # ------------------------------------------------------------------
    # Video Models (SVD-XT)
    #
    # stabilityai/stable-video-diffusion-img2vid-xt
    #   The main SVD-XT pipeline (~9GB fp16 weights).
    #   Ships fp16 variant natively.
    #
    # No separate VAE needed — SVD-XT bundles its own temporal VAE.
    # No separate CLIP needed — image encoder is bundled in the pipeline.
    # ------------------------------------------------------------------
    {
        "id": "stabilityai/stable-video-diffusion-img2vid-xt",
        "type": "svd",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_cached(repo_id: str) -> bool:
    """
    Return True if config.json for this repo is already in the local cache.
    Used to skip re-downloading models on repeated preload runs.
    """
    result = try_to_load_from_cache(repo_id, "config.json", cache_dir=HF_HOME)
    return result is not None


def cleanup(pipe=None):
    """Delete a pipeline object and flush VRAM."""
    if pipe is not None:
        del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Per-type download + verification logic
# ---------------------------------------------------------------------------

def preload_snapshot_only(repo_id: str, m_type: str):
    """
    Download the full repo snapshot without loading into a pipeline.
    Used for VAE and transformer models that are loaded directly by
    their own from_pretrained calls in the service (not via a pipeline).
    """
    logger.info(f"[{m_type}] Downloading snapshot: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_HOME,
        ignore_patterns=["*.ckpt", "*.msgpack"],
    )
    logger.info(f"[{m_type}] ✓ Downloaded: {repo_id}")


def preload_sd(repo_id: str, device: str):
    """Download and verify an SD 1.5 pipeline."""
    logger.info(f"[sd] Downloading: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_HOME,
        ignore_patterns=["*.ckpt", "*.msgpack"],
    )

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"[sd] Verifying: {repo_id}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            variant=None,
            cache_dir=HF_HOME,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        logger.warning(f"[sd] Load failed ({e}), retrying with float32...")
        pipe = StableDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float32,
            variant=None,
            cache_dir=HF_HOME,
            low_cpu_mem_usage=True,
        )

    cleanup(pipe)
    logger.info(f"[sd] ✓ Verified: {repo_id}")


def preload_sdxl(repo_id: str, device: str):
    """Download and verify an SDXL pipeline."""
    logger.info(f"[sdxl] Downloading: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_HOME,
        ignore_patterns=["*.ckpt", "*.msgpack"],
    )

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    # tiny-sd does not ship fp16 variants; SDXL models generally do
    use_variant = device == "cuda" and "tiny" not in repo_id.lower()

    logger.info(f"[sdxl] Verifying: {repo_id}")
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            variant="fp16" if use_variant else None,
            cache_dir=HF_HOME,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        logger.warning(f"[sdxl] Variant load failed ({e}), retrying without variant...")
        pipe = AutoPipelineForText2Image.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            variant=None,
            cache_dir=HF_HOME,
            low_cpu_mem_usage=True,
        )

    cleanup(pipe)
    logger.info(f"[sdxl] ✓ Verified: {repo_id}")


def preload_svd(repo_id: str, device: str):
    """
    Download and verify the SVD-XT pipeline.

    SVD-XT specific notes:
      - Ships fp16 variant natively → always prefer it on CUDA
      - Weights are large (~9GB) — snapshot_download fetches in chunks
        and resumes automatically if interrupted
      - We verify with low_cpu_mem_usage=True and do NOT call .to(device)
        during preload to avoid OOM on low-VRAM machines
      - The pipeline bundles its own temporal VAE and image encoder —
        no separate downloads are needed
      - ignore_patterns excludes legacy .ckpt files (not used by diffusers)
    """
    logger.info(f"[svd] Downloading SVD-XT weights (~9GB fp16): {repo_id}")
    logger.info("[svd] This may take several minutes on first run...")

    snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_HOME,
        ignore_patterns=["*.ckpt", "*.msgpack"],
    )
    logger.info(f"[svd] Snapshot downloaded: {repo_id}")

    # Verify the pipeline loads cleanly from the local snapshot
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    use_variant = device == "cuda"  # SVD-XT ships fp16 variant natively

    logger.info(f"[svd] Verifying pipeline load: {repo_id}")
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            variant="fp16" if use_variant else None,
            cache_dir=HF_HOME,
            low_cpu_mem_usage=True,
            # Do NOT call .to(device) here — we just verify the weights
            # load correctly without risking OOM on small GPUs during preload
        )
    except Exception as e:
        logger.warning(f"[svd] fp16 variant load failed ({e}), retrying without variant...")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            variant=None,
            cache_dir=HF_HOME,
            low_cpu_mem_usage=True,
        )

    cleanup(pipe)
    logger.info(f"[svd] ✓ Verified: {repo_id}")


# ---------------------------------------------------------------------------
# Handler dispatch table
# ---------------------------------------------------------------------------

PRELOAD_HANDLERS = {
    "sd":          preload_sd,
    "sdxl":        preload_sdxl,
    "svd":         preload_svd,
    "vae":         None,   # snapshot only
    "transformer": None,   # snapshot only
}


# ---------------------------------------------------------------------------
# Main preload loop
# ---------------------------------------------------------------------------

def preload():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting preload on device: {device}")
    logger.info(f"HF_HOME: {HF_HOME}")

    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")

    total = len(MODELS)
    success = 0
    skipped = 0
    failed = []

    for idx, model in enumerate(MODELS, start=1):
        repo_id = model["id"]
        m_type = model["type"]

        logger.info(f"--- [{idx}/{total}] {repo_id} ({m_type}) ---")

        # Skip if already fully cached
        if is_cached(repo_id):
            logger.info(f"Already cached, skipping: {repo_id}")
            skipped += 1
            continue

        try:
            handler = PRELOAD_HANDLERS.get(m_type)

            if handler is None:
                # snapshot-only types (vae, transformer)
                preload_snapshot_only(repo_id, m_type)
            else:
                handler(repo_id, device)

            success += 1

        except Exception as e:
            logger.error(
                f"CRITICAL failure for {repo_id}: {e}",
                exc_info=True,
            )
            failed.append(repo_id)
            # Continue — don't let one failure block remaining models

    # Final summary
    logger.info("=" * 60)
    logger.info(
        f"Preload complete: {success} downloaded, "
        f"{skipped} skipped, {len(failed)} failed"
    )
    if failed:
        logger.error(f"Failed models: {failed}")
        logger.error("Re-run this script to retry failed downloads.")
    else:
        logger.info("All models ready. Bipod is ready to imagine and animate!")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        preload()
    except Exception as top_e:
        logger.error(f"Global preload failure: {top_e}", exc_info=True)
        sys.exit(1)