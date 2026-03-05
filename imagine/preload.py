import torch
import logging
import os
import sys
from huggingface_hub import snapshot_download, try_to_load_from_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/preload.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("bipod.preload")

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
HF_HOME = os.environ.get("HF_HOME", "/app/models")

MODELS = [
    {"id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",  "type": "sd"},
    {"id": "stabilityai/sd-vae-ft-mse",                 "type": "vae"},
    {"id": "ByteDance/SDXL-Lightning",                  "type": "sdxl"},
    {"id": "madebyollin/sdxl-vae-fp16-fix",             "type": "vae"},
    {"id": "segmind/tiny-sd",                           "type": "sd"},
    {"id": "caidas/swin2SR-classical-sr-x2-64",         "type": "transformer"},
    {"id": "black-forest-labs/FLUX.1-schnell",          "type": "flux"},
]

# ---------------------------------------------------------------------------
# File patterns per model type
#
# These allow_patterns prevent downloading unnecessary variants (inpainting,
# safety checker, non-fp16 weights, etc.) that bloat download size massively.
# ---------------------------------------------------------------------------

# Shared config/tokenizer files needed by all pipeline types
BASE_PATTERNS = [
    "*.json",
    "*.txt",
    "tokenizer/*",
    "tokenizer_2/*",
    "scheduler/*",
    "feature_extractor/*",
]

# SD 1.5: safetensors only (no root blobs, no inpainting, no .bin files)
SD_PATTERNS = BASE_PATTERNS + [
    "unet/*.safetensors",
    "text_encoder/*.safetensors",
    "vae/*.safetensors",
]

# SDXL: safetensors only (no root blobs)
SDXL_PATTERNS = BASE_PATTERNS + [
    "unet/*.safetensors",
    "text_encoder/*.safetensors",
    "text_encoder_2/*.safetensors",
    "vae/*.safetensors",
]

# Flux: bf16 safetensors (no root blobs)
# T5-XXL and transformer are the large files (~24GB combined)
FLUX_PATTERNS = BASE_PATTERNS + [
    "text_encoder/*.safetensors",    # CLIP-L (~235MB)
    "text_encoder_2/*.safetensors",  # T5-XXL (~9.5GB)
    "transformer/*.safetensors",     # DiT (~24GB)
    "vae/*.safetensors",             # VAE (~160MB)
]

# VAE and transformer repos are small — download everything
VAE_PATTERNS = None
TRANSFORMER_PATTERNS = None


def get_total_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def is_cached(repo_id):
    result = try_to_load_from_cache(repo_id, "config.json", cache_dir=HF_HOME)
    return result is not None


def download_snapshot(repo_id, allow_patterns=None):
    """
    Download a HuggingFace repo snapshot.
    allow_patterns limits which files are fetched — critical for large repos
    like Realistic_Vision that ship many variants we don't need.
    """
    snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_HOME,
        ignore_patterns=["*.ckpt", "*.msgpack", "*.bin"],  # always skip legacy formats
        allow_patterns=allow_patterns,
    )


# ---------------------------------------------------------------------------
# Per-type handlers (simplified - no verification on limited VRAM)
# ---------------------------------------------------------------------------

def preload_sd(repo_id):
    logger.info(f"[sd] Downloading (safetensors only): {repo_id}")
    download_snapshot(repo_id, allow_patterns=SD_PATTERNS)
    logger.info(f"[sd] ✓ Downloaded: {repo_id}")


def preload_sdxl(repo_id):
    logger.info(f"[sdxl] Downloading (safetensors only): {repo_id}")
    download_snapshot(repo_id, allow_patterns=SDXL_PATTERNS)
    logger.info(f"[sdxl] ✓ Downloaded: {repo_id}")


def preload_flux(repo_id):
    logger.info(f"[flux] Downloading Flux.1-schnell (~34GB bf16): {repo_id}")
    logger.info("[flux] Downloading transformer + T5-XXL + VAE + CLIP only...")
    download_snapshot(repo_id, allow_patterns=FLUX_PATTERNS)
    logger.info(f"[flux] ✓ Downloaded: {repo_id}")
    logger.info("[flux] Note: Verification skipped (requires >5.5GB VRAM). Weights cached for runtime.")


def preload_vae(repo_id):
    logger.info(f"[vae] Downloading: {repo_id}")
    download_snapshot(repo_id, allow_patterns=VAE_PATTERNS)
    logger.info(f"[vae] ✓ Downloaded: {repo_id}")


def preload_transformer(repo_id):
    logger.info(f"[transformer] Downloading: {repo_id}")
    download_snapshot(repo_id, allow_patterns=TRANSFORMER_PATTERNS)
    logger.info(f"[transformer] ✓ Downloaded: {repo_id}")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

PRELOAD_HANDLERS = {
    "sd":          preload_sd,
    "sdxl":        preload_sdxl,
    "flux":        preload_flux,
    "vae":         preload_vae,
    "transformer": preload_transformer,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preload():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting preload | device={device} | HF_HOME={HF_HOME}")

    if device == "cuda":
        vram = get_total_vram_gb()
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f}GB VRAM)")

    logger.info("=" * 60)
    logger.info("Model download sizes (with safetensors filtering):")
    logger.info("  Realistic Vision V6     ~2.0 GB  (safetensors only)")
    logger.info("  sd-vae-ft-mse           ~0.3 GB")
    logger.info("  SDXL-Lightning          ~6.5 GB  (safetensors only)")
    logger.info("  sdxl-vae-fp16-fix       ~0.2 GB")
    logger.info("  Tiny-SD                 ~0.4 GB")
    logger.info("  Swin2SR upscaler        ~0.1 GB")
    logger.info("  Flux.1-schnell          ~34  GB  (bf16 safetensors only)")
    logger.info("  ─────────────────────────────────────────────────────")
    logger.info("  Total                   ~43.5 GB")
    logger.info("")
    logger.info("NOTE: Verification skipped to prevent downloading")
    logger.info("      unwanted files. Runtime loading will validate.")
    logger.info("=" * 60)

    total   = len(MODELS)
    success = 0
    skipped = 0
    failed  = []

    for idx, model in enumerate(MODELS, start=1):
        repo_id = model["id"]
        m_type  = model["type"]

        logger.info(f"--- [{idx}/{total}] {repo_id} ({m_type}) ---")

        if is_cached(repo_id):
            logger.info(f"Already cached, skipping: {repo_id}")
            skipped += 1
            continue

        try:
            handler = PRELOAD_HANDLERS.get(m_type)
            if handler is None:
                logger.error(f"Unknown model type '{m_type}' for {repo_id}")
                failed.append(repo_id)
            else:
                handler(repo_id)
                success += 1

        except Exception as e:
            logger.error(f"CRITICAL failure for {repo_id}: {e}", exc_info=True)
            failed.append(repo_id)

    logger.info("=" * 60)
    logger.info(
        f"Preload complete: {success} downloaded, "
        f"{skipped} skipped, {len(failed)} failed"
    )
    if failed:
        logger.error(f"Failed: {failed}")
        logger.error("Re-run to retry failed downloads.")
    else:
        logger.info("All models ready. Bipod is ready to imagine!")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        preload()
    except Exception as e:
        logger.error(f"Global preload failure: {e}", exc_info=True)
        sys.exit(1)