import torch
import logging
import os
import sys
import glob
from huggingface_hub import snapshot_download

LOG_PATH = "/app/preload.log" if os.path.isdir("/app") else os.path.join(os.getcwd(), "preload.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("bipod.preload")

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
HF_HOME = os.environ.get("HF_HOME", "/app/models")

MODELS = [
    {"id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",   "type": "sd"},
    {"id": "stabilityai/sd-vae-ft-mse",                 "type": "vae"},
    {"id": "stabilityai/stable-diffusion-xl-base-1.0",  "type": "sdxl-base"},
    {"id": "ByteDance/SDXL-Lightning",                  "type": "sdxl-lightning-unet"},
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
    "*.safetensors",
    "unet/*.safetensors",
    "text_encoder/*.safetensors",
    "vae/*.safetensors",
    "unet/*.bin",
    "text_encoder/*.bin",
    "vae/*.bin",
]

# SDXL: safetensors only (no root blobs)
SDXL_PATTERNS = BASE_PATTERNS + [
    "unet/*.safetensors",
    "text_encoder/*.safetensors",
    "text_encoder_2/*.safetensors",
    "vae/*.safetensors",
]

# SDXL Lightning repo is an UNet checkpoint repo; runtime expects the 4-step UNet file.
SDXL_LIGHTNING_PATTERNS = [
    "sdxl_lightning_4step_unet.safetensors",
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


REQUIRED_PATTERNS = {
    "sd": [
        "model_index.json",
        "*.safetensors|unet/*.safetensors|unet/*.bin",
    ],
    "sdxl-base": [
        "model_index.json",
        "unet/*.safetensors",
        "text_encoder/*.safetensors",
        "text_encoder_2/*.safetensors",
    ],
    "sdxl-lightning-unet": [
        "sdxl_lightning_4step_unet.safetensors",
    ],
    "flux": [
        "model_index.json",
        "text_encoder/*.safetensors",
        "text_encoder_2/*.safetensors",
        "transformer/*.safetensors",
        "vae/*.safetensors",
    ],
    "vae": [
        "**/*.safetensors",
    ],
    "transformer": [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors|pytorch_model.bin",
    ],
}


def _has_any_match(snapshot_path, pattern_expr: str) -> bool:
    alternatives = [p.strip() for p in pattern_expr.split("|")]
    for pat in alternatives:
        if glob.glob(os.path.join(snapshot_path, pat), recursive=True):
            return True
    return False


def _get_snapshot_path_if_cached(repo_id: str):
    try:
        return snapshot_download(
            repo_id=repo_id,
            cache_dir=HF_HOME,
            local_files_only=True,
        )
    except Exception:
        return None


def is_cached_and_valid(model_cfg: dict) -> bool:
    repo_id = model_cfg["id"]
    m_type = model_cfg["type"]
    required = REQUIRED_PATTERNS.get(m_type, [])

    snapshot_path = _get_snapshot_path_if_cached(repo_id)
    if not snapshot_path:
        return False

    for pattern in required:
        if not _has_any_match(snapshot_path, pattern):
            logger.warning(f"Cache check failed for {repo_id}: missing '{pattern}'")
            return False
    return True


def download_snapshot(repo_id, allow_patterns=None, ignore_patterns=None):
    """
    Download a HuggingFace repo snapshot.
    allow_patterns limits which files are fetched — critical for large repos
    like Realistic_Vision that ship many variants we don't need.
    """
    snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_HOME,
        ignore_patterns=ignore_patterns or ["*.ckpt", "*.msgpack"],
        allow_patterns=allow_patterns,
    )


# ---------------------------------------------------------------------------
# Per-type handlers (simplified - no verification on limited VRAM)
# ---------------------------------------------------------------------------

def preload_sd(repo_id):
    logger.info(f"[sd] Downloading (optimized SD weights): {repo_id}")
    download_snapshot(
        repo_id,
        allow_patterns=SD_PATTERNS,
        ignore_patterns=["*.ckpt", "*.msgpack"],
    )
    logger.info(f"[sd] ✓ Downloaded: {repo_id}")


def preload_sdxl(repo_id):
    logger.info(f"[sdxl] Downloading (safetensors only): {repo_id}")
    download_snapshot(
        repo_id,
        allow_patterns=SDXL_PATTERNS,
        ignore_patterns=["*.ckpt", "*.msgpack", "*.bin"],
    )
    logger.info(f"[sdxl] ✓ Downloaded: {repo_id}")


def preload_flux(repo_id):
    logger.info(f"[flux] Downloading Flux.1-schnell (~34GB bf16): {repo_id}")
    logger.info("[flux] Downloading transformer + T5-XXL + VAE + CLIP only...")
    download_snapshot(
        repo_id,
        allow_patterns=FLUX_PATTERNS,
        ignore_patterns=["*.ckpt", "*.msgpack", "*.bin"],
    )
    logger.info(f"[flux] ✓ Downloaded: {repo_id}")
    logger.info("[flux] ✓ Downloaded and ready for strict cache validation.")


def preload_sdxl_lightning_unet(repo_id):
    logger.info(f"[sdxl-lightning] Downloading UNet checkpoint: {repo_id}")
    download_snapshot(
        repo_id,
        allow_patterns=SDXL_LIGHTNING_PATTERNS,
        ignore_patterns=["*.ckpt", "*.msgpack", "*.bin"],
    )
    logger.info(f"[sdxl-lightning] ✓ Downloaded: {repo_id}")


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
    "sd":                    preload_sd,
    "sdxl-base":             preload_sdxl,
    "sdxl-lightning-unet":   preload_sdxl_lightning_unet,
    "flux":                  preload_flux,
    "vae":                   preload_vae,
    "transformer":           preload_transformer,
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
    logger.info("  SDXL Base 1.0           ~6.5 GB  (safetensors only)")
    logger.info("  SDXL Lightning UNet     ~0.2 GB  (4-step checkpoint)")
    logger.info("  sdxl-vae-fp16-fix       ~0.2 GB")
    logger.info("  Tiny-SD                 ~0.4 GB")
    logger.info("  Swin2SR upscaler        ~0.1 GB")
    logger.info("  Flux.1-schnell          ~34  GB  (bf16 safetensors only)")
    logger.info("  ─────────────────────────────────────────────────────")
    logger.info("  Total                   ~43.7 GB")
    logger.info("")
    logger.info("NOTE: Strict cache verification is enabled.")
    logger.info("      Each repo is validated for required runtime files.")
    logger.info("=" * 60)

    total   = len(MODELS)
    success = 0
    skipped = 0
    failed  = []

    for idx, model in enumerate(MODELS, start=1):
        repo_id = model["id"]
        m_type  = model["type"]

        logger.info(f"--- [{idx}/{total}] {repo_id} ({m_type}) ---")

        if is_cached_and_valid(model):
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
                if is_cached_and_valid(model):
                    success += 1
                else:
                    logger.error(f"Post-download validation failed for {repo_id}")
                    failed.append(repo_id)

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
