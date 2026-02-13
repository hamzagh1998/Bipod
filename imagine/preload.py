import torch
import logging
import os
import sys
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    FluxPipeline,
)
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

FLUX_MIN_VRAM_GB = 5.5

# ---------------------------------------------------------------------------
# Model Stack (~43.5 GB)
#
#   Realistic Vision V6     ~2.0 GB   portraits / photorealism (SD 1.5)
#   sd-vae-ft-mse           ~0.3 GB   VAE for SD 1.5
#   SDXL-Lightning           ~6.5 GB   fast general generation
#   sdxl-vae-fp16-fix       ~0.2 GB   VAE for SDXL
#   Tiny-SD                 ~0.4 GB   low-resource fallback
#   Swin2SR upscaler        ~0.1 GB   2x upscale post-processing
#   Flux.1-schnell           ~34  GB   best photorealism + complex prompts
#
# Removed vs previous stack:
#   SDXL-Turbo    (-6.5 GB) — Lightning covers fast generation
#   SDXL-Base     (-6.5 GB) — Flux covers quality ceiling
#   SVD-XT        (-9.0 GB) — video not needed for Option B
# ---------------------------------------------------------------------------

MODELS = [
    # ------------------------------------------------------------------
    # SD 1.5 — portraits and photorealism at lower VRAM cost
    # ------------------------------------------------------------------
    {"id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",   "type": "sd"},
    {"id": "stabilityai/sd-vae-ft-mse",                  "type": "vae"},

    # ------------------------------------------------------------------
    # SDXL — fast general generation
    # Only Lightning kept — Turbo and Base dropped (redundant given Flux)
    # ------------------------------------------------------------------
    {"id": "ByteDance/SDXL-Lightning",                   "type": "sdxl"},
    {"id": "madebyollin/sdxl-vae-fp16-fix",              "type": "vae"},

    # ------------------------------------------------------------------
    # Lightweight fallback
    # ------------------------------------------------------------------
    {"id": "segmind/tiny-sd",                            "type": "sd"},

    # ------------------------------------------------------------------
    # Upscaler
    # ------------------------------------------------------------------
    {"id": "caidas/swin2SR-classical-sr-x2-64",          "type": "transformer"},

    # ------------------------------------------------------------------
    # Flux.1-schnell — photorealism quality ceiling
    #
    # ~34GB on disk (bf16 DiT + T5-XXL).
    # Runs in ~5.5GB VRAM at runtime via NF4 4-bit T5 quantization.
    # Requires: pip install bitsandbytes
    # Requires: HuggingFace token with accepted Flux license
    # ------------------------------------------------------------------
    {"id": "black-forest-labs/FLUX.1-schnell",           "type": "flux"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_total_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def is_cached(repo_id: str) -> bool:
    result = try_to_load_from_cache(repo_id, "config.json", cache_dir=HF_HOME)
    return result is not None


def cleanup(pipe=None):
    if pipe is not None:
        del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def download_snapshot(repo_id: str):
    snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_HOME,
        ignore_patterns=["*.ckpt", "*.msgpack"],
    )


# ---------------------------------------------------------------------------
# Per-type handlers
# ---------------------------------------------------------------------------

def preload_snapshot_only(repo_id: str, m_type: str):
    logger.info(f"[{m_type}] Downloading: {repo_id}")
    download_snapshot(repo_id)
    logger.info(f"[{m_type}] ✓ Done: {repo_id}")


def preload_sd(repo_id: str, device: str):
    logger.info(f"[sd] Downloading: {repo_id}")
    download_snapshot(repo_id)

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"[sd] Verifying: {repo_id}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            cache_dir=HF_HOME,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        logger.warning(f"[sd] Retrying with float32 after: {e}")
        pipe = StableDiffusionPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.float32,
            cache_dir=HF_HOME,
            low_cpu_mem_usage=True,
        )
    cleanup(pipe)
    logger.info(f"[sd] ✓ Verified: {repo_id}")


def preload_sdxl(repo_id: str, device: str):
    logger.info(f"[sdxl] Downloading: {repo_id}")
    download_snapshot(repo_id)

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
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
        logger.warning(f"[sdxl] Retrying without variant after: {e}")
        pipe = AutoPipelineForText2Image.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            variant=None,
            cache_dir=HF_HOME,
            low_cpu_mem_usage=True,
        )
    cleanup(pipe)
    logger.info(f"[sdxl] ✓ Verified: {repo_id}")


def preload_flux(repo_id: str, device: str):
    """
    Download Flux.1-schnell and verify the pipeline loads cleanly.

    Disk:    ~34 GB (bf16 DiT ~24GB + T5-XXL ~10GB)
    Runtime: ~5.5 GB VRAM (NF4 4-bit T5 + bf16 DiT + sequential offload)

    Verification is skipped on CPU or cards below FLUX_MIN_VRAM_GB —
    the snapshot is still downloaded and ready for runtime use.
    """
    logger.info(f"[flux] Downloading Flux.1-schnell (~34GB): {repo_id}")
    logger.info("[flux] This will take a while on first run...")
    download_snapshot(repo_id)
    logger.info(f"[flux] Snapshot complete: {repo_id}")

    total_vram = get_total_vram_gb()

    if device != "cuda" or total_vram < FLUX_MIN_VRAM_GB:
        logger.warning(
            f"[flux] Skipping verification "
            f"(device={device}, vram={total_vram:.1f}GB). "
            "Weights are on disk — runtime will load them."
        )
        return

    use_4bit_t5 = total_vram < 10.0
    logger.info(
        f"[flux] Verifying | T5: {'NF4 4-bit' if use_4bit_t5 else 'bf16'} | "
        f"VRAM: {total_vram:.1f}GB"
    )

    try:
        if use_4bit_t5:
            try:
                from transformers import T5EncoderModel, BitsAndBytesConfig
            except ImportError:
                logger.warning(
                    "[flux] bitsandbytes not installed — skipping verification. "
                    "Install with: pip install bitsandbytes"
                )
                return

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                repo_id,
                subfolder="text_encoder_2",
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                cache_dir=HF_HOME,
                low_cpu_mem_usage=True,
            )
            pipe = FluxPipeline.from_pretrained(
                repo_id,
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.bfloat16,
                cache_dir=HF_HOME,
                low_cpu_mem_usage=True,
            )
        else:
            pipe = FluxPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.bfloat16,
                cache_dir=HF_HOME,
                low_cpu_mem_usage=True,
            )

        cleanup(pipe)
        logger.info(f"[flux] ✓ Verified: {repo_id}")

    except Exception as e:
        logger.warning(
            f"[flux] Verification failed: {e}. "
            "Snapshot is intact — runtime loading may still succeed."
        )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

PRELOAD_HANDLERS = {
    "sd":          preload_sd,
    "sdxl":        preload_sdxl,
    "flux":        preload_flux,
    "vae":         None,
    "transformer": None,
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

    # Print expected download summary upfront
    logger.info("=" * 60)
    logger.info("Option B stack — expected download sizes:")
    logger.info("  Realistic Vision V6     ~2.0 GB")
    logger.info("  sd-vae-ft-mse           ~0.3 GB")
    logger.info("  SDXL-Lightning          ~6.5 GB")
    logger.info("  sdxl-vae-fp16-fix       ~0.2 GB")
    logger.info("  Tiny-SD                 ~0.4 GB")
    logger.info("  Swin2SR upscaler        ~0.1 GB")
    logger.info("  Flux.1-schnell          ~34  GB")
    logger.info("  ──────────────────────────────")
    logger.info("  Total                   ~43.5 GB")
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
                preload_snapshot_only(repo_id, m_type)
            else:
                handler(repo_id, device)
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
        logger.error("Re-run this script to retry failed downloads.")
    else:
        logger.info("All models ready. Bipod is ready to imagine!")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        preload()
    except Exception as e:
        logger.error(f"Global preload failure: {e}", exc_info=True)
        sys.exit(1)