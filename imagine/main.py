import os
import gc

# 1. Environment Configuration
# MUST be set before torch is imported for maximum effect
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# "expandable_segments:True" helps prevent VRAM fragmentation on 6GB cards
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import logging
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from io import BytesIO
import base64
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    StableVideoDiffusionPipeline,
    AutoencoderKL,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler
)
from diffusers.utils import export_to_video
from PIL import Image
import tempfile

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bipod.imagine")

app = FastAPI(title="Bipod Imagine Service")

# Global variables for caching models
txt2img_pipe = None
img2img_pipe = None
svd_pipe = None
current_model_id = None
current_video_model_id = None


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    image: Optional[str] = None  # Base64 encoded input image for img2img
    negative_prompt: Optional[str] = (
        "extra fingers, mutated hands, blurry, low quality, masterpiece, "
        "worst quality, (disfigured, ugly, bad anatomy, bad proportions), "
        "watermark, text, sign, profile, logo, 3d render"
    )
    steps: int = Field(default=4, ge=1, le=100)
    strength: float = Field(default=0.6, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    model_type: str = "stable-diffusion-xl"

    @validator("image")
    def validate_image_size(cls, v):
        if v is None:
            return v
        # Estimate decoded size (base64 is ~33% larger than raw)
        estimated_size = len(v) * 0.75
        max_size = 18 * 1024 * 1024  # 18MB limit
        if estimated_size > max_size:
            raise ValueError("Image too large. Max 18MB")
        return v


class UpscaleRequest(BaseModel):
    image: str  # Base64 encoded input image

    @validator("image")
    def validate_image_size(cls, v):
        if not v:
            raise ValueError("Image is required")
        if len(v) * 0.75 > 10 * 1024 * 1024:
            raise ValueError("Image too large for upscaling (Max 10MB)")
        return v


class VideoRequest(BaseModel):
    image: str  # Base64 encoded conditioning image
    motion_bucket_id: int = Field(default=127, ge=1, le=255)
    noise_aug_strength: float = Field(default=0.02, ge=0.0, le=1.0)
    fps: int = Field(default=7, ge=1, le=30)
    decode_chunk_size: Optional[int] = Field(default=None, ge=1, le=25)
    num_frames: Optional[int] = Field(default=None, ge=1, le=25)
    num_inference_steps: int = Field(default=25, ge=1, le=50)
    output_format: str = Field(default="mp4")

    @validator("image")
    def validate_image(cls, v):
        if not v:
            raise ValueError("Conditioning image is required")
        if len(v) * 0.75 > 20 * 1024 * 1024:
            raise ValueError("Image too large. Max 20MB")
        return v

    @validator("output_format")
    def validate_format(cls, v):
        if v not in ("mp4", "gif", "frames"):
            raise ValueError("output_format must be 'mp4', 'gif', or 'frames'")
        return v


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_HOME = os.environ.get("HF_HOME", "/app/models")
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "true").lower() == "true"
SVD_REPO = "stabilityai/stable-video-diffusion-img2vid-xt"

# Model type → repo ID mapping (single source of truth)
MODEL_REPO_MAP = {
    "dalle-mini": "segmind/tiny-sd",
    "stable-diffusion": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
    "sdxl-base": "stabilityai/stable-diffusion-xl-base-1.0",
    # Fixed: was "ByteDance/SDXL-Lightning-4step-base" which does not exist
    "stable-diffusion-xl": "ByteDance/SDXL-Lightning",
}


# ---------------------------------------------------------------------------
# Device & VRAM Utilities
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Detect available device with preference: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_vram_info() -> Optional[dict]:
    """Get current VRAM usage statistics"""
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)

    return {
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - reserved, 2),
    }


def get_vram_tier() -> Optional[str]:
    """
    Classify GPU VRAM into tiers for optimization strategy.

    Tiers:
      low      < 5 GB   (4GB cards — CPU offload everything)
      mid_low  5–8 GB   (6GB cards — sequential offload, aggressive res cap)
      medium   8–12 GB  (8–10GB cards — model offload, moderate res)
      high     12–18 GB (12–16GB cards — keep on GPU, high res)
      ultra    > 18 GB  (24GB+ — full speed, max res)
    Returns None when no CUDA device is present.
    """
    if not torch.cuda.is_available():
        return None

    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    if total_vram < 5:
        return "low"
    elif total_vram < 8:
        return "mid_low"   # 6GB cards — most common consumer tier
    elif total_vram < 12:
        return "medium"
    elif total_vram < 18:
        return "high"
    else:
        return "ultra"


def get_optimal_resolution(vram_tier: Optional[str], is_xl: bool) -> tuple[int, int]:
    """
    Return (max_resolution, recommended_resolution) for a given VRAM tier.

    For low/mid_low tiers the cap is enforced at recommended_resolution, not
    max_resolution, because max_resolution can still OOM on those cards.
    """
    if vram_tier is None:  # CPU
        return (512, 512)

    if is_xl:
        return {
            "low":     (512,  512),   # bare minimum to avoid OOM
            "mid_low": (768,  768),   # 6GB sweet spot for SDXL
            "medium":  (1024, 1024),
            "high":    (1536, 1024),
            "ultra":   (2048, 1536),
        }.get(vram_tier, (1024, 1024))
    else:
        # SD 1.5 / Tiny-SD
        return {
            "low":     (512, 512),
            "mid_low": (512, 512),
            "medium":  (768, 512),
            "high":    (1024, 768),
            "ultra":   (1536, 1024),
        }.get(vram_tier, (512, 512))


# ---------------------------------------------------------------------------
# SVD-XT Tier Config
# ---------------------------------------------------------------------------

SVD_TIER_CONFIG = {
    "low": {
        "resolution": (512, 288),
        "num_frames": 14,
        "decode_chunk_size": 2,
        "offload": "sequential",
        "note": "4GB card — reduced resolution, 14 frames, aggressive offload",
    },
    "mid_low": {
        "resolution": (576, 320),
        "num_frames": 14,
        "decode_chunk_size": 4,
        "offload": "sequential",
        "note": "6GB card — reduced resolution, 14 frames, sequential offload",
    },
    "medium": {
        "resolution": (768, 432),
        "num_frames": 25,
        "decode_chunk_size": 8,
        "offload": "model",
        "note": "8-10GB card — medium resolution, full 25 frames, model offload",
    },
    "high": {
        "resolution": (1024, 576),
        "num_frames": 25,
        "decode_chunk_size": 14,
        "offload": "model",
        "note": "12-16GB card — native resolution, full 25 frames, model offload",
    },
    "ultra": {
        "resolution": (1024, 576),
        "num_frames": 25,
        "decode_chunk_size": 25,
        "offload": "none",
        "note": "24GB+ card — native resolution, all frames in one pass, no offload",
    },
}

SVD_CPU_CONFIG = {
    "resolution": (256, 144),
    "num_frames": 8,
    "decode_chunk_size": 1,
    "offload": "none",
    "note": "CPU fallback — minimal resolution and frames",
}


def get_tier_config(vram_tier: Optional[str]) -> dict:
    if vram_tier is None:
        return SVD_CPU_CONFIG
    return SVD_TIER_CONFIG.get(vram_tier, SVD_TIER_CONFIG["medium"])


def align_dimension(dim: int, alignment: int = 8) -> int:
    """Align dimension to nearest multiple (SDXL requires divisibility by 8)"""
    return (dim // alignment) * alignment


def aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def should_use_sequential_offload(vram_tier: Optional[str]) -> bool:
    """
    Determine if sequential CPU offload should be used.
    Sequential offload adds ~15-20% latency but saves ~2-3GB VRAM.
    Used for low and mid_low tiers (< 8GB cards).
    """
    if vram_tier is None:
        return False
    return vram_tier in ("low", "mid_low")


# ---------------------------------------------------------------------------
# Pipeline Loading
# ---------------------------------------------------------------------------

def _apply_optimizations(pipe, vram_tier: Optional[str], pipe_label: str = ""):
    """
    Apply memory optimizations to a single pipeline IN THE CORRECT ORDER.

    Critical ordering rules:
      1. VAE tiling/slicing MUST come before CPU offload calls
      2. xFormers and sequential_cpu_offload are INCOMPATIBLE — pick one
      3. For shared-component pipes (from_pipe), only apply to the source pipe
    """
    use_sequential = should_use_sequential_offload(vram_tier)
    label = f"[{pipe_label}] " if pipe_label else ""

    # Step 1: VAE optimizations — MUST be before any offload call
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    logger.info(f"{label}✓ VAE tiling and slicing enabled")

    # Step 2: Attention optimization — only when NOT using sequential offload
    # (xFormers is incompatible with sequential offload; the two conflict)
    if not use_sequential:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info(f"{label}✓ xFormers memory efficient attention enabled")
        except Exception as e:
            logger.warning(f"{label}xFormers unavailable ({e}), falling back to attention slicing")
            pipe.enable_attention_slicing(slice_size="auto")
            logger.info(f"{label}✓ Attention slicing enabled")
    else:
        # Attention slicing is safe alongside sequential offload
        pipe.enable_attention_slicing(slice_size="auto")
        logger.info(f"{label}✓ Attention slicing enabled (sequential offload mode)")

    # Step 3: CPU offload strategy — MUST be the LAST optimization applied
    if use_sequential:
        # Low/Mid_low VRAM (<8GB): sequential offload, aggressive memory saving
        pipe.enable_sequential_cpu_offload()
        logger.info(f"{label}✓ Sequential CPU offload enabled (memory-optimized mode)")
    else:
        # High/Medium/Ultra VRAM: model offload keeps GPU utilization high
        pipe.enable_model_cpu_offload()
        logger.info(f"{label}✓ Model CPU offload enabled (speed-optimized mode)")


def load_pipelines(model_type: str):
    global txt2img_pipe, img2img_pipe, current_model_id

    repo_id = MODEL_REPO_MAP.get(model_type, MODEL_REPO_MAP["stable-diffusion-xl"])

    # Return cached pipelines if model hasn't changed
    if txt2img_pipe is not None and current_model_id == repo_id:
        logger.info(f"Reusing cached model: {current_model_id}")
        return txt2img_pipe, img2img_pipe

    # Unload existing model before loading a new one
    if txt2img_pipe is not None:
        logger.info(f"Unloading model '{current_model_id}' to free VRAM...")
        del txt2img_pipe
        del img2img_pipe
        txt2img_pipe = None
        img2img_pipe = None
        aggressive_cleanup()

    device = get_device()
    vram_tier = get_vram_tier()
    logger.info(
        f"Loading {repo_id} on {device} "
        f"(VRAM Tier: {vram_tier}, Offline: {OFFLINE_MODE})..."
    )

    if device == "cuda":
        logger.info(f"VRAM Status: {get_vram_info()}")

    try:
        common_args = {
            "cache_dir": HF_HOME,
            "local_files_only": OFFLINE_MODE,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }

        is_xl = "xl" in repo_id.lower() or "sdxl" in repo_id.lower() or "lightning" in repo_id.lower()

        if device == "cuda":
            # --- Step 1: Load VAE separately --------------------------------
            vae_id = (
                "madebyollin/sdxl-vae-fp16-fix"
                if is_xl
                else "stabilityai/sd-vae-ft-mse"
            )
            logger.info(f"Loading VAE: {vae_id}")
            vae = AutoencoderKL.from_pretrained(
                vae_id,
                torch_dtype=torch.float16,
                **common_args,
            )

            # --- Step 2: Load Text2Img pipeline -----------------------------
            logger.info("Loading Text2Img pipeline...")
            # tiny-sd does not ship fp16 variants on HF
            use_fp16_variant = is_xl and "tiny" not in repo_id.lower()
            txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                repo_id,
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16" if use_fp16_variant else None,
                **common_args,
            )

            # --- Step 3: Configure scheduler --------------------------------
            # Must happen BEFORE from_pipe() so the scheduler is inherited
            if "lightning" in repo_id.lower() or "turbo" in repo_id.lower():
                logger.info("Configuring Euler scheduler for Lightning/Turbo...")
                txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
                    txt2img_pipe.scheduler.config,
                    timestep_spacing="trailing",
                )
            elif "tiny" in repo_id.lower():
                logger.info("Configuring DPM++ scheduler for Tiny-SD...")
                txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    txt2img_pipe.scheduler.config,
                    use_karras_sigmas=True,
                )

            # --- Step 4: Apply optimizations to txt2img ONLY ----------------
            # img2img is created via from_pipe() which shares the same underlying
            # UNet/VAE/encoder objects. Applying optimizations to both pipes
            # double-applies them to the same objects → undefined behavior.
            _apply_optimizations(txt2img_pipe, vram_tier, pipe_label="txt2img")

            # --- Step 5: Create img2img AFTER optimizations are locked in ---
            logger.info("Creating Img2Img pipeline from Text2Img...")
            img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)
            # DO NOT call any optimization methods on img2img_pipe —
            # it shares components with txt2img_pipe and inherits everything.

            aggressive_cleanup()
            logger.info(f"Model loaded. VRAM after optimizations: {get_vram_info()}")

        else:
            # --- CPU / MPS fallback -----------------------------------------
            logger.info("Loading for CPU/MPS...")
            txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                repo_id, **common_args
            )

            if "lightning" in repo_id.lower() or "turbo" in repo_id.lower():
                txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
                    txt2img_pipe.scheduler.config,
                    timestep_spacing="trailing",
                )
            elif "tiny" in repo_id.lower():
                txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    txt2img_pipe.scheduler.config,
                    use_karras_sigmas=True,
                )

            img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)
            txt2img_pipe.to(device)
            img2img_pipe.to(device)

        current_model_id = repo_id
        logger.info(f"✓ Model '{repo_id}' loaded successfully!")
        return txt2img_pipe, img2img_pipe

    except Exception as e:
        logger.error(f"Failed to load model {repo_id}: {e}", exc_info=True)
        error_msg = str(e)
        if OFFLINE_MODE and (
            "Local" in error_msg or "not found" in error_msg.lower()
        ):
            error_msg = (
                f"Model '{repo_id}' not found in local cache. "
                "Please run 'docker exec -it bipod_imagine python preload.py' first."
            )
        raise HTTPException(status_code=500, detail=error_msg)


# ---------------------------------------------------------------------------
# SVD Pipeline Loading
# ---------------------------------------------------------------------------

def load_svd_pipeline() -> StableVideoDiffusionPipeline:
    """Load SVD-XT with optimizations matched to detected VRAM tier."""
    global svd_pipe, current_video_model_id

    if svd_pipe is not None and current_video_model_id == SVD_REPO:
        logger.info("Reusing cached SVD-XT pipeline")
        return svd_pipe

    if svd_pipe is not None:
        logger.info("Unloading existing pipeline...")
        del svd_pipe
        svd_pipe = None
        aggressive_cleanup()

    device = get_device()
    vram_tier = get_vram_tier()
    tier_cfg = get_tier_config(vram_tier)

    logger.info(
        f"Loading SVD-XT on {device} | "
        f"VRAM tier: {vram_tier} | "
        f"Config: {tier_cfg['note']}"
    )

    try:
        common_args = {
            "cache_dir": HF_HOME,
            "local_files_only": OFFLINE_MODE,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda":
            svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
                SVD_REPO,
                torch_dtype=torch.float16,
                variant="fp16",
                **common_args,
            )
            _apply_svd_optimizations(svd_pipe, vram_tier, tier_cfg)

        else:
            # CPU/MPS: load in float32
            svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
                SVD_REPO,
                **common_args,
            )
            svd_pipe.to(device)

        current_video_model_id = SVD_REPO
        aggressive_cleanup()

        logger.info("✓ SVD-XT pipeline ready")
        return svd_pipe

    except Exception as e:
        logger.error(f"Failed to load SVD-XT: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _apply_svd_optimizations(
    pipe: StableVideoDiffusionPipeline,
    vram_tier: Optional[str],
    tier_cfg: dict,
):
    offload_strategy = tier_cfg["offload"]

    # Step 1: VAE optimizations
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    logger.info("✓ VAE tiling and slicing enabled")

    # Step 2: Attention optimization
    if offload_strategy == "none":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("✓ xFormers enabled")
        except:
            pipe.enable_attention_slicing(slice_size="auto")
    else:
        pipe.enable_attention_slicing(slice_size="auto")
        logger.info("✓ Attention slicing enabled")

    # Step 3: CPU offload
    if offload_strategy == "sequential":
        pipe.enable_sequential_cpu_offload()
        logger.info("✓ Sequential CPU offload enabled")
    elif offload_strategy == "model":
        pipe.enable_model_cpu_offload()
        logger.info("✓ Model CPU offload enabled")
    else:
        pipe.to("cuda")
        logger.info("✓ Full GPU mode")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    aggressive_cleanup()

    device = get_device()
    vram_tier = get_vram_tier()

    if device == "cuda":
        logger.info(f"Pre-generation VRAM: {get_vram_info()}")

    try:
        t2i, i2i = load_pipelines(req.model_type)

        # Derive flags from current_model_id (authoritative after load)
        loaded_id = current_model_id or ""
        is_xl = "xl" in loaded_id.lower() or "sdxl" in loaded_id.lower() or "lightning" in loaded_id.lower()
        is_lightning = "lightning" in loaded_id.lower()
        is_turbo = "turbo" in loaded_id.lower()
        is_tiny = "tiny" in loaded_id.lower()

        # --- Step & guidance defaults per model type -----------------------
        # Auto-configure for distilled/fast models regardless of user input
        if is_lightning:
            steps = 4
            guidance = 0.0
            logger.info("Lightning model: forcing 4 steps, CFG=0")
        elif is_turbo:
            steps = 1
            guidance = 0.0
            logger.info("Turbo model: forcing 1 step, CFG=0")
        elif req.steps == 4:
            # User left the default — pick sensible values for standard models
            if is_tiny:
                steps = 20
                guidance = 7.5
            else:
                steps = 30
                guidance = 5.0
            logger.info(f"Standard model: auto-adjusted to {steps} steps, CFG={guidance}")
        else:
            # User explicitly set steps — honour them, but enforce a minimum
            steps = max(req.steps, 10) if not (is_lightning or is_turbo) else req.steps
            guidance = req.guidance_scale

        # --- Resolution capping --------------------------------------------
        width = req.width
        height = req.height

        if device == "cuda":
            max_res, rec_res = get_optimal_resolution(vram_tier, is_xl)

            # For low/mid_low tiers enforce the recommended (safe) cap, not max,
            # because max can still OOM on SDXL with 6GB cards.
            effective_cap = rec_res if vram_tier in ("low", "mid_low") else max_res

            if width > effective_cap or height > effective_cap:
                logger.warning(
                    f"Resolution {width}x{height} exceeds safe cap "
                    f"({effective_cap}x{effective_cap}) for VRAM tier '{vram_tier}'. "
                    f"Capping automatically."
                )
                width = min(width, effective_cap)
                height = min(height, effective_cap)

        # Align to multiple of 8 (SDXL requirement)
        width = align_dimension(width, 8)
        height = align_dimension(height, 8)

        if width != req.width or height != req.height:
            logger.info(f"Dimension adjustment: {req.width}x{req.height} → {width}x{height}")

        safe_prompt = req.prompt[:50].replace("\n", " ").replace("\r", " ")
        logger.info(
            f"Generating: '{safe_prompt}...' | "
            f"Model: {loaded_id} | Steps: {steps} | "
            f"Size: {width}x{height} | Img2Img: {bool(req.image)}"
        )

        if req.image:
            img_bytes = base64.b64decode(req.image)
            init_image = Image.open(BytesIO(img_bytes)).convert("RGB")
            init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)

            output = i2i(
                prompt=req.prompt,
                image=init_image,
                strength=req.strength,
                negative_prompt=req.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
        else:
            output = t2i(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
            )

        image = output.images[0]

        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        aggressive_cleanup()

        if device == "cuda":
            logger.info(f"Post-generation VRAM: {get_vram_info()}")

        return {
            "status": "success",
            "image_base64": img_str,
            "model_used": loaded_id,
            "actual_size": f"{width}x{height}",
            "steps_used": steps,
            "vram_tier": vram_tier,
        }

    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM Error — try lower resolution or a lighter model")
        aggressive_cleanup()
        raise HTTPException(
            status_code=507,
            detail=(
                "GPU out of memory. Try: "
                "(1) Lower resolution, "
                "(2) Use 'dalle-mini' model, or "
                "(3) Restart the service"
            ),
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        aggressive_cleanup()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upscale")
async def upscale_image(req: UpscaleRequest):
    """Upscale image 2x using Swin2SR"""
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    global txt2img_pipe, img2img_pipe

    aggressive_cleanup()
    logger.info("Starting Upscale request...")

    try:
        if torch.cuda.is_available():
            vram = get_vram_info()
            if vram["free_gb"] < 1.0:
                if txt2img_pipe is not None:
                    logger.warning(
                        "VRAM tight — unloading generation pipeline for upscale..."
                    )
                    del txt2img_pipe
                    del img2img_pipe
                    txt2img_pipe = None
                    img2img_pipe = None
                    aggressive_cleanup()

        model_id = "caidas/swin2SR-classical-sr-x2-64"
        device = get_device()

        processor = Swin2SRImageProcessor.from_pretrained(
            model_id, cache_dir=HF_HOME
        )
        model = Swin2SRForImageSuperResolution.from_pretrained(
            model_id, cache_dir=HF_HOME
        )
        model.to(device)

        img_bytes = base64.b64decode(req.image)
        init_image = Image.open(BytesIO(img_bytes)).convert("RGB")

        inputs = processor(init_image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        output = (
            outputs.reconstruction.data.squeeze()
            .float()
            .cpu()
            .clamp_(0, 1)
            .numpy()
        )
        output = np.moveaxis(output, 0, -1)
        output = (output * 255.0).round().astype(np.uint8)
        output_image = Image.fromarray(output)

        buffered = BytesIO()
        output_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        del model
        del processor
        aggressive_cleanup()

        return {
            "status": "success",
            "image_base64": img_str,
            "method": "Swin2SR x2",
        }

    except Exception as e:
        logger.error(f"Upscale failed: {e}", exc_info=True)
        aggressive_cleanup()
        raise HTTPException(status_code=500, detail=f"Upscale failed: {str(e)}")


@app.post("/generate-video")
async def generate_video(req: VideoRequest):
    aggressive_cleanup()
    device = get_device()
    vram_tier = get_vram_tier()
    tier_cfg = get_tier_config(vram_tier)

    try:
        pipe = load_svd_pipeline()
        max_w, max_h = tier_cfg["resolution"]
        auto_chunk = tier_cfg["decode_chunk_size"]
        decode_chunk_size = min(req.decode_chunk_size, auto_chunk) if req.decode_chunk_size else auto_chunk
        auto_frames = tier_cfg["num_frames"]
        num_frames = min(req.num_frames, auto_frames) if req.num_frames else auto_frames

        img_bytes = base64.b64decode(req.image)
        init_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        init_image = init_image.resize((max_w, max_h), Image.Resampling.LANCZOS)

        with torch.inference_mode():
            output = pipe(
                image=init_image,
                num_frames=num_frames,
                decode_chunk_size=decode_chunk_size,
                motion_bucket_id=req.motion_bucket_id,
                noise_aug_strength=req.noise_aug_strength,
                num_inference_steps=req.num_inference_steps,
                generator=torch.manual_seed(42) if device == "cpu" else None,
            )

        frames = output.frames[0]

        if req.output_format == "frames":
            encoded_frames = []
            for frame in frames:
                buf = BytesIO()
                frame.save(buf, format="JPEG", quality=92)
                encoded_frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            return {
                "status": "success",
                "format": "frames",
                "frames": encoded_frames,
                "frame_count": len(encoded_frames),
                "resolution": f"{max_w}x{max_h}",
                "fps": req.fps,
                "vram_tier": vram_tier,
            }

        else:
            suffix = ".mp4" if req.output_format == "mp4" else ".gif"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name

            if req.output_format == "mp4":
                export_to_video(frames, tmp_path, fps=req.fps)
            else:
                frames[0].save(
                    tmp_path,
                    save_all=True,
                    append_images=frames[1:],
                    loop=0,
                    duration=int(1000 / req.fps),
                    optimize=True,
                )

            with open(tmp_path, "rb") as f:
                video_bytes = f.read()
            os.unlink(tmp_path)
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")

            aggressive_cleanup()
            return {
                "status": "success",
                "format": req.output_format,
                "video_base64": video_b64,
                "frame_count": len(frames),
                "resolution": f"{max_w}x{max_h}",
                "vram_tier": vram_tier,
            }

    except torch.cuda.OutOfMemoryError:
        aggressive_cleanup()
        raise HTTPException(status_code=507, detail="GPU out of memory during video generation.")
    except Exception as e:
        logger.error(f"Video generation failed: {e}", exc_info=True)
        aggressive_cleanup()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/capabilities")
def capabilities():
    device = get_device()
    vram_tier = get_vram_tier()
    vram_info = get_vram_info()
    tier_cfg = get_tier_config(vram_tier)
    max_w, max_h = tier_cfg["resolution"]

    return {
        "device": device,
        "vram_tier": vram_tier,
        "vram_total_gb": vram_info["total_gb"] if vram_info else None,
        "model": SVD_REPO,
        "capabilities": {
            "max_resolution": f"{max_w}x{max_h}",
            "max_frames": tier_cfg["num_frames"],
            "decode_chunk_size": tier_cfg["decode_chunk_size"],
            "offload_strategy": tier_cfg["offload"],
            "note": tier_cfg["note"],
        },
        "all_tiers": SVD_TIER_CONFIG
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": get_device(),
        "loaded_model": current_model_id,
        "loaded_video_model": current_video_model_id,
        "vram": get_vram_info(),
        "vram_tier": get_vram_tier(),
    }


@app.get("/models")
def list_models():
    """List available model types with dynamic specs based on current hardware"""
    device = get_device()
    vram_tier = get_vram_tier()
    vram_info = get_vram_info()

    models = [
        {
            "id": "sdxl-turbo",
            "name": "SDXL Turbo (1-Step)",
            "repo": "stabilityai/sdxl-turbo",
            "speed": "instant",
        },
        {
            "id": "stable-diffusion-xl",
            "name": "SDXL Lightning (4-Step)",
            "repo": "ByteDance/SDXL-Lightning", 
            "speed": "fast",
        },
        {
            "id": "sdxl-base",
            "name": "SDXL Base 1.0 (High Quality)",
            "repo": "stabilityai/stable-diffusion-xl-base-1.0",
            "speed": "slow",
        },
        {
            "id": "stable-diffusion",
            "name": "Realistic Vision V6",
            "repo": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
            "speed": "medium",
        },
        {
            "id": "dalle-mini",
            "name": "Tiny-SD (Lightweight)",
            "repo": "segmind/tiny-sd",
            "speed": "very fast",
        },
    ]

    for model in models:
        is_xl = "sdxl" in model["id"] or model["id"] == "stable-diffusion-xl"
        max_res, rec_res = get_optimal_resolution(vram_tier, is_xl)

        if device == "cuda" and vram_info:
            if is_xl:
                vram_usage = {
                    "low": "~4-5GB",
                    "mid_low": "~5-6GB",
                    "medium": "~5-7GB",
                    "high": "~6-8GB",
                    "ultra": "~6-8GB",
                }.get(vram_tier, "~6-8GB")
            elif model["id"] == "stable-diffusion":
                vram_usage = "~3-4GB"
            else:
                vram_usage = "~2-3GB"

            model["vram_usage"] = vram_usage
            model["max_resolution"] = f"{max_res}x{max_res}"
            model["recommended_resolution"] = f"{rec_res}x{rec_res}"
        else:
            model["vram_usage"] = "N/A (CPU)"
            model["max_resolution"] = "512x512"
            model["recommended_resolution"] = "512x512"

    return {
        "device": device,
        "vram_tier": vram_tier,
        "total_vram": vram_info["total_gb"] if vram_info else None,
        "models": models,
    }


@app.get("/system")
def system_info():
    """Get detailed system information"""
    device = get_device()
    vram_tier = get_vram_tier()
    vram_info = get_vram_info()

    info = {
        "device": device,
        "vram_tier": vram_tier,
        "vram": vram_info,
    }

    if device == "cuda":
        use_seq = should_use_sequential_offload(vram_tier)
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["optimization_strategy"] = {
            "sequential_offload": use_seq,
            "reason": (
                "Memory-optimized / sequential offload (<8GB VRAM)"
                if use_seq
                else "Speed-optimized / model offload (≥8GB VRAM)"
            ),
        }

    return info