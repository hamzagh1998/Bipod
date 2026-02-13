import os
import gc

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
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
    AutoencoderKL,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    FluxPipeline,
)
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bipod.imagine")

app = FastAPI(title="Bipod Imagine Service")

# ---------------------------------------------------------------------------
# Global Pipeline State
# ---------------------------------------------------------------------------

txt2img_pipe  = None
img2img_pipe  = None
current_model_id = None

# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    negative_prompt: Optional[str] = (
        "extra fingers, mutated hands, blurry, low quality, masterpiece, "
        "worst quality, (disfigured, ugly, bad anatomy, bad proportions), "
        "watermark, text, sign, profile, logo, 3d render"
    )
    steps: int           = Field(default=4,    ge=1,   le=100)
    strength: float      = Field(default=0.6,  ge=0.0, le=1.0)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    width: int           = Field(default=1024, ge=64,  le=2048)
    height: int          = Field(default=1024, ge=64,  le=2048)
    model_type: str      = "sdxl-lightning"

    @validator("image")
    def validate_image_size(cls, v):
        if v is None:
            return v
        if len(v) * 0.75 > 18 * 1024 * 1024:
            raise ValueError("Image too large. Max 18MB")
        return v


class UpscaleRequest(BaseModel):
    image: str

    @validator("image")
    def validate_image_size(cls, v):
        if not v:
            raise ValueError("Image is required")
        if len(v) * 0.75 > 10 * 1024 * 1024:
            raise ValueError("Image too large for upscaling. Max 10MB")
        return v


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_HOME      = os.environ.get("HF_HOME", "/app/models")
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "true").lower() == "true"

# Option B — single source of truth for model_type → repo_id
# Removed: sdxl-turbo, sdxl-base (redundant given Lightning + Flux)
MODEL_REPO_MAP = {
    "dalle-mini":       "segmind/tiny-sd",
    "stable-diffusion": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "sdxl-lightning":   "ByteDance/SDXL-Lightning",
    "flux-schnell":     "black-forest-labs/FLUX.1-schnell",
}

FLUX_MIN_VRAM_GB = 5.5


# ---------------------------------------------------------------------------
# Device & VRAM Utilities
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_vram_info() -> Optional[dict]:
    if not torch.cuda.is_available():
        return None
    dev       = torch.cuda.current_device()
    total     = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    reserved  = torch.cuda.memory_reserved(dev)  / (1024 ** 3)
    return {
        "total_gb":     round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb":  round(reserved, 2),
        "free_gb":      round(total - reserved, 2),
    }


def get_vram_tier() -> Optional[str]:
    """
    low      < 5 GB   (4GB cards)
    mid_low  5–8 GB   (6GB cards  ← RTX 4050)
    medium   8–12 GB  (8–10GB)
    high     12–18 GB (12–16GB)
    ultra    > 18 GB  (24GB+)
    """
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total < 5:    return "low"
    elif total < 8:  return "mid_low"
    elif total < 12: return "medium"
    elif total < 18: return "high"
    else:            return "ultra"


def get_total_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def get_optimal_resolution(vram_tier: Optional[str], is_xl: bool) -> tuple[int, int]:
    """Return (max_res, recommended_res) for SDXL/SD models."""
    if vram_tier is None:
        return (512, 512)
    if is_xl:
        return {
            "low":     (512,  512),
            "mid_low": (768,  768),
            "medium":  (1024, 1024),
            "high":    (1536, 1024),
            "ultra":   (2048, 1536),
        }.get(vram_tier, (1024, 1024))
    else:
        return {
            "low":     (512, 512),
            "mid_low": (512, 512),
            "medium":  (768, 512),
            "high":    (1024, 768),
            "ultra":   (1536, 1024),
        }.get(vram_tier, (512, 512))


def get_flux_resolution(vram_tier: Optional[str]) -> tuple[int, int]:
    """Return (width, height) safe for Flux.1 on the given VRAM tier."""
    return {
        "mid_low": (768,  768),
        "medium":  (1024, 1024),
        "high":    (1024, 1024),
        "ultra":   (1360, 768),
    }.get(vram_tier or "mid_low", (768, 768))


def align_dimension(dim: int, alignment: int = 8) -> int:
    return (dim // alignment) * alignment


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def should_use_sequential_offload(vram_tier: Optional[str]) -> bool:
    if vram_tier is None:
        return False
    return vram_tier in ("low", "mid_low")


# ---------------------------------------------------------------------------
# SDXL / SD Pipeline Loader
# ---------------------------------------------------------------------------

def _apply_optimizations(pipe, vram_tier: Optional[str], label: str = ""):
    """
    Apply SDXL/SD optimizations in mandatory order:
      1. VAE tiling + slicing  (before any offload)
      2. Attention              (xFormers or slicing — not both)
      3. CPU offload            (always last)
    """
    pfx = f"[{label}] " if label else ""
    use_seq = should_use_sequential_offload(vram_tier)

    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    logger.info(f"{pfx}✓ VAE tiling + slicing enabled")

    if not use_seq:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info(f"{pfx}✓ xFormers enabled")
        except Exception as e:
            logger.warning(f"{pfx}xFormers unavailable ({e}), using attention slicing")
            pipe.enable_attention_slicing(slice_size="auto")
    else:
        pipe.enable_attention_slicing(slice_size="auto")
        logger.info(f"{pfx}✓ Attention slicing enabled")

    if use_seq:
        pipe.enable_sequential_cpu_offload()
        logger.info(f"{pfx}✓ Sequential CPU offload enabled")
    else:
        pipe.enable_model_cpu_offload()
        logger.info(f"{pfx}✓ Model CPU offload enabled")


def load_sdxl_pipeline(repo_id: str, device: str, vram_tier: Optional[str]):
    global txt2img_pipe, img2img_pipe

    common = {
        "cache_dir":        HF_HOME,
        "local_files_only": OFFLINE_MODE,
        "use_safetensors":  True,
        "low_cpu_mem_usage": True,
    }

    is_xl = "lightning" in repo_id.lower()

    if device == "cuda":
        vae_id = (
            "madebyollin/sdxl-vae-fp16-fix"
            if is_xl
            else "stabilityai/sd-vae-ft-mse"
        )
        logger.info(f"Loading VAE: {vae_id}")
        vae = AutoencoderKL.from_pretrained(
            vae_id, torch_dtype=torch.float16, **common
        )

        use_fp16_variant = is_xl and "tiny" not in repo_id.lower()
        logger.info("Loading Text2Img pipeline...")
        txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
            repo_id,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16" if use_fp16_variant else None,
            **common,
        )

        # Scheduler — before from_pipe() so img2img inherits it
        if "lightning" in repo_id.lower():
            txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
                txt2img_pipe.scheduler.config, timestep_spacing="trailing"
            )
        elif "tiny" in repo_id.lower():
            txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                txt2img_pipe.scheduler.config, use_karras_sigmas=True
            )

        # Optimizations on txt2img only — img2img shares components via from_pipe
        _apply_optimizations(txt2img_pipe, vram_tier, label="txt2img")
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)

    else:
        txt2img_pipe = AutoPipelineForText2Image.from_pretrained(repo_id, **common)
        if "lightning" in repo_id.lower():
            txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
                txt2img_pipe.scheduler.config, timestep_spacing="trailing"
            )
        elif "tiny" in repo_id.lower():
            txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                txt2img_pipe.scheduler.config, use_karras_sigmas=True
            )
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)
        txt2img_pipe.to(device)
        img2img_pipe.to(device)


# ---------------------------------------------------------------------------
# Flux.1-schnell Pipeline Loader
# ---------------------------------------------------------------------------

def load_flux_pipeline(repo_id: str, device: str, vram_tier: Optional[str]):
    """
    Load Flux.1-schnell with NF4 4-bit T5 quantization on cards < 10GB.

    Key Flux constraints enforced at call sites:
      - No negative_prompt (not supported by the model)
      - guidance_scale must be 0.0 (schnell is CFG-distilled)
      - No img2img (FluxPipeline doesn't support it yet)
      - 16-pixel dimension alignment (VAE factor is 16, not 8)
    """
    global txt2img_pipe, img2img_pipe

    if device != "cuda":
        raise HTTPException(
            status_code=400,
            detail="Flux.1 requires CUDA. CPU/MPS is not supported."
        )

    if get_total_vram_gb() < FLUX_MIN_VRAM_GB:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Flux.1 requires ≥{FLUX_MIN_VRAM_GB}GB VRAM. "
                f"Detected {get_total_vram_gb():.1f}GB. "
                "Use 'sdxl-lightning' or 'stable-diffusion' instead."
            ),
        )

    use_4bit_t5  = vram_tier in ("mid_low", "medium")
    use_seq      = vram_tier in ("mid_low",)

    common = {
        "cache_dir":        HF_HOME,
        "local_files_only": OFFLINE_MODE,
        "use_safetensors":  True,
        "low_cpu_mem_usage": True,
    }

    logger.info(
        f"[flux] Loading | T5: {'NF4 4-bit' if use_4bit_t5 else 'bf16'} | "
        f"offload: {'sequential' if use_seq else 'model'}"
    )

    if use_4bit_t5:
        try:
            from transformers import T5EncoderModel, BitsAndBytesConfig
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="bitsandbytes required for Flux on 6GB. Run: pip install bitsandbytes",
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("[flux] Loading T5-XXL in NF4 4-bit...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            repo_id,
            subfolder="text_encoder_2",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            **common,
        )
        txt2img_pipe = FluxPipeline.from_pretrained(
            repo_id,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
            **common,
        )
    else:
        txt2img_pipe = FluxPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            **common,
        )

    # VAE optimizations — before offload
    txt2img_pipe.enable_vae_tiling()
    txt2img_pipe.enable_vae_slicing()
    logger.info("[flux] ✓ VAE tiling + slicing enabled")

    # DiT does not support xFormers or attention_slicing — offload only
    if use_seq:
        txt2img_pipe.enable_sequential_cpu_offload()
        logger.info("[flux] ✓ Sequential CPU offload enabled")
    else:
        txt2img_pipe.enable_model_cpu_offload()
        logger.info("[flux] ✓ Model CPU offload enabled")

    # Flux has no img2img pipeline
    img2img_pipe = None


# ---------------------------------------------------------------------------
# Unified Loader
# ---------------------------------------------------------------------------

def load_pipelines(model_type: str):
    global txt2img_pipe, img2img_pipe, current_model_id

    repo_id = MODEL_REPO_MAP.get(model_type)
    if repo_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model_type '{model_type}'. "
                f"Valid options: {list(MODEL_REPO_MAP.keys())}"
            ),
        )

    if txt2img_pipe is not None and current_model_id == repo_id:
        logger.info(f"Reusing cached: {current_model_id}")
        return txt2img_pipe, img2img_pipe

    if txt2img_pipe is not None:
        logger.info(f"Unloading '{current_model_id}'...")
        del txt2img_pipe, img2img_pipe
        txt2img_pipe = None
        img2img_pipe = None
        aggressive_cleanup()

    device    = get_device()
    vram_tier = get_vram_tier()

    logger.info(
        f"Loading {repo_id} | device={device} | "
        f"vram_tier={vram_tier} | offline={OFFLINE_MODE}"
    )
    if device == "cuda":
        logger.info(f"VRAM before load: {get_vram_info()}")

    try:
        if model_type == "flux-schnell":
            load_flux_pipeline(repo_id, device, vram_tier)
        else:
            load_sdxl_pipeline(repo_id, device, vram_tier)

        current_model_id = repo_id
        aggressive_cleanup()

        if device == "cuda":
            logger.info(f"VRAM after load: {get_vram_info()}")

        logger.info(f"✓ Loaded: {repo_id}")
        return txt2img_pipe, img2img_pipe

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load {repo_id}: {e}", exc_info=True)
        msg = str(e)
        if OFFLINE_MODE and ("Local" in msg or "not found" in msg.lower()):
            msg = (
                f"'{repo_id}' not in local cache. "
                "Run 'docker exec -it bipod_imagine python preload.py' first."
            )
        raise HTTPException(status_code=500, detail=msg)


# ---------------------------------------------------------------------------
# Generation Helpers
# ---------------------------------------------------------------------------

def _resolve_steps_and_guidance(
    loaded_id: str, req_steps: int, req_guidance: float, is_flux: bool
) -> tuple[int, float]:
    if is_flux:
        # schnell is 4-step CFG-distilled — guidance must be 0
        return 4, 0.0
    if "lightning" in loaded_id.lower():
        return 4, 0.0
    if "tiny" in loaded_id.lower():
        return (req_steps if req_steps != 4 else 20), 7.5
    # Standard SD 1.5 (Realistic Vision)
    return (req_steps if req_steps != 4 else 30), (req_guidance or 5.0)


def _resolve_resolution(
    req_w: int, req_h: int,
    vram_tier: Optional[str],
    is_xl: bool, is_flux: bool,
    device: str,
) -> tuple[int, int]:
    w, h = req_w, req_h

    if device == "cuda":
        if is_flux:
            max_w, max_h = get_flux_resolution(vram_tier)
            if vram_tier in ("mid_low", "low"):
                w = min(w, max_w)
                h = min(h, max_h)
        else:
            max_res, rec_res = get_optimal_resolution(vram_tier, is_xl)
            cap = rec_res if vram_tier in ("low", "mid_low") else max_res
            w = min(w, cap)
            h = min(h, cap)

    # Flux needs 16-pixel alignment; everything else needs 8
    align = 16 if is_flux else 8
    return align_dimension(w, align), align_dimension(h, align)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    aggressive_cleanup()

    device    = get_device()
    vram_tier = get_vram_tier()
    is_flux   = req.model_type == "flux-schnell"

    if device == "cuda":
        logger.info(f"Pre-generation VRAM: {get_vram_info()}")

    try:
        t2i, i2i = load_pipelines(req.model_type)

        loaded_id = current_model_id or ""
        is_xl = "lightning" in loaded_id.lower() or "flux" in loaded_id.lower()

        # Flux does not support img2img
        if is_flux and req.image:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Flux.1 does not support img2img. "
                    "Remove the 'image' field or use 'sdxl-lightning'."
                ),
            )

        steps, guidance = _resolve_steps_and_guidance(
            loaded_id, req.steps, req.guidance_scale, is_flux
        )
        width, height = _resolve_resolution(
            req.width, req.height, vram_tier, is_xl, is_flux, device
        )

        if width != req.width or height != req.height:
            logger.info(f"Resolution: {req.width}x{req.height} → {width}x{height}")

        safe_prompt = req.prompt[:50].replace("\n", " ")
        logger.info(
            f"Generating | {loaded_id} | "
            f"{steps} steps | {width}x{height} | "
            f"img2img={bool(req.image)} | '{safe_prompt}...'"
        )

        if req.image:
            # img2img — SDXL / SD 1.5 only (Flux blocked above)
            img_bytes  = base64.b64decode(req.image)
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
            if is_flux:
                # Flux does not accept negative_prompt
                output = t2i(
                    prompt=req.prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
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

        buf = BytesIO()
        image.save(buf, format="JPEG", quality=95)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        aggressive_cleanup()

        if device == "cuda":
            logger.info(f"Post-generation VRAM: {get_vram_info()}")

        return {
            "status":       "success",
            "image_base64": img_str,
            "model_used":   loaded_id,
            "actual_size":  f"{width}x{height}",
            "steps_used":   steps,
            "vram_tier":    vram_tier,
        }

    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM during generation")
        aggressive_cleanup()
        raise HTTPException(
            status_code=507,
            detail=(
                "GPU out of memory. Try: "
                "(1) Lower resolution, "
                "(2) Switch to 'sdxl-lightning' or 'dalle-mini', "
                "(3) Restart the service"
            ),
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        aggressive_cleanup()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upscale")
async def upscale_image(req: UpscaleRequest):
    """2x upscale via Swin2SR."""
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    global txt2img_pipe, img2img_pipe

    aggressive_cleanup()
    logger.info("Upscale request received")

    try:
        if torch.cuda.is_available():
            vram = get_vram_info()
            if vram["free_gb"] < 1.0 and txt2img_pipe is not None:
                logger.warning("VRAM tight — unloading generation pipeline for upscale")
                del txt2img_pipe, img2img_pipe
                txt2img_pipe = None
                img2img_pipe = None
                aggressive_cleanup()

        model_id = "caidas/swin2SR-classical-sr-x2-64"
        device   = get_device()

        processor = Swin2SRImageProcessor.from_pretrained(model_id, cache_dir=HF_HOME)
        model     = Swin2SRForImageSuperResolution.from_pretrained(model_id, cache_dir=HF_HOME)
        model.to(device)

        img_bytes  = base64.b64decode(req.image)
        init_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        inputs     = processor(init_image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        output = (
            outputs.reconstruction.data.squeeze()
            .float().cpu().clamp_(0, 1).numpy()
        )
        output       = np.moveaxis(output, 0, -1)
        output       = (output * 255.0).round().astype(np.uint8)
        output_image = Image.fromarray(output)

        buf = BytesIO()
        output_image.save(buf, format="JPEG", quality=95)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        del model, processor
        aggressive_cleanup()

        return {"status": "success", "image_base64": img_str, "method": "Swin2SR x2"}

    except Exception as e:
        logger.error(f"Upscale failed: {e}", exc_info=True)
        aggressive_cleanup()
        raise HTTPException(status_code=500, detail=f"Upscale failed: {str(e)}")


# ---------------------------------------------------------------------------
# Info Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status":       "ok",
        "device":       get_device(),
        "loaded_model": current_model_id,
        "vram":         get_vram_info(),
        "vram_tier":    get_vram_tier(),
    }


@app.get("/models")
def list_models():
    device     = get_device()
    vram_tier  = get_vram_tier()
    vram_info  = get_vram_info()
    total_vram = get_total_vram_gb()

    models = [
        {
            "id":       "sdxl-lightning",
            "name":     "SDXL Lightning (4-step)",
            "repo":     "ByteDance/SDXL-Lightning",
            "speed":    "fast",
            "use_case": "General generation, fast drafts",
            "supports_img2img":      True,
            "supports_negative_prompt": True,
        },
        {
            "id":       "stable-diffusion",
            "name":     "Realistic Vision V6",
            "repo":     "SG161222/Realistic_Vision_V6.0_B1_noVAE",
            "speed":    "medium",
            "use_case": "Portrait photography",
            "supports_img2img":      True,
            "supports_negative_prompt": True,
        },
        {
            "id":       "dalle-mini",
            "name":     "Tiny-SD (Lightweight)",
            "repo":     "segmind/tiny-sd",
            "speed":    "very fast",
            "use_case": "Low-resource fallback",
            "supports_img2img":      True,
            "supports_negative_prompt": True,
        },
        {
            "id":       "flux-schnell",
            "name":     "Flux.1-schnell (4-bit, Photorealism)",
            "repo":     "black-forest-labs/FLUX.1-schnell",
            "speed":    "medium",
            "use_case": "Best photorealism, text in image, complex prompts",
            "supports_img2img":         False,
            "supports_negative_prompt": False,
            "available":           total_vram >= FLUX_MIN_VRAM_GB or device != "cuda",
            "requires_vram_gb":    FLUX_MIN_VRAM_GB,
            "quantization":        "NF4 4-bit T5 on <10GB cards",
        },
    ]

    for m in models:
        is_flux = m["id"] == "flux-schnell"
        is_xl   = m["id"] == "sdxl-lightning"

        if device == "cuda" and vram_info:
            if is_flux:
                w, h = get_flux_resolution(vram_tier)
                m["max_resolution"]         = f"{w}x{h}"
                m["recommended_resolution"] = f"{w}x{h}"
                m["vram_usage"] = {
                    "mid_low": "~5-6GB (4-bit T5)",
                    "medium":  "~7-8GB (4-bit T5)",
                    "high":    "~10-12GB (bf16)",
                    "ultra":   "~14-16GB (bf16)",
                }.get(vram_tier or "mid_low", "~6-8GB")
            else:
                max_res, rec_res = get_optimal_resolution(vram_tier, is_xl)
                m["max_resolution"]         = f"{max_res}x{max_res}"
                m["recommended_resolution"] = f"{rec_res}x{rec_res}"
                m["vram_usage"] = (
                    "~5-6GB" if is_xl
                    else "~3-4GB" if m["id"] == "stable-diffusion"
                    else "~2-3GB"
                )
        else:
            m["max_resolution"]         = "512x512"
            m["recommended_resolution"] = "512x512"
            m["vram_usage"]             = "N/A (CPU)"

    return {
        "device":     device,
        "vram_tier":  vram_tier,
        "total_vram": vram_info["total_gb"] if vram_info else None,
        "models":     models,
    }


@app.get("/system")
def system_info():
    device    = get_device()
    vram_tier = get_vram_tier()
    info = {"device": device, "vram_tier": vram_tier, "vram": get_vram_info()}

    if device == "cuda":
        use_seq = should_use_sequential_offload(vram_tier)
        info.update({
            "cuda_version":  torch.version.cuda,
            "gpu_name":      torch.cuda.get_device_name(0),
            "gpu_count":     torch.cuda.device_count(),
            "flux_available": get_total_vram_gb() >= FLUX_MIN_VRAM_GB,
            "optimization_strategy": {
                "sequential_offload": use_seq,
                "reason": (
                    "Memory-optimized / sequential offload (<8GB)"
                    if use_seq else
                    "Speed-optimized / model offload (≥8GB)"
                ),
            },
        })
    return info