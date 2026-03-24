import os
import gc
import binascii
import glob
import sys
import time

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import logging
import numpy as np
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from io import BytesIO
import base64
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoencoderKL,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    FluxPipeline,
    StableDiffusionPipeline,
)
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bipod.imagine")

app = FastAPI(title="Bipod Imagine Service")

# ---------------------------------------------------------------------------
# Global Pipeline State
# ---------------------------------------------------------------------------

txt2img_pipe = None
img2img_pipe = None
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
    steps: int = Field(default=4, ge=1, le=100)
    strength: float = Field(default=0.6, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    model_type: str = "sdxl-lightning"
    output_format: Literal["jpeg", "png", "webp"] = "jpeg"

    @validator("image")
    def validate_image_size(cls, v):
        if v is None:
            return v
        if len(v) * 0.75 > 18 * 1024 * 1024:
            raise ValueError("Image too large. Max 18MB")
        return v


class UpscaleRequest(BaseModel):
    image: str
    scale: Literal[2, 4, 8] = 2
    upscaler: Literal["swin2sr", "realesrgan"] = "swin2sr"
    output_format: Literal["jpeg", "png", "webp"] = "jpeg"

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

HF_HOME = os.environ.get("HF_HOME", "/app/models")
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "true").lower() == "true"

# Option B — single source of truth for model_type → repo_id
# Removed: sdxl-turbo, sdxl-base (redundant given Lightning + Flux)
MODEL_REPO_MAP = {
    "dalle-mini": "segmind/tiny-sd",
    "stable-diffusion": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "sdxl-lightning": "ByteDance/SDXL-Lightning",
    "juggernaut-xl": "RunDiffusion/Juggernaut-XL-v9",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}
SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
SDXL_LIGHTNING_UNET_FILE = "sdxl_lightning_4step_unet.safetensors"

FLUX_MIN_VRAM_GB = 5.0
SDXL_LIGHTNING_NATIVE_EDGE = 1024
REALESRGAN_MODEL_FILENAME = "RealESRGAN_x4plus.pth"
REALESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
REALISTIC_VISION_RESOLUTION_LIMITS = {
    "low": (768, 512),
    "mid_low": (1536, 1536),
    "medium": (1280, 1024),
    "high": (1536, 1024),
    "ultra": (1536, 1024),
}
REALISTIC_VISION_DEFAULT_GUIDANCE = 6.5

OUTPUT_FORMAT_METADATA = {
    "jpeg": {"pil_format": "JPEG", "mime_type": "image/jpeg", "extension": "jpg"},
    "png": {"pil_format": "PNG", "mime_type": "image/png", "extension": "png"},
    "webp": {"pil_format": "WEBP", "mime_type": "image/webp", "extension": "webp"},
}


# ---------------------------------------------------------------------------
# Device & VRAM Utilities
# ---------------------------------------------------------------------------


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_cuda_busy_error(exc: Exception) -> bool:
    """Detect CUDA 'busy or unavailable' errors (distinct from OOM)."""
    return isinstance(exc, RuntimeError) and "busy or unavailable" in str(exc).lower()


def warmup_cuda_context():
    """Force-initialise the CUDA context so later allocations don't race
    with other containers that share the same GPU."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.init()
        # A tiny allocation is enough to pin the CUDA context in this process.
        _probe = torch.zeros(1, device="cuda")
        del _probe
        torch.cuda.empty_cache()
        logger.info("CUDA context warmed up successfully")
    except RuntimeError as e:
        logger.warning("CUDA warmup failed: %s", e)


def get_vram_info() -> Optional[dict]:
    if not torch.cuda.is_available():
        return None
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(dev) / (1024**3)
    reserved = torch.cuda.memory_reserved(dev) / (1024**3)
    return {
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - reserved, 2),
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
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total < 5:
        return "low"
    elif total < 8:
        return "mid_low"
    elif total < 12:
        return "medium"
    elif total < 18:
        return "high"
    else:
        return "ultra"


def get_total_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def get_optimal_resolution(vram_tier: Optional[str], is_xl: bool) -> tuple[int, int]:
    """Return (max_res, recommended_res) for SDXL/SD models."""
    if vram_tier is None:
        return (512, 512)
    if is_xl:
        return {
            "low": (1024, 768),
            "mid_low": (2048, 1024),
            "medium": (3072, 1536),
            "high": (4096, 2048),
            "ultra": (4096, 2048),
        }.get(vram_tier, (1024, 1024))
    else:
        return {
            "low": (768, 512),
            "mid_low": (1536, 768),
            "medium": (2048, 1024),
            "high": (3072, 1536),
            "ultra": (4096, 2048),
        }.get(vram_tier, (512, 512))


def get_flux_resolution(vram_tier: Optional[str]) -> tuple[int, int]:
    """Return (width, height) safe for Flux.1 on the given VRAM tier."""
    return {
        "mid_low": (512, 512),
        "medium": (768, 768),
        "high": (1024, 1024),
        "ultra": (1360, 768),
    }.get(vram_tier or "mid_low", (512, 512))


def get_model_resolution_limits(
    model_id: str,
    vram_tier: Optional[str],
    is_xl: bool,
) -> tuple[int, int]:
    lowered = (model_id or "").lower()
    if "lightning" in lowered:
        # SDXL-Lightning is trained for 1024px-class native generation.
        # Larger exports should come from the upscaler, not direct 2K renders.
        return (SDXL_LIGHTNING_NATIVE_EDGE, SDXL_LIGHTNING_NATIVE_EDGE)
    if "realistic_vision" in lowered or "realistic vision" in lowered:
        # Realistic Vision is an SD 1.5 checkpoint; it degrades quickly when pushed
        # far beyond ~768-1024 native resolution, especially on 6GB-class GPUs.
        return REALISTIC_VISION_RESOLUTION_LIMITS.get(
            vram_tier or "mid_low", REALISTIC_VISION_RESOLUTION_LIMITS["mid_low"]
        )
    if "flux" in lowered:
        edge = max(get_flux_resolution(vram_tier))
        return (edge, edge)
    return get_optimal_resolution(vram_tier, is_xl)


def resolve_upscale_scale(
    requested_scale: int,
    requested_device: str,
    vram_tier: Optional[str],
    upscaler: str,
) -> tuple[int, bool]:
    if requested_scale != 8:
        return requested_scale, False
    if upscaler != "realesrgan":
        return 4, True
    return 8, False


def align_dimension(dim: int, alignment: int = 8) -> int:
    return (dim // alignment) * alignment


def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_generation_pipelines():
    global txt2img_pipe, img2img_pipe, current_model_id

    if txt2img_pipe is None and img2img_pipe is None:
        return

    logger.info("Unloading generation pipelines to free VRAM...")
    txt2img_pipe = None
    img2img_pipe = None
    current_model_id = None
    aggressive_cleanup()


def is_cuda_oom_error(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def align_tile_size(size: int, step: int = 64, minimum: int = 192) -> int:
    return max(minimum, (size // step) * step)


def recommend_upscale_tile_sizes(
    device: str, vram_tier: Optional[str], image: Image.Image
) -> list[Optional[int]]:
    if device != "cuda":
        return [512] if max(image.size) > 1536 else [None]

    width, height = image.size
    longest_edge = max(width, height)
    vram_info = get_vram_info() or {}
    free_gb = float(vram_info.get("free_gb") or 0.0)
    total_gb = float(vram_info.get("total_gb") or 0.0)

    # Keep a modest safety buffer, but size tiles from actual headroom instead of
    # using only coarse VRAM tiers. This pushes the GPU harder on cards that still
    # have several GB free after unloading generation pipelines.
    headroom_gb = 0.9 if longest_edge >= 4096 else 0.7 if longest_edge >= 2048 else 0.5
    usable_gb = max(1.25, free_gb - headroom_gb)
    if total_gb > 0:
        usable_gb = min(usable_gb, total_gb * 0.9)

    tier_floor = {
        "low": 256,
        "mid_low": 384,
        "medium": 640,
        "high": 896,
        "ultra": 1024,
    }.get(vram_tier or "mid_low", 384)
    tier_ceiling = {
        "low": 448,
        "mid_low": 768,
        "medium": 1152,
        "high": 1536,
        "ultra": 2048,
    }.get(vram_tier or "mid_low", 768)

    dynamic_tile = int(448 * ((usable_gb / 2.5) ** 0.5))
    dynamic_tile = align_tile_size(dynamic_tile)
    max_effective_tile = align_tile_size(min(longest_edge, tier_ceiling), minimum=192)
    primary_tile = min(max_effective_tile, max(tier_floor, dynamic_tile))
    primary_tile = align_tile_size(primary_tile)

    candidates = []
    for delta in (0, 64, 128, 192, 256, 320):
        candidate = align_tile_size(primary_tile - delta)
        if candidate <= 0:
            continue
        if candidate not in candidates:
            candidates.append(candidate)

    logger.info(
        "Dynamic upscale tile plan | input=%sx%s | vram_tier=%s | free_gb=%.2f | usable_gb=%.2f | candidates=%s",
        width,
        height,
        vram_tier,
        free_gb,
        usable_gb,
        candidates,
    )
    return candidates


def recommend_realesrgan_tile_sizes(
    device: str, vram_tier: Optional[str], image: Image.Image
) -> list[Optional[int]]:
    base_candidates = recommend_upscale_tile_sizes(device, vram_tier, image)
    adjusted = []
    for candidate in base_candidates:
        if candidate is None:
            adjusted.append(None)
            continue
        shrink = align_tile_size(int(candidate * 0.75), minimum=128)
        if shrink not in adjusted:
            adjusted.append(shrink)
    return adjusted or [256]


def build_tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _run_swin2sr_forward(
    model, processor, tile_image: Image.Image, device: str
) -> np.ndarray:
    inputs = processor(tile_image, return_tensors="pt")
    inputs = {
        key: (
            value.to(device=device, dtype=torch.float32)
            if torch.is_floating_point(value)
            else value.to(device)
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, 0, -1)
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    return (output * 255.0).round().astype(np.uint8)


def load_upscale_model():
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    model_id = "caidas/swin2SR-classical-sr-x2-64"
    processor = Swin2SRImageProcessor.from_pretrained(
        model_id,
        cache_dir=HF_HOME,
        local_files_only=OFFLINE_MODE,
    )
    model = Swin2SRForImageSuperResolution.from_pretrained(
        model_id,
        cache_dir=HF_HOME,
        local_files_only=OFFLINE_MODE,
    )
    model.eval()
    return model, processor


def resolve_realesrgan_model_path() -> str:
    model_dir = os.path.join(HF_HOME, "realesrgan")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, REALESRGAN_MODEL_FILENAME)
    if os.path.exists(model_path):
        return model_path
    if OFFLINE_MODE:
        raise HTTPException(
            status_code=500,
            detail=(
                "Real-ESRGAN weights are not cached locally. "
                "Run 'docker exec -it bipod_imagine python preload.py' first."
            ),
        )

    from basicsr.utils.download_util import load_file_from_url

    return load_file_from_url(
        REALESRGAN_MODEL_URL,
        model_dir=model_dir,
        file_name=REALESRGAN_MODEL_FILENAME,
        progress=True,
    )


def run_realesrgan_inference(
    init_image: Image.Image,
    device: str,
    tile_size: Optional[int] = None,
    outscale: int = 4,
) -> Image.Image:
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        # basicsr/realesrgan still imports the pre-0.20 torchvision module path.
        import torchvision.transforms._functional_tensor as functional_tensor

        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    tile = tile_size or 0
    upsampler = RealESRGANer(
        scale=4,
        model_path=resolve_realesrgan_model_path(),
        model=model,
        tile=tile,
        tile_pad=max(10, (tile_size or 0) // 8) if tile_size else 10,
        pre_pad=0,
        half=device == "cuda",
        gpu_id=0 if device == "cuda" else None,
    )

    logger.info(
        "Running Real-ESRGAN upscale | device=%s | tile_size=%s | outscale=%s | input=%sx%s",
        device,
        tile_size,
        outscale,
        init_image.size[0],
        init_image.size[1],
    )

    bgr_input = np.array(init_image)[:, :, ::-1]
    output, _ = upsampler.enhance(bgr_input, outscale=outscale)
    rgb_output = output[:, :, ::-1]
    return Image.fromarray(rgb_output.astype(np.uint8))


def run_upscale_inference(
    model,
    processor,
    init_image: Image.Image,
    device: str,
    tile_size: Optional[int] = None,
):
    model = model.to(device)

    if not tile_size:
        output = _run_swin2sr_forward(model, processor, init_image, device)
        return Image.fromarray(output)

    overlap = max(32, tile_size // 8)
    stride = max(1, tile_size - overlap)
    width, height = init_image.size
    scale = 2
    x_starts = build_tile_starts(width, tile_size, stride)
    y_starts = build_tile_starts(height, tile_size, stride)
    accum = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
    weights = np.zeros((height * scale, width * scale, 1), dtype=np.float32)

    logger.info(
        "Running tiled upscale | device=%s | tile_size=%s | overlap=%s | input=%sx%s",
        device,
        tile_size,
        overlap,
        width,
        height,
    )

    for top in y_starts:
        for left in x_starts:
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            tile = init_image.crop((left, top, right, bottom))
            tile_output = _run_swin2sr_forward(model, processor, tile, device).astype(
                np.float32
            )
            expected_h = (bottom - top) * scale
            expected_w = (right - left) * scale
            tile_output = tile_output[:expected_h, :expected_w]

            out_top = top * scale
            out_left = left * scale
            dest_h = min(tile_output.shape[0], accum.shape[0] - out_top)
            dest_w = min(tile_output.shape[1], accum.shape[1] - out_left)
            if dest_h <= 0 or dest_w <= 0:
                continue

            tile_output = tile_output[:dest_h, :dest_w]
            accum[
                out_top : out_top + dest_h, out_left : out_left + dest_w
            ] += tile_output
            weights[out_top : out_top + dest_h, out_left : out_left + dest_w] += 1.0

    output = (accum / np.clip(weights, 1.0, None)).round().astype(np.uint8)
    return Image.fromarray(output)


def run_upscale_pass(
    model,
    processor,
    init_image: Image.Image,
    requested_device: str,
    vram_tier: Optional[str],
) -> tuple[Image.Image, str, Optional[int]]:
    upscale_device = choose_upscale_device(requested_device, vram_tier, init_image)
    tile_candidates = recommend_upscale_tile_sizes(
        upscale_device, vram_tier, init_image
    )
    tile_size = tile_candidates[0]

    try:
        output_image = None
        last_error = None
        if upscale_device == "cuda":
            for candidate in tile_candidates:
                tile_size = candidate
                try:
                    output_image = run_upscale_inference(
                        model,
                        processor,
                        init_image,
                        upscale_device,
                        tile_size=tile_size,
                    )
                    break
                except Exception as e:
                    last_error = e
                    if is_cuda_oom_error(e):
                        logger.warning(
                            "GPU OOM during upscale with tile_size=%s; trying a smaller tile.",
                            tile_size,
                        )
                        aggressive_cleanup()
                        continue
                    raise
            if output_image is None and last_error is not None:
                raise last_error
        else:
            output_image = run_upscale_inference(
                model,
                processor,
                init_image,
                upscale_device,
                tile_size=tile_size,
            )
    except Exception as e:
        if upscale_device == "cuda" and is_cuda_oom_error(e):
            logger.warning(
                "GPU OOM during upscale; retrying on CPU. VRAM status before fallback: %s",
                get_vram_info(),
            )
            aggressive_cleanup()
            upscale_device = "cpu"
            tile_size = recommend_upscale_tile_sizes("cpu", vram_tier, init_image)[0]
            output_image = run_upscale_inference(
                model,
                processor,
                init_image,
                "cpu",
                tile_size=tile_size,
            )
        else:
            raise

    return output_image, upscale_device, tile_size


def run_realesrgan_pass(
    init_image: Image.Image,
    requested_device: str,
    vram_tier: Optional[str],
    outscale: int,
) -> tuple[Image.Image, str, Optional[int]]:
    upscale_device = choose_upscale_device(requested_device, vram_tier, init_image)
    tile_candidates = recommend_realesrgan_tile_sizes(
        upscale_device, vram_tier, init_image
    )
    tile_size = tile_candidates[0]

    try:
        output_image = None
        last_error = None
        if upscale_device == "cuda":
            for candidate in tile_candidates:
                tile_size = candidate
                try:
                    output_image = run_realesrgan_inference(
                        init_image,
                        upscale_device,
                        tile_size=tile_size,
                        outscale=outscale,
                    )
                    break
                except Exception as e:
                    last_error = e
                    if is_cuda_oom_error(e):
                        logger.warning(
                            "GPU OOM during Real-ESRGAN upscale with tile_size=%s; trying a smaller tile.",
                            tile_size,
                        )
                        aggressive_cleanup()
                        continue
                    raise
            if output_image is None and last_error is not None:
                raise last_error
        else:
            output_image = run_realesrgan_inference(
                init_image,
                upscale_device,
                tile_size=tile_size,
                outscale=outscale,
            )
    except Exception as e:
        if upscale_device == "cuda" and is_cuda_oom_error(e):
            logger.warning(
                "GPU OOM during Real-ESRGAN upscale; retrying on CPU. VRAM status before fallback: %s",
                get_vram_info(),
            )
            aggressive_cleanup()
            upscale_device = "cpu"
            tile_size = recommend_realesrgan_tile_sizes("cpu", vram_tier, init_image)[0]
            output_image = run_realesrgan_inference(
                init_image,
                "cpu",
                tile_size=tile_size,
                outscale=outscale,
            )
        else:
            raise

    return output_image, upscale_device, tile_size


def choose_upscale_device(
    requested_device: str, vram_tier: Optional[str], image: Image.Image
) -> str:
    if requested_device != "cuda":
        return requested_device

    width, height = image.size
    pixel_count = width * height

    return "cuda"


def should_use_sequential_offload(
    vram_tier: Optional[str],
    model_type: Optional[str] = None,
) -> bool:
    if vram_tier is None:
        return False
    if vram_tier == "low":
        return True
    if vram_tier != "mid_low":
        return False

    lowered = (model_type or "").lower()
    if lowered in {"stable-diffusion", "dalle-mini"}:
        return False
    if "realistic_vision" in lowered or "realistic vision" in lowered:
        return False
    return True


def decode_base64_image(image_data: str) -> bytes:
    payload = image_data
    if image_data.startswith("data:image") and "," in image_data:
        payload = image_data.split(",", 1)[1]

    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")


def encode_image_output(image: Image.Image, output_format: str) -> tuple[str, str, str]:
    spec = OUTPUT_FORMAT_METADATA[output_format]
    save_image = image
    if spec["pil_format"] in {"JPEG", "WEBP"} and image.mode not in ("RGB", "L"):
        save_image = image.convert("RGB")

    buf = BytesIO()
    save_kwargs = {}
    if spec["pil_format"] == "JPEG":
        save_kwargs.update({"quality": 95, "optimize": True})
    elif spec["pil_format"] == "WEBP":
        save_kwargs.update({"quality": 95, "method": 6})

    save_image.save(buf, format=spec["pil_format"], **save_kwargs)
    return (
        base64.b64encode(buf.getvalue()).decode("utf-8"),
        spec["mime_type"],
        spec["extension"],
    )


def resolve_cached_snapshot_path(repo_id: str) -> str:
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=HF_HOME,
        local_files_only=True,
    )


def find_single_file_checkpoint(snapshot_path: str) -> str:
    candidates = sorted(
        path
        for path in glob.glob(os.path.join(snapshot_path, "*.safetensors"))
        if "inpainting" not in os.path.basename(path).lower()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No root safetensors checkpoint found in {snapshot_path}"
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# SDXL / SD Pipeline Loader
# ---------------------------------------------------------------------------


def _apply_optimizations(
    pipe,
    vram_tier: Optional[str],
    label: str = "",
    model_type: Optional[str] = None,
):
    """
    Apply SDXL/SD optimizations in mandatory order:
      1. VAE tiling + slicing  (before any offload)
      2. Attention              (xFormers or slicing — not both)
      3. CPU offload            (always last)
    """
    pfx = f"[{label}] " if label else ""
    use_seq = should_use_sequential_offload(vram_tier, model_type=model_type)

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

    offload_fn = (
        pipe.enable_sequential_cpu_offload if use_seq else pipe.enable_model_cpu_offload
    )
    offload_label = "Sequential" if use_seq else "Model"
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            offload_fn()
            logger.info(f"{pfx}✓ {offload_label} CPU offload enabled")
            break
        except RuntimeError as e:
            if _is_cuda_busy_error(e) and attempt < max_retries:
                wait = 2 * attempt
                logger.warning(
                    "%sCUDA busy during %s offload (attempt %d/%d); "
                    "retrying in %ds...",
                    pfx,
                    offload_label,
                    attempt,
                    max_retries,
                    wait,
                )
                aggressive_cleanup()
                warmup_cuda_context()
                time.sleep(wait)
            else:
                raise


def load_sdxl_pipeline(
    repo_id: str,
    device: str,
    vram_tier: Optional[str],
    model_type: Optional[str] = None,
):
    global txt2img_pipe, img2img_pipe

    is_lightning = model_type == "sdxl-lightning"
    is_juggernaut = model_type == "juggernaut-xl"
    is_tiny = model_type == "dalle-mini"
    is_single_file_sd = model_type == "stable-diffusion"
    base_repo_id = SDXL_BASE_REPO if is_lightning else repo_id
    use_safetensors = not is_tiny

    common = {
        "cache_dir": HF_HOME,
        "local_files_only": OFFLINE_MODE,
        "use_safetensors": use_safetensors,
        "low_cpu_mem_usage": True,
    }

    is_xl = is_lightning or is_juggernaut
    pipeline_kwargs = dict(common)

    # The filtered RV V6 cache intentionally skips the safety checker weights.
    # Only the single-file Stable Diffusion path needs this override.
    if is_single_file_sd:
        pipeline_kwargs.update(
            {
                "safety_checker": None,
                "feature_extractor": None,
                "requires_safety_checker": False,
            }
        )

    if device == "cuda":
        vae = None
        if is_xl:
            vae_id = "madebyollin/sdxl-vae-fp16-fix"
            logger.info(f"Loading VAE: {vae_id}")
            vae = AutoencoderKL.from_pretrained(
                vae_id, torch_dtype=torch.float16, **common
            )
        elif is_single_file_sd:
            vae_id = "stabilityai/sd-vae-ft-mse"
            logger.info(f"Loading VAE: {vae_id}")
            vae = AutoencoderKL.from_pretrained(
                vae_id,
                torch_dtype=torch.float16,
                cache_dir=HF_HOME,
                local_files_only=OFFLINE_MODE,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

        use_fp16_variant = is_xl and "tiny" not in base_repo_id.lower()
        logger.info("Loading Text2Img pipeline...")
        if is_single_file_sd:
            snapshot_path = resolve_cached_snapshot_path(base_repo_id)
            checkpoint_path = find_single_file_checkpoint(snapshot_path)
            logger.info(
                f"Loading single-file Stable Diffusion checkpoint: {os.path.basename(checkpoint_path)}"
            )
            txt2img_pipe = StableDiffusionPipeline.from_single_file(
                checkpoint_path,
                config=snapshot_path,
                vae=vae,
                torch_dtype=torch.float16,
                **pipeline_kwargs,
            )
        else:
            load_kwargs = dict(
                torch_dtype=torch.float16,
                variant="fp16" if use_fp16_variant else None,
                **pipeline_kwargs,
            )
            if vae is not None:
                load_kwargs["vae"] = vae
            txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                base_repo_id,
                **load_kwargs,
            )

        if is_lightning:
            logger.info("Applying SDXL-Lightning 4-step UNet weights...")
            lightning_unet_path = hf_hub_download(
                repo_id=SDXL_LIGHTNING_REPO,
                filename=SDXL_LIGHTNING_UNET_FILE,
                cache_dir=HF_HOME,
                local_files_only=OFFLINE_MODE,
            )
            state_dict = load_file(lightning_unet_path, device="cpu")
            missing, unexpected = txt2img_pipe.unet.load_state_dict(
                state_dict, strict=False
            )
            if unexpected:
                raise RuntimeError(
                    f"Unexpected SDXL-Lightning UNet keys: {unexpected[:5]}"
                )
            if missing:
                logger.warning(f"Missing SDXL-Lightning UNet keys: {len(missing)}")
            logger.info("✓ SDXL-Lightning UNet loaded")

        # Scheduler — before from_pipe() so img2img inherits it
        if is_lightning or is_juggernaut:
            txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
                txt2img_pipe.scheduler.config, timestep_spacing="trailing"
            )
        elif "tiny" in base_repo_id.lower():
            txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                txt2img_pipe.scheduler.config, use_karras_sigmas=True
            )

        # Optimizations on txt2img only — img2img shares components via from_pipe
        _apply_optimizations(
            txt2img_pipe,
            vram_tier,
            label="txt2img",
            model_type=model_type,
        )
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)

    else:
        if is_single_file_sd:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                cache_dir=HF_HOME,
                local_files_only=OFFLINE_MODE,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            snapshot_path = resolve_cached_snapshot_path(base_repo_id)
            checkpoint_path = find_single_file_checkpoint(snapshot_path)
            logger.info(
                f"Loading single-file Stable Diffusion checkpoint: {os.path.basename(checkpoint_path)}"
            )
            txt2img_pipe = StableDiffusionPipeline.from_single_file(
                checkpoint_path,
                config=snapshot_path,
                vae=vae,
                **pipeline_kwargs,
            )
        else:
            txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                base_repo_id, **pipeline_kwargs
            )

        if is_lightning or is_juggernaut:
            logger.info("Applying SDXL-Lightning 4-step UNet weights...")
            lightning_unet_path = hf_hub_download(
                repo_id=SDXL_LIGHTNING_REPO,
                filename=SDXL_LIGHTNING_UNET_FILE,
                cache_dir=HF_HOME,
                local_files_only=OFFLINE_MODE,
            )
            state_dict = load_file(lightning_unet_path, device="cpu")
            missing, unexpected = txt2img_pipe.unet.load_state_dict(
                state_dict, strict=False
            )
            if unexpected:
                raise RuntimeError(
                    f"Unexpected SDXL-Lightning UNet keys: {unexpected[:5]}"
                )
            if missing:
                logger.warning(f"Missing SDXL-Lightning UNet keys: {len(missing)}")
            logger.info("✓ SDXL-Lightning UNet loaded")

        if is_lightning or is_juggernaut:
            txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
                txt2img_pipe.scheduler.config, timestep_spacing="trailing"
            )
        elif "tiny" in base_repo_id.lower():
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
            status_code=400, detail="Flux.1 requires CUDA. CPU/MPS is not supported."
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

    use_4bit_t5 = vram_tier in ("mid_low", "medium")
    use_seq = vram_tier in ("mid_low", "medium")

    common = {
        "cache_dir": HF_HOME,
        "local_files_only": OFFLINE_MODE,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
    }

    t5_device = (
        "cpu"
        if (use_4bit_t5 and vram_tier == "mid_low")
        else ("cuda" if use_4bit_t5 else "pipeline-default")
    )
    logger.info(
        f"[flux] Loading | T5: {'NF4 4-bit' if use_4bit_t5 else 'bf16'} | "
        f"offload: {'sequential' if use_seq else 'model'} | "
        f"T5 device: {t5_device}"
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

        def _load_flux_with_4bit_t5():
            # On low-VRAM cards, load T5 to CPU first. Sequential CPU offload
            # will move individual layers to GPU one at a time during inference.
            load_device = "cpu" if vram_tier == "mid_low" else "cuda"
            logger.info(f"[flux] Loading T5-XXL in NF4 4-bit on {load_device}...")
            encoder_kwargs = {
                "torch_dtype": torch.bfloat16,
                **common,
            }
            # bitsandbytes quantization only works on CUDA tensors;
            # for CPU-first loading we skip quantization and rely on
            # sequential offload to keep only one layer on GPU at a time.
            if load_device == "cuda":
                encoder_kwargs["quantization_config"] = bnb_config
            else:
                encoder_kwargs["device_map"] = "cpu"

            text_encoder_2 = T5EncoderModel.from_pretrained(
                repo_id,
                subfolder="text_encoder_2",
                **encoder_kwargs,
            )
            return FluxPipeline.from_pretrained(
                repo_id,
                text_encoder_2=text_encoder_2,
                torch_dtype=torch.bfloat16,
                **common,
            )

        try:
            txt2img_pipe = _load_flux_with_4bit_t5()
        except torch.OutOfMemoryError:
            logger.warning("[flux] CUDA OOM while loading 4-bit T5 on CUDA")
            aggressive_cleanup()
            raise HTTPException(
                status_code=400,
                detail=(
                    "Flux.1 4-bit T5 loading ran out of VRAM on this GPU. "
                    "Use 'sdxl-lightning' or run Flux on a GPU with more VRAM."
                ),
            )
    else:
        txt2img_pipe = FluxPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            **common,
        )

    # VAE optimizations — before offload
    if hasattr(txt2img_pipe, "enable_vae_tiling"):
        txt2img_pipe.enable_vae_tiling()
    if hasattr(txt2img_pipe, "enable_vae_slicing"):
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

    device = get_device()
    vram_tier = get_vram_tier()

    logger.info(
        f"Loading {repo_id} | device={device} | "
        f"vram_tier={vram_tier} | offline={OFFLINE_MODE}"
    )
    if device == "cuda":
        warmup_cuda_context()
        logger.info(f"VRAM before load: {get_vram_info()}")

    try:
        if model_type == "flux-schnell":
            load_flux_pipeline(repo_id, device, vram_tier)
        else:
            load_sdxl_pipeline(repo_id, device, vram_tier, model_type=model_type)

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
    if "lightning" in loaded_id.lower() or "juggernaut" in loaded_id.lower():
        # Keep Lightning's CFG setting, but honor the step count sent by the UI.
        return req_steps, 0.0
    if "tiny" in loaded_id.lower():
        return (req_steps if req_steps != 4 else 20), 7.5
    # Standard SD 1.5 (Realistic Vision)
    return (
        req_steps if req_steps != 4 else 30,
        req_guidance or REALISTIC_VISION_DEFAULT_GUIDANCE,
    )


def _resolve_resolution(
    req_w: int,
    req_h: int,
    vram_tier: Optional[str],
    model_id: str,
    is_xl: bool,
    is_flux: bool,
    device: str,
) -> tuple[int, int]:
    align = 16 if is_flux else 8
    requested = (align_dimension(req_w, align), align_dimension(req_h, align))

    if device != "cuda":
        return requested

    if is_flux:
        cap = max(get_flux_resolution(vram_tier))
    else:
        max_res, rec_res = get_model_resolution_limits(model_id, vram_tier, is_xl)
        cap = rec_res if vram_tier in ("low", "mid_low") else max_res

    if max(requested) <= cap:
        return requested

    return _scale_resolution_to_longest_edge(
        requested[0],
        requested[1],
        cap,
        align,
    )


def _scale_resolution_to_longest_edge(
    req_w: int,
    req_h: int,
    longest_edge: int,
    align: int,
) -> tuple[int, int]:
    current_longest = max(req_w, req_h)
    if current_longest <= 0:
        return (align, align)

    scale = longest_edge / current_longest
    width = max(align, int(round(req_w * scale)))
    height = max(align, int(round(req_h * scale)))
    return (
        align_dimension(width, align),
        align_dimension(height, align),
    )


def build_generation_resolution_candidates(
    req_w: int,
    req_h: int,
    vram_tier: Optional[str],
    model_id: str,
    is_xl: bool,
    is_flux: bool,
    device: str,
) -> list[tuple[int, int]]:
    align = 16 if is_flux else 8
    requested = (align_dimension(req_w, align), align_dimension(req_h, align))
    initial = requested

    if device == "cuda":
        if is_flux:
            initial_cap = max(get_flux_resolution(vram_tier))
        else:
            initial_cap, _ = get_model_resolution_limits(model_id, vram_tier, is_xl)

        if max(requested) > initial_cap:
            initial = _scale_resolution_to_longest_edge(
                requested[0],
                requested[1],
                initial_cap,
                align,
            )

    fallback = _resolve_resolution(
        req_w, req_h, vram_tier, model_id, is_xl, is_flux, device
    )
    candidates = [initial]
    if fallback != initial:
        candidates.append(fallback)

    logger.info(
        "Generation resolution plan | requested=%sx%s | initial=%sx%s | fallback=%sx%s | candidates=%s",
        requested[0],
        requested[1],
        initial[0],
        initial[1],
        fallback[0],
        fallback[1],
        candidates,
    )
    return candidates


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/generate")
async def generate_image(req: GenerateRequest):
    aggressive_cleanup()

    device = get_device()
    vram_tier = get_vram_tier()
    is_flux = req.model_type == "flux-schnell"

    if device == "cuda":
        logger.info(f"Pre-generation VRAM: {get_vram_info()}")

    try:
        t2i, i2i = load_pipelines(req.model_type)

        loaded_id = current_model_id or ""
        is_xl = (
            "lightning" in loaded_id.lower()
            or "juggernaut" in loaded_id.lower()
            or "flux" in loaded_id.lower()
        )

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
        resolution_candidates = build_generation_resolution_candidates(
            req.width,
            req.height,
            vram_tier,
            loaded_id or req.model_type,
            is_xl,
            is_flux,
            device,
        )
        safe_prompt = req.prompt[:50].replace("\n", " ")
        output = None
        width, height = resolution_candidates[0]

        for idx, (candidate_w, candidate_h) in enumerate(resolution_candidates):
            width, height = candidate_w, candidate_h
            if idx == 0 and (width != req.width or height != req.height):
                logger.info(
                    "Initial generation resolution adjusted: %sx%s → %sx%s",
                    req.width,
                    req.height,
                    width,
                    height,
                )
            elif idx > 0:
                logger.info(
                    "Retrying generation at a lower resolution after OOM: %sx%s → %sx%s",
                    resolution_candidates[idx - 1][0],
                    resolution_candidates[idx - 1][1],
                    width,
                    height,
                )

            logger.info(
                f"Generating | {loaded_id} | "
                f"{steps} steps | {width}x{height} | "
                f"img2img={bool(req.image)} | '{safe_prompt}...'"
            )

            try:
                if req.image:
                    # img2img — SDXL / SD 1.5 only (Flux blocked above)
                    img_bytes = decode_base64_image(req.image)
                    init_image = Image.open(BytesIO(img_bytes)).convert("RGB")
                    init_image = init_image.resize(
                        (width, height), Image.Resampling.LANCZOS
                    )
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
                break
            except torch.cuda.OutOfMemoryError:
                aggressive_cleanup()
                if idx + 1 < len(resolution_candidates):
                    logger.warning(
                        "CUDA OOM during generation at %sx%s; trying the next resolution candidate.",
                        width,
                        height,
                    )
                    continue
                raise

        image = output.images[0]

        img_str, mime_type, file_extension = encode_image_output(
            image, req.output_format
        )

        aggressive_cleanup()

        if device == "cuda":
            logger.info(f"Post-generation VRAM: {get_vram_info()}")

        return {
            "status": "success",
            "image_base64": img_str,
            "mime_type": mime_type,
            "file_extension": file_extension,
            "output_format": req.output_format,
            "model_used": loaded_id,
            "requested_size": f"{req.width}x{req.height}",
            "actual_size": f"{width}x{height}",
            "steps_used": steps,
            "vram_tier": vram_tier,
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
    """AI upscale via Swin2SR or Real-ESRGAN."""
    aggressive_cleanup()
    logger.info(
        "Upscale request received | scale=%sx | upscaler=%s", req.scale, req.upscaler
    )

    try:
        img_bytes = decode_base64_image(req.image)
        init_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        requested_device = get_device()
        vram_tier = get_vram_tier()
        actual_scale, was_capped = resolve_upscale_scale(
            req.scale, requested_device, vram_tier, req.upscaler
        )
        total_passes = (
            {2: 1, 4: 2, 8: 3}[actual_scale]
            if req.upscaler == "swin2sr"
            else {2: 1, 4: 1, 8: 2}[actual_scale]
        )
        output_image = init_image
        final_device = requested_device
        final_tile_size = None
        used_cpu_fallback = False

        if was_capped:
            logger.warning(
                "Requested upscale x%s capped to x%s | upscaler=%s | device=%s | vram_tier=%s",
                req.scale,
                actual_scale,
                req.upscaler,
                requested_device,
                vram_tier,
            )

        if requested_device == "cuda":
            unload_generation_pipelines()

        if req.upscaler == "swin2sr":
            model, processor = load_upscale_model()
            try:
                for pass_index in range(total_passes):
                    logger.info(
                        "Upscale pass %s/%s | requested_device=%s | input=%sx%s",
                        pass_index + 1,
                        total_passes,
                        requested_device,
                        output_image.size[0],
                        output_image.size[1],
                    )
                    output_image, final_device, final_tile_size = run_upscale_pass(
                        model,
                        processor,
                        output_image,
                        requested_device,
                        vram_tier,
                    )
                    used_cpu_fallback = (
                        used_cpu_fallback or final_device != requested_device
                    )
            finally:
                del model, processor
                aggressive_cleanup()
        else:
            scales = {2: [2], 4: [4], 8: [4, 2]}[actual_scale]
            for pass_index, pass_scale in enumerate(scales, start=1):
                logger.info(
                    "Upscale pass %s/%s | requested_device=%s | input=%sx%s | outscale=%sx",
                    pass_index,
                    len(scales),
                    requested_device,
                    output_image.size[0],
                    output_image.size[1],
                    pass_scale,
                )
                output_image, final_device, final_tile_size = run_realesrgan_pass(
                    output_image,
                    requested_device,
                    vram_tier,
                    outscale=pass_scale,
                )
                used_cpu_fallback = (
                    used_cpu_fallback or final_device != requested_device
                )
            aggressive_cleanup()

        img_str, mime_type, file_extension = encode_image_output(
            output_image,
            req.output_format,
        )

        method = f"{'Real-ESRGAN' if req.upscaler == 'realesrgan' else 'Swin2SR'} x{req.scale}"
        if was_capped:
            method += f" (capped to x{actual_scale})"
        if used_cpu_fallback:
            method += " (CPU fallback)"

        if requested_device == "cuda":
            free_after = get_vram_info()
            logger.info(
                "Upscale finished | requested_scale=%sx | passes=%s | requested_device=%s | actual_device=%s | tile_size=%s | vram=%s",
                actual_scale,
                total_passes,
                requested_device,
                final_device,
                final_tile_size,
                free_after,
            )

        return {
            "status": "success",
            "image_base64": img_str,
            "mime_type": mime_type,
            "file_extension": file_extension,
            "output_format": req.output_format,
            "method": method,
            "scale": actual_scale,
            "requested_scale": req.scale,
            "upscaler": req.upscaler,
            "was_capped": was_capped,
            "passes": total_passes,
            "tile_size": final_tile_size,
        }

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
        "status": "ok",
        "device": get_device(),
        "loaded_model": current_model_id,
        "vram": get_vram_info(),
        "vram_tier": get_vram_tier(),
    }


@app.get("/models")
def list_models():
    device = get_device()
    vram_tier = get_vram_tier()
    vram_info = get_vram_info()
    total_vram = get_total_vram_gb()

    models = [
        {
            "id": "sdxl-lightning",
            "name": "SDXL Lightning (4-step)",
            "repo": "stabilityai/stable-diffusion-xl-base-1.0 + ByteDance/SDXL-Lightning",
            "speed": "fast",
            "use_case": "General generation, fast drafts",
            "supports_img2img": True,
            "supports_negative_prompt": True,
        },
        {
            "id": "stable-diffusion",
            "name": "Realistic Vision V6",
            "repo": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
            "speed": "medium",
            "use_case": "Portrait photography",
            "supports_img2img": True,
            "supports_negative_prompt": True,
        },
        {
            "id": "juggernaut-xl",
            "name": "Juggernaut XL v9 (Quality)",
            "repo": "RunDiffusion/Juggernaut-XL-v9",
            "speed": "medium",
            "use_case": "High-quality SDXL, stylized + realistic",
            "supports_img2img": True,
            "supports_negative_prompt": True,
        },
        {
            "id": "dalle-mini",
            "name": "Tiny-SD (Lightweight)",
            "repo": "segmind/tiny-sd",
            "speed": "very fast",
            "use_case": "Low-resource fallback",
            "supports_img2img": True,
            "supports_negative_prompt": True,
        },
        {
            "id": "flux-schnell",
            "name": "Flux.1-schnell (4-bit, Photorealism)",
            "repo": "black-forest-labs/FLUX.1-schnell",
            "speed": "medium",
            "use_case": "Best photorealism, text in image, complex prompts",
            "supports_img2img": False,
            "supports_negative_prompt": False,
            "available": device == "cuda" and total_vram >= FLUX_MIN_VRAM_GB,
            "requires_vram_gb": FLUX_MIN_VRAM_GB,
            "quantization": "NF4 4-bit T5 on <10GB cards",
        },
    ]

    for m in models:
        is_flux = m["id"] == "flux-schnell"
        is_xl = m["id"] in ("sdxl-lightning", "juggernaut-xl")

        if device == "cuda" and vram_info:
            if is_flux:
                w, h = get_flux_resolution(vram_tier)
                m["max_resolution"] = f"{w}x{h}"
                m["recommended_resolution"] = f"{w}x{h}"
                m["vram_usage"] = {
                    "mid_low": "~5-6GB (4-bit T5, sequential CPU offload, 512x512)",
                    "medium": "~7-8GB (4-bit T5)",
                    "high": "~10-12GB (bf16)",
                    "ultra": "~14-16GB (bf16)",
                }.get(vram_tier or "mid_low", "~6-8GB")
            else:
                max_res, rec_res = get_model_resolution_limits(
                    m["id"], vram_tier, is_xl
                )
                m["max_resolution"] = f"{max_res}x{max_res}"
                m["recommended_resolution"] = f"{rec_res}x{rec_res}"
                m["vram_usage"] = (
                    "~5-6GB"
                    if is_xl
                    else "~3-4GB" if m["id"] == "stable-diffusion" else "~2-3GB"
                )
        else:
            m["max_resolution"] = "512x512"
            m["recommended_resolution"] = "512x512"
            m["vram_usage"] = "N/A (CPU)"

    return {
        "device": device,
        "vram_tier": vram_tier,
        "total_vram": vram_info["total_gb"] if vram_info else None,
        "models": models,
    }


@app.get("/system")
def system_info():
    device = get_device()
    vram_tier = get_vram_tier()
    info = {"device": device, "vram_tier": vram_tier, "vram": get_vram_info()}

    if device == "cuda":
        use_seq = should_use_sequential_offload(vram_tier, model_type="sdxl-lightning")
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "flux_available": get_total_vram_gb() >= FLUX_MIN_VRAM_GB,
                "optimization_strategy": {
                    "sequential_offload": use_seq,
                    "reason": (
                        "Memory-optimized / sequential offload for SDXL/Flux on this tier"
                        if use_seq
                        else "Speed-optimized / model offload for SD 1.5-class models"
                    ),
                },
            }
        )
    return info
