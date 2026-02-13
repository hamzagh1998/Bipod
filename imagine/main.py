import os
import gc

# 1. Environment Configuration
# MUST be set before torch is imported for maximum effect
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# "expandable_segments:True" helps prevent VRAM fragmentation on 6GB cards
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
import base64
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoencoderKL,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler
)
from PIL import Image

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bipod.imagine")

app = FastAPI(title="Bipod Imagine Service")

# Global variables for caching models
txt2img_pipe = None
img2img_pipe = None
current_model_id = None

class GenerateRequest(BaseModel):
    prompt: str
    image: Optional[str] = None # Base64 encoded input image for img2img
    negative_prompt: Optional[str] = (
        "extra fingers, mutated hands, blurry, low quality, masterpiece, "
        "worst quality, (disfigured, ugly, bad anatomy, bad proportions), "
        "watermark, text, sign, profile, logo, 3d render"
    )
    steps: int = 4 
    strength: float = 0.6
    guidance_scale: float = 0.0 
    width: int = 1024
    height: int = 1024
    model_type: str = "stable-diffusion-xl"

HF_HOME = os.environ.get("HF_HOME", "/app/models")
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "true").lower() == "true"

def get_device():
    """Detect available device with preference: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_vram_info():
    """Get current VRAM usage statistics"""
    if not torch.cuda.is_available():
        return None
    
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    
    return {
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - reserved, 2)
    }

def get_vram_tier():
    """
    Classify GPU VRAM into tiers for optimization strategy.
    Returns: 'low' (<6GB), 'medium' (6-10GB), 'high' (10-16GB), 'ultra' (>16GB), or None (no CUDA)
    """
    if not torch.cuda.is_available():
        return None
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if total_vram < 6:
        return "low"
    elif total_vram < 10:
        return "medium"
    elif total_vram < 16:
        return "high"
    else:
        return "ultra"

def get_optimal_resolution(model_type: str, vram_tier: str, is_xl: bool):
    """
    Get optimal max resolution based on VRAM tier and model type.
    Returns: (max_resolution, recommended_resolution)
    """
    if vram_tier is None:  # CPU
        return (512, 512)
    
    # SDXL Models
    if is_xl:
        if vram_tier == "low":  # <6GB
            return (768, 512)  # Max, Recommended
        elif vram_tier == "medium":  # 6-10GB
            return (1024, 1024)
        elif vram_tier == "high":  # 10-16GB
            return (1536, 1024)
        else:  # ultra >16GB
            return (2048, 1536)
    
    # SD 1.5 Models
    else:
        if vram_tier == "low":
            return (512, 512)
        elif vram_tier == "medium":
            return (768, 512)
        elif vram_tier == "high":
            return (1024, 768)
        else:  # ultra
            return (1536, 1024)

def align_dimension(dim: int, alignment: int = 8) -> int:
    """Align dimension to nearest multiple (SDXL requires divisibility by 8)"""
    return (dim // alignment) * alignment

def aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def should_use_sequential_offload(vram_tier: str) -> bool:
    """
    Determine if sequential CPU offload should be used based on VRAM.
    Sequential offload adds ~15-20% latency but saves ~2GB VRAM.
    """
    # Use sequential offload only for low/medium VRAM
    # High/ultra VRAM can keep everything on GPU for speed
    return vram_tier in ["low", "medium"]

def load_pipelines(model_type: str):
    global txt2img_pipe, img2img_pipe, current_model_id
    
    # Map model types to repo IDs
    if model_type == "dalle-mini":
        repo_id = "segmind/tiny-sd"
    elif model_type == "stable-diffusion":
        repo_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    else:
        # Default to SDXL Lightning
        repo_id = "ByteDance/SDXL-Lightning-4step-base"
    
    # Return existing pipelines if model hasn't changed
    if txt2img_pipe is not None and current_model_id == repo_id:
        logger.info(f"Reusing cached model: {current_model_id}")
        return txt2img_pipe, img2img_pipe

    # CLEANUP: Aggressive memory clearing before loading new model
    if txt2img_pipe is not None:
        logger.info(f"Unloading model '{current_model_id}' to free VRAM...")
        del txt2img_pipe
        del img2img_pipe
        txt2img_pipe = None
        img2img_pipe = None
        aggressive_cleanup()

    device = get_device()
    vram_tier = get_vram_tier()
    logger.info(f"Loading {repo_id} on {device} (VRAM Tier: {vram_tier}, Offline: {OFFLINE_MODE})...")
    
    if device == "cuda":
        vram_info = get_vram_info()
        logger.info(f"VRAM Status: {vram_info}")

    try:
        common_args = {
            "cache_dir": HF_HOME,
            "local_files_only": OFFLINE_MODE,
            "use_safetensors": True,
            "low_cpu_mem_usage": True
        }

        if device == "cuda":
            # 1. Load VAE separately (FP16 Fix for SDXL or standard for others)
            is_xl = "xl" in repo_id.lower()
            vae_id = "madebyollin/sdxl-vae-fp16-fix" if is_xl else "stabilityai/sd-vae-ft-mse"
            
            logger.info(f"Loading VAE: {vae_id}")
            vae = AutoencoderKL.from_pretrained(
                vae_id, 
                torch_dtype=torch.float16, 
                **common_args
            )
            
            # 2. Load Text2Img Pipeline
            logger.info(f"Loading Text2Img pipeline...")
            txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                repo_id, 
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16" if "tiny" not in repo_id.lower() else None,
                **common_args
            )
            
            # 3. CONFIGURE SCHEDULER (Must happen BEFORE from_pipe to be inherited)
            if "lightning" in repo_id.lower() or "turbo" in repo_id.lower():
                logger.info("Configuring Euler scheduler for Lightning/Turbo...")
                txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
                    txt2img_pipe.scheduler.config, 
                    timestep_spacing="trailing"
                )
            elif "tiny" in repo_id.lower():
                logger.info("Configuring DPM++ scheduler for Tiny-SD...")
                txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    txt2img_pipe.scheduler.config, 
                    use_karras_sigmas=True
                )

            # 4. Create Img2Img Pipeline sharing the SAME components (inherits scheduler)
            logger.info("Creating Img2Img pipeline from Text2Img...")
            img2img_pipe = AutoPipelineForImage2Image.from_pipe(txt2img_pipe)
            
            # 5. Apply Memory Optimizations (Strategy varies by VRAM tier)
            use_sequential = should_use_sequential_offload(vram_tier)
            logger.info(f"Applying optimizations for VRAM tier '{vram_tier}' (Sequential offload: {use_sequential})...")
            
            # Try xFormers first (most efficient)
            try:
                txt2img_pipe.enable_xformers_memory_efficient_attention()
                img2img_pipe.enable_xformers_memory_efficient_attention()
                logger.info("✓ xFormers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"xFormers unavailable ({e}), falling back to attention slicing")
                txt2img_pipe.enable_attention_slicing(slice_size="auto")
                img2img_pipe.enable_attention_slicing(slice_size="auto")
                logger.info("✓ Attention slicing enabled")

            # CPU offload strategy based on VRAM
            if use_sequential:
                # Low/Medium VRAM: Use sequential offload (aggressive, slower)
                txt2img_pipe.enable_sequential_cpu_offload()
                img2img_pipe.enable_sequential_cpu_offload()
                logger.info("✓ Sequential CPU offload enabled (memory-optimized mode)")
            else:
                # High/Ultra VRAM: Use model offload or keep on GPU (faster)
                txt2img_pipe.enable_model_cpu_offload()
                img2img_pipe.enable_model_cpu_offload()
                logger.info("✓ Model CPU offload enabled (speed-optimized mode)")
            
            # VAE optimizations (always useful)
            txt2img_pipe.enable_vae_tiling()
            txt2img_pipe.enable_vae_slicing()
            img2img_pipe.enable_vae_tiling()
            img2img_pipe.enable_vae_slicing()
            logger.info("✓ VAE tiling and slicing enabled")
            
            # Final cleanup
            aggressive_cleanup()
            
            vram_after = get_vram_info()
            logger.info(f"Model loaded. VRAM after optimizations: {vram_after}")

        else:
            # CPU/MPS Fallback
            logger.info("Loading for CPU/MPS (no VRAM optimizations needed)...")
            txt2img_pipe = AutoPipelineForText2Image.from_pretrained(repo_id, **common_args)
            
            # Sync scheduler for CPU too
            if "lightning" in repo_id.lower() or "turbo" in repo_id.lower():
                txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
                    txt2img_pipe.scheduler.config, 
                    timestep_spacing="trailing"
                )
            elif "tiny" in repo_id.lower():
                txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    txt2img_pipe.scheduler.config, 
                    use_karras_sigmas=True
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
        if OFFLINE_MODE and ("Local" in error_msg or "not found" in error_msg.lower()):
             error_msg = f"Model '{repo_id}' not found in local cache. Please run 'docker exec -it bipod_imagine python preload.py' first."
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    # Proactive cleanup before generation
    aggressive_cleanup()
    
    device = get_device()
    vram_tier = get_vram_tier()
    
    if device == "cuda":
        vram_before = get_vram_info()
        logger.info(f"Pre-generation VRAM: {vram_before}")

    try:
        t2i, i2i = load_pipelines(req.model_type)
        
        # Model-Specific Constraints & Step Defaults
        steps = req.steps
        guidance = req.guidance_scale
        
        is_xl = "xl" in (current_model_id or "").lower()
        is_lightning = "lightning" in (current_model_id or "").lower()
        is_turbo = "turbo" in (current_model_id or "").lower()
        is_tiny = "tiny" in (current_model_id or "").lower()
        
        # Auto-configure steps and guidance for fast models
        if is_lightning:
            steps = 4
            guidance = 0.0
            logger.info("Lightning model: forcing 4 steps, CFG=0")
        elif is_turbo:
            steps = 1
            guidance = 0.0
            logger.info("Turbo model: forcing 1 step, CFG=0")
        elif req.steps == 4:  # User left default but we're on standard SD
            if is_tiny:
                steps = 20
                guidance = 7.5
            else:
                steps = 30
                guidance = 7.0
            logger.info(f"Standard model: adjusting to {steps} steps, CFG={guidance}")

        # Get optimal resolution based on VRAM tier
        width = req.width
        height = req.height
        
        if device == "cuda":
            max_res, recommended_res = get_optimal_resolution(req.model_type, vram_tier, is_xl)
            
            if width > max_res or height > max_res:
                logger.warning(
                    f"Resolution {width}x{height} exceeds safe limit for "
                    f"VRAM tier '{vram_tier}' with {current_model_id}. "
                    f"Capping to {max_res}x{max_res}"
                )
                width = min(width, max_res)
                height = min(height, max_res)
        
        # Align dimensions to multiple of 8 (SDXL requirement)
        width = align_dimension(width, 8)
        height = align_dimension(height, 8)
        
        if width != req.width or height != req.height:
            logger.info(f"Aligned dimensions: {req.width}x{req.height} → {width}x{height}")

        logger.info(
            f"Generating: '{req.prompt[:50]}...' | "
            f"Model: {current_model_id} | "
            f"Steps: {steps} | "
            f"Size: {width}x{height} | "
            f"Img2Img: {bool(req.image)}"
        )

        if req.image:
            # Img2Img Path
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
            # Text2Img Path
            output = t2i(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height
            )
            
        image = output.images[0]

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Post-generation cleanup
        aggressive_cleanup()
        
        if device == "cuda":
            vram_after = get_vram_info()
            logger.info(f"Post-generation VRAM: {vram_after}")
        
        return {
            "status": "success", 
            "image_base64": img_str,
            "model_used": current_model_id,
            "actual_size": f"{width}x{height}",
            "steps_used": steps,
            "vram_tier": vram_tier
        }

    except torch.cuda.OutOfMemoryError as e:
        logger.error("CUDA OOM Error - try lower resolution or different model")
        aggressive_cleanup()
        raise HTTPException(
            status_code=507,
            detail=(
                "GPU out of memory. Try: (1) Lower resolution, "
                "(2) Use 'dalle-mini' model, or (3) Restart the service"
            )
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        aggressive_cleanup()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    vram = get_vram_info() if torch.cuda.is_available() else None
    vram_tier = get_vram_tier()
    
    return {
        "status": "ok",
        "device": get_device(),
        "loaded_model": current_model_id,
        "vram": vram,
        "vram_tier": vram_tier
    }

@app.get("/models")
def list_models():
    """List available model types with dynamic specs based on current hardware"""
    device = get_device()
    vram_tier = get_vram_tier()
    vram_info = get_vram_info()
    
    # Base model info
    models = [
        {
            "id": "stable-diffusion-xl",
            "name": "SDXL Lightning (4-step)",
            "repo": "ByteDance/SDXL-Lightning-4step-base",
            "speed": "fast"
        },
        {
            "id": "stable-diffusion",
            "name": "Realistic Vision V6",
            "repo": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
            "speed": "medium"
        },
        {
            "id": "dalle-mini",
            "name": "Tiny-SD (Lightweight)",
            "repo": "segmind/tiny-sd",
            "speed": "very fast"
        }
    ]
    
    # Add dynamic specs based on current hardware
    if device == "cuda" and vram_info:
        for model in models:
            is_xl = model["id"] == "stable-diffusion-xl"
            max_res, rec_res = get_optimal_resolution(model["id"], vram_tier, is_xl)
            
            # Estimate VRAM usage (rough approximations)
            if model["id"] == "stable-diffusion-xl":
                if vram_tier == "low":
                    vram_usage = "~5-6GB"
                elif vram_tier == "medium":
                    vram_usage = "~5-7GB"
                else:
                    vram_usage = "~6-8GB"
            elif model["id"] == "stable-diffusion":
                vram_usage = "~3-4GB"
            else:  # tiny-sd
                vram_usage = "~2-3GB"
            
            model["vram_usage"] = vram_usage
            model["max_resolution"] = f"{max_res}x{max_res}"
            model["recommended_resolution"] = f"{rec_res}x{rec_res}"
    else:
        # CPU mode
        for model in models:
            model["vram_usage"] = "N/A (CPU)"
            model["max_resolution"] = "512x512"
            model["recommended_resolution"] = "512x512"
    
    return {
        "device": device,
        "vram_tier": vram_tier,
        "total_vram": vram_info["total_gb"] if vram_info else None,
        "models": models
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
        "vram": vram_info
    }
    
    if device == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        
        # Optimization strategy being used
        info["optimization_strategy"] = {
            "sequential_offload": should_use_sequential_offload(vram_tier),
            "reason": "Memory-optimized (slower)" if should_use_sequential_offload(vram_tier) else "Speed-optimized (faster)"
        }
    
    return info
