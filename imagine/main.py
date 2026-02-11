import os
import torch
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
import base64
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bipod.imagine")

app = FastAPI(title="Bipod Imagine Service")

# Global variables for caching models
pipeline = None
current_model_id = None

class GenerateRequest(BaseModel):
    prompt: str
    image: Optional[str] = None # Base64 encoded input image for img2img
    negative_prompt: Optional[str] = "blurred, low quality, ugly, disfigured, watermark, extra limbs, bad anatomy, out of focus, low res"
    steps: int = 40
    strength: float = 0.75 # How much to transform the input image (0.0=min, 1.0=max)
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    model_type: str = "stable-diffusion"

# Environment configuration for cache stability and memory efficiency
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
HF_HOME = os.environ.get("HF_HOME", "/app/models")
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "true").lower() == "true"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(model_type: str):
    global pipeline, current_model_id
    
    device = get_device()
    logger.info(f"Loading model '{model_type}' on {device} (Offline: {OFFLINE_MODE})...")

    if model_type == "dalle-mini":
        repo_id = "segmind/tiny-sd"
    else:
        repo_id = "runwayml/stable-diffusion-v1-5"

    if pipeline is not None and current_model_id == repo_id:
        return pipeline

    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()

    try:
        common_args = {
            "cache_dir": HF_HOME,
            "local_files_only": OFFLINE_MODE,
            "low_cpu_mem_usage": True,
            "use_safetensors": None # Allow flexible loading
        }

        if device == "cuda":
            try:
                # First try default fp16 loading
                pipe = StableDiffusionPipeline.from_pretrained(
                    repo_id, 
                    torch_dtype=torch.float16,
                    variant="fp16",
                    **common_args
                )
            except Exception as e:
                logger.warning(f"Failed to load fp16 variant for {repo_id}, trying default: {e}")
                pipe = StableDiffusionPipeline.from_pretrained(
                    repo_id, 
                    torch_dtype=torch.float32,
                    variant=None,
                    **common_args
                )
            
            # Optimization: Try model offloading for much faster performance
            # This is safe because Bipod unloads Ollama from VRAM before calling us.
            try:
                # Try model offload (faster than sequential)
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()
                logger.info("Enabled model CPU offloading and attention slicing.")
            except Exception as opt_e:
                logger.warning(f"Could not enable model CPU offloading: {opt_e}. Trying sequential fallback...")
                try:
                    pipe.enable_sequential_cpu_offload()
                    pipe.enable_attention_slicing()
                    logger.info("Enabled sequential CPU offloading fallback.")
                except Exception as seq_e:
                    logger.warning(f"Sequential offload also failed: {seq_e}. Falling back to CPU mode.")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        repo_id, 
                        torch_dtype=torch.float32,
                        **common_args
                    )
                    pipe.to("cpu")
                    pipe.enable_attention_slicing()
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                repo_id, 
                **common_args
            )
            pipe.to("cpu")
            pipe.enable_attention_slicing()

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        
        pipeline = pipe
        current_model_id = repo_id
        logger.info(f"Model '{repo_id}' loaded successfully from {'local cache' if OFFLINE_MODE else 'Hub/cache'}.")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        error_msg = str(e)
        if OFFLINE_MODE and "Local look-up not found" in error_msg:
            error_msg = f"Model '{repo_id}' not found in local cache. Please run 'docker exec -it bipod_imagine python preload.py' first."
        raise HTTPException(status_code=500, detail=f"Model loading failed: {error_msg}")

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    global pipeline
    
    try:
        pipe = load_model(req.model_type)
        
        # Determine step count
        steps = req.steps
        if get_device() == "cpu" and steps > 10:
            steps = 2
            if req.model_type == "dalle-mini":
                 steps = 8
        
        logger.info(f"Generating image for prompt: '{req.prompt}' with {steps} steps")

        generator = None
        # Handle Img2Img if image is provided
        if req.image:
            logger.info("Input image provided - switching to Img2Img mode")
            # Decode input image
            img_bytes = base64.b64decode(req.image)
            init_image = Image.open(BytesIO(img_bytes)).convert("RGB")
            init_image = init_image.resize((req.width, req.height)) # Resize to avoid dimension errors

            # Create img2img pipeline from existing txt2img pipe (zero-copy)
            img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(pipe)
            img2img_pipe.to(pipe.device)
            
            output = img2img_pipe(
                prompt=req.prompt,
                image=init_image,
                strength=req.strength,
                negative_prompt=req.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=req.guidance_scale,
            )
        else:
            # Standard Text2Img
            output = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height
            )
            
        image = output.images[0]

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success", 
            "image_base64": img_str,
            "model_used": current_model_id
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "device": get_device(), "loaded_model": current_model_id}
