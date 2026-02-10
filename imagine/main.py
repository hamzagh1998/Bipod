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
    negative_prompt: Optional[str] = "blurred, low quality, ugly, disfigured, watermark"
    steps: int = 25
    strength: float = 0.75 # How much to transform the input image (0.0=min, 1.0=max)
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    model_type: str = "stable-diffusion"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(model_type: str):
    global pipeline, current_model_id
    
    device = get_device()
    logger.info(f"Loading model '{model_type}' on {device}...")

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
        if device == "cuda":
            pipe = StableDiffusionPipeline.from_pretrained(
                repo_id, 
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            pipe.to("cuda")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                repo_id, 
                use_safetensors=True
            )
            pipe.to("cpu")
            pipe.enable_attention_slicing()

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        pipeline = pipe
        current_model_id = repo_id
        logger.info(f"Model '{repo_id}' loaded successfully.")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

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
