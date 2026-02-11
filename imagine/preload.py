import torch
import logging
import os
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bipod.imagine.preload")

# Ensure environment is consistent
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# HF_HOME is usually set in Dockerfile/compose, but we ensure it's used
HF_HOME = os.environ.get("HF_HOME", "/app/models")

MODELS = [
    "runwayml/stable-diffusion-v1-5",
    "segmind/tiny-sd"
]

def preload():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting model preloading on {device}...")
    logger.info(f"Using HF_HOME: {HF_HOME}")
    
    for repo_id in MODELS:
        try:
            logger.info(f"Preloading model: {repo_id}...")
            
            # Step 1: Force download ONLY necessary files using snapshot_download
            # We ignore massive standalone checkpoints (.ckpt, .msgpack, etc) 
            # and only keep the diffusers-format subfolders.
            logger.info(f"Downloading optimized snapshot for {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                cache_dir=HF_HOME,
                local_files_only=False,
                ignore_patterns=[
                    "*.ckpt", 
                    "*.safetensors", 
                    "*.bin",
                    "*.pt",
                    "*.msgpack"
                ],
                # We explicitly allow the subfolder files we actually need
                allow_patterns=[
                    "*.json",
                    "*.txt",
                    "unet/*",
                    "vae/*",
                    "tokenizer/*",
                    "text_encoder/*",
                    "scheduler/*",
                    "feature_extractor/*",
                    "safety_checker/*"
                ]
            )

            # Step 2: Verify by loading onto device
            # This will trigger the actual download of the specific tensors we need (.bin or .safetensors)
            # inside those subfolders, but only for the specific variant/dtype we choose.
            logger.info(f"Verifying {repo_id} by loading into memory...")
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            variant = "fp16" if device == "cuda" else None
            
            try:
                StableDiffusionPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    cache_dir=HF_HOME,
                    local_files_only=False,
                    low_cpu_mem_usage=True
                )
            except Exception as variant_error:
                if variant == "fp16":
                    logger.warning(f"Failed to load fp16 variant for {repo_id}, falling back to default weights: {variant_error}")
                    StableDiffusionPipeline.from_pretrained(
                        repo_id,
                        torch_dtype=torch.float32, # Fallback to fp32
                        variant=None,
                        cache_dir=HF_HOME,
                        local_files_only=False,
                        low_cpu_mem_usage=True
                    )
                else:
                    raise variant_error

            logger.info(f"Successfully preloaded and verified {repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to preload {repo_id}: {e}")

if __name__ == "__main__":
    preload()
    logger.info("Preloading complete! Bipod is ready to imagine.")
