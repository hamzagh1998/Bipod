import torch
import logging
import os
import sys
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from huggingface_hub import snapshot_download

# Configure Logging to both file and stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/app/preload.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("bipod.imagine.preload")

# Ensure environment is consistent
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
HF_HOME = os.environ.get("HF_HOME", "/app/models")

MODELS = [
    # Photorealism
    {"id": "SG161222/Realistic_Vision_V6.0_B1_noVAE", "type": "sd"},
    {"id": "stabilityai/sd-vae-ft-mse", "type": "vae"},
    
    # THE BIG ONE (SDXL Lightning)
    {"id": "ByteDance/SDXL-Lightning-4step-base", "type": "sdxl"},
    {"id": "madebyollin/sdxl-vae-fp16-fix", "type": "vae"}
]

def preload():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting model preloading on {device}...")
    logger.info(f"Using HF_HOME: {HF_HOME}")
    
    for model in MODELS:
        repo_id = model["id"]
        m_type = model["type"]
        
        try:
            logger.info(f"Processing {m_type}: {repo_id}...")
            
            if m_type == "vae":
                logger.info(f"Downloading VAE: {repo_id}...")
                snapshot_download(repo_id=repo_id, cache_dir=HF_HOME, local_files_only=False)
                logger.info(f"Successfully downloaded VAE: {repo_id}")
                continue

            # Step 1: Download necessary files
            logger.info(f"Downloading snapshot for {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                cache_dir=HF_HOME,
                local_files_only=False,
                ignore_patterns=["*.ckpt", "*.msgpack"]
            )

            # Step 2: Verify and trigger tensor download
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            variant = "fp16" if device == "cuda" else None
            
            pipe_cls = StableDiffusionXLPipeline if m_type == "sdxl" else StableDiffusionPipeline
            
            logger.info(f"Verifying {repo_id} with {pipe_cls.__name__}...")
            try:
                pipe_cls.from_pretrained(
                    repo_id,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    cache_dir=HF_HOME,
                    local_files_only=False,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                logger.warning(f"Failed to load optimized variant ({variant}), trying default: {e}")
                pipe_cls.from_pretrained(
                    repo_id,
                    torch_dtype=torch_dtype, # Keep dtype if possible
                    variant=None,
                    cache_dir=HF_HOME,
                    local_files_only=False,
                    low_cpu_mem_usage=True
                )

            logger.info(f"Successfully verified {repo_id}")
            
        except Exception as e:
            logger.error(f"CRITICAL failure for {repo_id}: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        preload()
        logger.info("Preloading complete! Bipod is ready to imagine.")
    except Exception as top_e:
        logger.error(f"Global preload script failure: {top_e}", exc_info=True)
        sys.exit(1)
