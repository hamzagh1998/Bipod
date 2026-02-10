import os
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import router as api_router
from app.core.config import settings
from app.core.logger import setup_logging, get_logger
from app.db.database import init_db

# Initialize beautiful colored logs
setup_logging()
logger = get_logger("bipod.main")

app = FastAPI(
    title="Bipod Logic Server",
    description="The brain of your weightless AI companion.",
    version="0.1.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Enable Python JIT check
if os.environ.get("PYTHON_JIT") == "on":
    logger.info("🚀 Python JIT is enabled and running.")

@app.on_event("startup")
async def startup_event():
    await init_db()
    logger.info(f"Bipod is waking up on hardware: {settings.HARDWARE_TARGET}")
    logger.info(f"Active Brain Model: {settings.ACTIVE_MODEL}")
    logger.info(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
    logger.info(f"HARDWARE_TARGET: {settings.HARDWARE_TARGET}")
    if settings.USE_GPU:
        logger.info("Detected NVIDIA GPU - Using Heavy Inference mode.")
    else:
        logger.warning("No GPU detected - Running in CPU/Efficient mode.")
    
    # Initialize services, VAD, etc here

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

