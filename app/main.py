import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from app.api import router as api_router
import traceback
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

# Mount generated files (Bipod's creations)
# Ensure directory exists first (handled in config, but safe to check)
if not os.path.exists(settings.GENERATED_DIR):
    os.makedirs(settings.GENERATED_DIR)
app.mount("/generated", StaticFiles(directory=settings.GENERATED_DIR), name="generated")

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

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception caught: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"---> {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"<--- {request.method} {request.url.path} - {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Middleware Exception: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": str(e)})

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

