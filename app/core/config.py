import os
import logging
import platform
import subprocess
from typing import Literal
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("bipod.config")

def detect_hardware_arch() -> str:
    """Returns 'arm64' or 'amd64' based on the system architecture."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "arm64"
    return "amd64"

def detect_gpu_presence() -> bool:
    """Checks if an NVIDIA GPU is accessible within the container."""
    try:
        # We check for nvidia-smi. If it fails, we assume CPU-only.
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=True, 
        extra="ignore"
    )

    PROJECT_NAME: str = "Bipod"
    API_V1_STR: str = "/api/v1"
    
    # --- Hardware Auto-Detection ---
    # These use default_factories to probe hardware at startup
    HARDWARE_TARGET: Literal["amd64", "arm64"] = Field(default_factory=detect_hardware_arch)
    USE_GPU: bool = Field(default_factory=detect_gpu_presence)
    
    # --- Storage Paths ---
    # Use /app/data if inside container, else local ./data
    DATA_DIR: str = "/app/data" if os.path.exists("/app") else os.path.join(os.getcwd(), "data")
    
    @computed_field
    @property
    def DOCUMENTS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "documents")

    @computed_field
    @property
    def MEMORY_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "memory")

    
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return f"sqlite:///{self.MEMORY_DIR}/bipod_memory.db"

    @computed_field
    @property
    def OLLAMA_BASE_URL(self) -> str:
        """Dynamic Ollama URL: 'ollama' inside Docker, 'localhost' outside."""
        import socket
        host = "ollama"
        try:
            socket.gethostbyname(host)
            return f"http://{host}:11434"
        except socket.gaierror:
            return "http://localhost:11434"

    
    # Brain tiers
    HEAVY_MODEL: str = "llama3.1:8b"   # For PC + GPU
    MEDIUM_MODEL: str = "llama3.2:3b"  # For PC + CPU
    LIGHT_MODEL: str = "llama3.2:1b"   # For Pi 5 / ARM64
    VISION_MODEL: str = "moondream"    # Specialized for eyes

    @computed_field
    @property
    def ACTIVE_MODEL(self) -> str:
        """Dynamically picks the best brain based on detected hardware."""
        if self.HARDWARE_TARGET == "arm64":
            return self.LIGHT_MODEL
        if self.USE_GPU:
            return self.HEAVY_MODEL
        return self.MEDIUM_MODEL

settings = Settings()

# Ensure directories exist on startup
os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
os.makedirs(settings.MEMORY_DIR, exist_ok=True)