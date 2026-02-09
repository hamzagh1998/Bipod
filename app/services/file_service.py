import os
import glob
from typing import List, Optional
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("bipod.services.file")

class FileService:
    """Service for managing local documents and host filesystem access."""
    
    def __init__(self):
        self.host_root = settings.HOST_ROOT

    def get_host_path(self, path: str) -> str:
        """Converts a standard host path to the container-mapped path."""
        # Clean leading slash for joining
        clean_path = path.lstrip("/")
        return os.path.join(self.host_root, clean_path)

    async def search_host(self, pattern: str, depth: int = 2) -> List[str]:
        """
        Searches the host filesystem for files matching a glob pattern.
        Example: search_host("*.pdf") or search_host("projects/secret_plans/*")
        """
        try:
            full_pattern = self.get_host_path(pattern)
            # Use glob for pattern matching
            files = glob.glob(full_pattern, recursive=True)
            
            # Convert back to clean host paths for the agent's response
            return [f.replace(self.host_root, "") for f in files]
        except Exception as e:
            logger.error(f"Failed to search host: {e}")
            return []

    async def read_host_file(self, path: str, max_chars: int = 5000) -> Optional[str]:
        """Reads a file from the host system."""
        try:
            full_path = self.get_host_path(path)
            if not os.path.exists(full_path):
                return None
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)
                return content
        except Exception as e:
            logger.error(f"Failed to read host file {path}: {e}")
            return None

file_service = FileService()
