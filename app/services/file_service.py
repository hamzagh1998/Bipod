import os
import subprocess
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
        clean_path = path.lstrip("/")
        return os.path.join(self.host_root, clean_path)

    async def search_host(self, pattern: str, root_dir: str = None) -> List[str]:
        """
        Efficiently searches the host filesystem using the 'find' command.
        If root_dir is specified, searches within that directory (mapped to host root).
        """
        try:
            if root_dir:
                # Map the given root dir through the host root
                search_dir = self.get_host_path(root_dir)
            else:
                search_dir = self.host_root
            
            # Use 'find' for high-performance searching
            # -iname for case-insensitive matching
            cmd = ["find", search_dir, "-maxdepth", "12", "-iname", f"*{pattern}*"]
            
            logger.info(f"Running search command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                logger.error(f"Find command failed: {result.stderr}")
                return []

            files = result.stdout.splitlines()
            # Convert back to clean host paths and limit results
            return [f.replace(self.host_root, "") for f in files[:20]]
        except subprocess.TimeoutExpired:
            logger.warning("File search timed out (large directory)")
            return ["Search timed out. Try a more specific directory if possible."]
        except Exception as e:
            logger.error(f"Failed to search host: {e}")
            return []

    async def read_host_file(self, path: str, max_chars: int = 5000) -> Optional[str]:
        """Reads a file from the host system, supporting text and PDF."""
        try:
            full_path = self.get_host_path(path)
            if not os.path.exists(full_path):
                return None
            
            # Handle PDF files
            if path.lower().endswith(".pdf"):
                return await self._read_pdf(full_path, max_chars)
            
            # Default text read
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)
                return content
        except Exception as e:
            logger.error(f"Failed to read host file {path}: {e}")
            return f"Error reading file: {str(e)}"

    async def _read_pdf(self, full_path: str, max_chars: int) -> str:
        """Extracts text from a PDF file."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(full_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
                if len(text) > max_chars:
                    break
            return text[:max_chars]
        except Exception as e:
            logger.error(f"PDF extraction failed for {full_path}: {e}")
            return f"Failed to extract text from PDF: {str(e)}"

    async def read_host_image(self, path: str) -> Optional[str]:
        """Reads an image file from the host system and returns its base64 encoding."""
        try:
            import base64
            full_path = self.get_host_path(path)
            if not os.path.exists(full_path):
                return None
            
            with open(full_path, 'rb') as f:
                content = f.read()
                return base64.b64encode(content).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to read image file {path}: {e}")
            return None

    async def write_host_file(self, path: str, content: str) -> str | None:
        """Writes content to a file on the host filesystem.
        
        Returns the host-visible path on success, or None on failure.
        The system prompt ensures this is only called when the user explicitly
        requests a file to be created or saved.
        """
        try:
            full_path = self.get_host_path(path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"File saved to container path: {full_path} (host path: {path})")
            return path  # Return the clean host path
        except Exception as e:
            logger.error(f"Failed to write host file {path}: {e}")
            return None

file_service = FileService()
