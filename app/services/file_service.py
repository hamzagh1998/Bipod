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
        self._host_root_realpath = os.path.realpath(self.host_root)

    def get_host_path(self, path: str) -> str:
        """Converts a host path to a validated, absolute path under HOST_ROOT."""
        if not path or not isinstance(path, str):
            raise ValueError("Invalid path")

        raw_path = path.strip()
        base = self._host_root_realpath

        # Allow direct absolute paths only when they are already under HOST_ROOT.
        if base == "/":
            candidate = os.path.realpath(raw_path if os.path.isabs(raw_path) else os.path.join("/", raw_path))
        elif os.path.isabs(raw_path):
            direct_candidate = os.path.realpath(raw_path)
            if os.path.commonpath([direct_candidate, base]) == base:
                candidate = direct_candidate
            else:
                candidate = os.path.realpath(os.path.join(base, raw_path.lstrip("/")))
        else:
            candidate = os.path.realpath(os.path.join(base, raw_path.lstrip("/")))

        try:
            if os.path.commonpath([candidate, base]) != base:
                raise ValueError("Path escapes allowed host root")
        except ValueError:
            # Raised by commonpath for mixed absolute/relative paths
            raise ValueError("Invalid path")

        return candidate

    def _to_visible_host_path(self, full_path: str) -> str:
        """Converts an internal mapped path back to a host-visible absolute path."""
        base = self._host_root_realpath
        if base == "/":
            return full_path
        if full_path.startswith(base):
            suffix = full_path[len(base):]
            return suffix if suffix.startswith("/") else f"/{suffix}"
        return full_path

    async def search_host(self, pattern: str, root_dir: str = None) -> List[str]:
        """
        Efficiently searches the host filesystem using the 'find' command.
        If root_dir is specified, searches within that directory (mapped to host root).
        """
        try:
            if not pattern:
                return []

            if root_dir:
                # Map the given root dir through the host root
                search_dir = self.get_host_path(root_dir)
            else:
                search_dir = self._host_root_realpath
            
            # Use 'find' for high-performance searching
            # -iname for case-insensitive matching. Exclude system dirs and hidden files.
            cmd = [
                "find", search_dir, "-maxdepth", "12", 
                "-not", "-path", "*/.*", 
                "-not", "-path", f"{self.host_root}/proc*", 
                "-not", "-path", f"{self.host_root}/sys*", 
                "-not", "-path", f"{self.host_root}/dev*",
                "-iname", f"*{pattern}*"
            ]
            
            logger.info(f"Running search command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                logger.error(f"Find command failed: {result.stderr}")
                return []

            files = result.stdout.splitlines()
            # Convert back to clean host paths and limit results
            return [self._to_visible_host_path(f) for f in files[:20]]
        except ValueError as e:
            logger.warning(f"Blocked unsafe path in search_host: {e}")
            return []
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
        except ValueError as e:
            logger.warning(f"Blocked unsafe path in read_host_file: {e}")
            return "Error reading file: invalid path"
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
        except ValueError as e:
            logger.warning(f"Blocked unsafe path in read_host_image: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to read image file {path}: {e}")
            return None

    async def write_host_file(self, path: str, content: str) -> str | None:
        """Writes content to a file on the host filesystem."""
        try:
            full_path = self.get_host_path(path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"File saved to container path: {full_path} (host path: {path})")
            return path
        except ValueError as e:
            logger.warning(f"Blocked unsafe path in write_host_file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to write host file {path}: {e}")
            return None

    async def move_host_file(self, src: str, dest: str) -> bool:
        """Moves or renames a file/directory on the host. Supports wildcards (glob)."""
        try:
            import glob
            import shutil
            
            src_full = self.get_host_path(src)
            dest_full = self.get_host_path(dest)
            
            # Find all matching source files
            src_matches = glob.glob(src_full)
            if not src_matches:
                logger.warning(f"No files matched source pattern: {src_full}")
                return False

            # Ensure destination directory exists if moving multiple files
            if len(src_matches) > 1 or os.path.isdir(dest_full):
                os.makedirs(dest_full, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest_full), exist_ok=True)

            moved_any = False
            for match in src_matches:
                real_match = os.path.realpath(match)
                if os.path.commonpath([real_match, self._host_root_realpath]) != self._host_root_realpath:
                    logger.warning(f"Blocked unsafe source match outside host root: {match}")
                    continue
                shutil.move(match, dest_full)
                logger.info(f"Moved {match} to {dest_full}")
                moved_any = True
            
            return moved_any
        except ValueError as e:
            logger.warning(f"Blocked unsafe path in move_host_file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to move {src} to {dest}: {e}")
            return False

    async def delete_host_file(self, path: str) -> bool:
        """Deletes a file or directory on the host."""
        try:
            full_path = self.get_host_path(path)
            if os.path.exists(full_path):
                if os.path.isdir(full_path):
                    import shutil
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)
                logger.info(f"Deleted {path}")
                return True
            return False
        except ValueError as e:
            logger.warning(f"Blocked unsafe path in delete_host_file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False

    async def organize_host_directory(self, directory: str) -> int:
        """Organizes files in a directory into subfolders by extension."""
        try:
            full_dir = self.get_host_path(directory)
            if not os.path.isdir(full_dir):
                return 0
            
            import shutil
            count = 0
            for item in os.listdir(full_dir):
                item_path = os.path.join(full_dir, item)
                if os.path.isfile(item_path):
                    ext = item.split(".")[-1].lower() if "." in item else "no_extension"
                    target_fold = os.path.join(full_dir, ext)
                    os.makedirs(target_fold, exist_ok=True)
                    shutil.move(item_path, os.path.join(target_fold, item))
                    count += 1
            return count
        except ValueError as e:
            logger.warning(f"Blocked unsafe path in organize_host_directory: {e}")
            return 0
        except Exception as e:
            logger.error(f"Failed to organize {directory}: {e}")
            return 0

file_service = FileService()
