import os

class FileService:
    """Service for managing local documents."""
    def list_files(self, directory: str):
        return os.listdir(directory)

file_service = FileService()
