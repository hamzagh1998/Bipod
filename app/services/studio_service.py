import base64
import contextlib
import os
import shutil
import uuid
from typing import List, Optional

from sqlalchemy import delete, desc, select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.logger import get_logger
from app.db.database import AsyncSessionLocal
from app.db.models import StudioImage, StudioProject

logger = get_logger("bipod.services.studio")


class StudioService:
    def _project_dir(self, user_id: int, project_id: str) -> str:
        return os.path.join(settings.GENERATED_DIR, "studio", str(user_id), project_id)

    def _relative_url(self, relative_path: str) -> str:
        return f"/generated/{relative_path.replace(os.sep, '/')}"

    def _extract_dimensions(self, metadata: Optional[dict]) -> tuple[Optional[int], Optional[int]]:
        size_value = None
        if metadata:
            size_value = metadata.get("actual_size") or metadata.get("requested_size")
        if not size_value or "x" not in str(size_value):
            return None, None

        try:
            width_str, height_str = str(size_value).lower().split("x", 1)
            return int(width_str.strip()), int(height_str.strip())
        except ValueError:
            return None, None

    async def create_project(self, user_id: int, title: str) -> StudioProject:
        project_id = str(uuid.uuid4())
        async with AsyncSessionLocal() as session:
            project = StudioProject(id=project_id, user_id=user_id, title=title.strip() or "New Project")
            session.add(project)
            await session.commit()
            await session.refresh(project)
            return project

    async def list_projects(self, user_id: int) -> List[dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(StudioProject)
                .where(StudioProject.user_id == user_id)
                .options(selectinload(StudioProject.images))
                .order_by(desc(StudioProject.created_at))
            )
            projects = result.scalars().all()

        response = []
        for project in projects:
            images = sorted(project.images, key=lambda img: img.created_at, reverse=True)
            cover_url = self._relative_url(images[0].relative_path) if images else None
            response.append(
                {
                    "id": project.id,
                    "title": project.title,
                    "created_at": project.created_at,
                    "image_count": len(images),
                    "cover_image_url": cover_url,
                }
            )
        return response

    async def get_project(self, project_id: str, user_id: int) -> Optional[StudioProject]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(StudioProject)
                .where(StudioProject.id == project_id, StudioProject.user_id == user_id)
            )
            return result.scalar_one_or_none()

    async def delete_project(self, project_id: str, user_id: int) -> bool:
        project = await self.get_project(project_id, user_id)
        if not project:
            return False

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(StudioImage)
                .join(StudioProject)
                .where(StudioProject.id == project_id, StudioProject.user_id == user_id)
            )
            images = result.scalars().all()

        for image in images:
            file_path = os.path.join(settings.GENERATED_DIR, image.relative_path)
            with contextlib.suppress(FileNotFoundError):
                os.remove(file_path)

        project_dir = self._project_dir(user_id, project_id)
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)

        async with AsyncSessionLocal() as session:
            await session.execute(delete(StudioImage).where(StudioImage.project_id == project_id))
            await session.execute(
                delete(StudioProject).where(
                    StudioProject.id == project_id,
                    StudioProject.user_id == user_id,
                )
            )
            await session.commit()
        return True

    async def list_project_images(self, project_id: str, user_id: int) -> List[dict]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(StudioImage)
                .join(StudioProject)
                .where(StudioProject.id == project_id, StudioProject.user_id == user_id)
                .order_by(desc(StudioImage.created_at))
            )
            images = result.scalars().all()

        return [self._serialize_image(image) for image in images]

    async def add_project_image(
        self,
        project_id: str,
        user_id: int,
        image_base64: str,
        mime_type: str,
        file_extension: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        project = await self.get_project(project_id, user_id)
        if not project:
            raise ValueError("Project not found")

        raw = base64.b64decode(image_base64)
        width, height = self._extract_dimensions(metadata)

        image_id = str(uuid.uuid4())
        safe_extension = (file_extension or "jpg").lower()
        filename = f"{image_id}.{safe_extension}"
        relative_path = os.path.join("studio", str(user_id), project_id, filename)
        absolute_path = os.path.join(settings.GENERATED_DIR, relative_path)
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

        with open(absolute_path, "wb") as f:
            f.write(raw)

        async with AsyncSessionLocal() as session:
            studio_image = StudioImage(
                id=image_id,
                project_id=project_id,
                filename=filename,
                relative_path=relative_path,
                mime_type=mime_type,
                file_extension=safe_extension,
                width=width,
                height=height,
                metadata_json=metadata,
            )
            session.add(studio_image)
            await session.commit()
            await session.refresh(studio_image)
            return self._serialize_image(studio_image)

    async def delete_project_image(self, project_id: str, image_id: str, user_id: int) -> bool:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(StudioImage)
                .join(StudioProject)
                .where(
                    StudioImage.id == image_id,
                    StudioImage.project_id == project_id,
                    StudioProject.user_id == user_id,
                )
            )
            image = result.scalar_one_or_none()
            if not image:
                return False

            file_path = os.path.join(settings.GENERATED_DIR, image.relative_path)
            with contextlib.suppress(FileNotFoundError):
                os.remove(file_path)

            await session.execute(delete(StudioImage).where(StudioImage.id == image_id))
            await session.commit()

        project_dir = self._project_dir(user_id, project_id)
        if os.path.isdir(project_dir) and not os.listdir(project_dir):
            os.rmdir(project_dir)
        return True

    def _serialize_image(self, image: StudioImage) -> dict:
        return {
            "id": image.id,
            "project_id": image.project_id,
            "filename": image.filename,
            "url": self._relative_url(image.relative_path),
            "mime_type": image.mime_type,
            "file_extension": image.file_extension,
            "width": image.width,
            "height": image.height,
            "created_at": image.created_at,
        }
studio_service = StudioService()
