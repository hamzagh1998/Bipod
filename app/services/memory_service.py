import hashlib
import uuid
from typing import List, Optional
from sqlalchemy import select, update, delete
from app.db.database import AsyncSessionLocal
from app.db.models import Conversation, Message
from app.core.logger import get_logger

logger = get_logger("bipod.services.memory")

class MemoryService:
    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    async def create_conversation(self, title: str = "New Conversation") -> str:
        conv_id = str(uuid.uuid4())
        async with AsyncSessionLocal() as session:
            new_conv = Conversation(id=conv_id, title=title)
            session.add(new_conv)
            await session.commit()
            return conv_id

    async def get_conversations(self) -> List[Conversation]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Conversation).order_by(Conversation.created_at.desc()))
            return result.scalars().all()

    async def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Conversation).where(Conversation.id == conv_id))
            return result.scalar_one_or_none()

    async def update_conversation(self, conv_id: str, title: Optional[str] = None, is_archived: Optional[bool] = None, password: Optional[str] = None):
        async with AsyncSessionLocal() as session:
            values = {}
            if title is not None:
                values["title"] = title
            if is_archived is not None:
                values["is_archived"] = is_archived
            if password:
                values["archive_password_hash"] = self._hash_password(password)
            
            if values:
                await session.execute(update(Conversation).where(Conversation.id == conv_id).values(**values))
                await session.commit()

    async def verify_archive_password(self, conv_id: str, password: str) -> bool:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Conversation.archive_password_hash).where(Conversation.id == conv_id))
            stored_hash = result.scalar_one_or_none()
            if not stored_hash:
                return True # No password set?
            return stored_hash == self._hash_password(password)

    async def delete_conversation(self, conv_id: str):
        async with AsyncSessionLocal() as session:
            await session.execute(delete(Conversation).where(Conversation.id == conv_id))
            await session.commit()

    async def add_message(self, conv_id: str, role: str, content: str):
        async with AsyncSessionLocal() as session:
            new_msg = Message(conversation_id=conv_id, role=role, content=content)
            session.add(new_msg)
            await session.commit()

    async def get_messages(self, conv_id: str) -> List[Message]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Message).where(Message.conversation_id == conv_id).order_by(Message.created_at.asc()))
            return result.scalars().all()

memory_service = MemoryService()
