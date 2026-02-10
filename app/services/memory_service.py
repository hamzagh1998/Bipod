import hashlib
import uuid
from typing import List, Optional
from sqlalchemy import select, update, delete
from app.db.database import AsyncSessionLocal
from app.db.models import Conversation, Message, User
from app.core.logger import get_logger

# Deferred import to avoid circular dependency
# vector_service is imported inside delete_conversation()

logger = get_logger("bipod.services.memory")

class MemoryService:
    def _hash_password(self, password: str) -> str:
        # Note: This is for ARCHIVE password (sha256). 
        # User auth uses bcrypt via auth_service.
        return hashlib.sha256(password.encode()).hexdigest()

    # --- User Operations ---
    async def get_user_by_username(self, username: str) -> Optional[User]:
        logger.info(f"Fetching user by username: {username}")
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(User).where(User.username == username))
            user = result.scalar_one_or_none()
            if not user:
                logger.warning(f"User not found with username: {username}")
            return user

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(User).where(User.id == user_id))
            return result.scalar_one_or_none()

    async def create_user(self, username: str, hashed_password: str) -> User:
        async with AsyncSessionLocal() as session:
            new_user = User(username=username, hashed_password=hashed_password)
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)
            return new_user

    # --- Conversation Operations ---
    async def create_conversation(self, user_id: int, title: str = "New Conversation") -> str:
        conv_id = str(uuid.uuid4())
        logger.info(f"Creating new conversation: {conv_id} for user_id: {user_id}")
        async with AsyncSessionLocal() as session:
            new_conv = Conversation(id=conv_id, title=title, user_id=user_id)
            session.add(new_conv)
            await session.commit()
            return conv_id

    async def get_conversations(self, user_id: int) -> List[Conversation]:
        logger.info(f"Listing conversations for user_id: {user_id}")
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.created_at.desc())
            )
            convs = result.scalars().all()
            logger.info(f"Found {len(convs)} conversations for user: {user_id}")
            return convs

    async def get_conversation(self, conv_id: str, user_id: int) -> Optional[Conversation]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.id == conv_id, Conversation.user_id == user_id)
            )
            return result.scalar_one_or_none()

    async def update_conversation(self, conv_id: str, user_id: int, title: Optional[str] = None, is_archived: Optional[bool] = None, password: Optional[str] = None):
        async with AsyncSessionLocal() as session:
            values = {}
            if title is not None:
                values["title"] = title
            if is_archived is not None:
                values["is_archived"] = is_archived
            if password:
                values["archive_password_hash"] = self._hash_password(password)
            
            if values:
                await session.execute(
                    update(Conversation)
                    .where(Conversation.id == conv_id, Conversation.user_id == user_id)
                    .values(**values)
                )
                await session.commit()

    async def verify_archive_password(self, conv_id: str, user_id: int, password: str) -> bool:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Conversation.archive_password_hash)
                .where(Conversation.id == conv_id, Conversation.user_id == user_id)
            )
            stored_hash = result.scalar_one_or_none()
            if not stored_hash:
                return True 
            return stored_hash == self._hash_password(password)

    async def delete_conversation(self, conv_id: str, user_id: int):
        """Deletes a conversation and ALL associated data (messages + vector memories)."""
        # 1. First verify the conversation belongs to this user
        async with AsyncSessionLocal() as session:
            conv_check = await session.execute(
                select(Conversation.id).where(
                    Conversation.id == conv_id, Conversation.user_id == user_id
                )
            )
            if not conv_check.scalar_one_or_none():
                logger.warning(f"Conversation {conv_id} not found for user {user_id}, skipping delete.")
                return

        # 2. Delete all messages explicitly (belt + suspenders with cascade)
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(Message).where(Message.conversation_id == conv_id)
            )
            await session.commit()
            logger.info(f"Deleted all messages for conversation {conv_id}")

        # 3. Delete the conversation itself
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(Conversation)
                .where(Conversation.id == conv_id, Conversation.user_id == user_id)
            )
            await session.commit()
            logger.info(f"Deleted conversation {conv_id} from SQLite")

        # 4. Clean up vector memories (deferred import to avoid circular dependency)
        try:
            from app.services.vector_service import vector_service
            await vector_service.delete_conversation_memories(conv_id)
            logger.info(f"Cleaned up vector memories for conversation {conv_id}")
        except Exception as e:
            logger.error(f"Failed to clean up vector memories for {conv_id}: {e}")

    async def add_message(self, conv_id: str, role: str, content: str, attachments: Optional[List[dict]] = None) -> Message:
        # We don't check user_id here as the chat flow checks it before calling add_message
        async with AsyncSessionLocal() as session:
            new_msg = Message(conversation_id=conv_id, role=role, content=content, attachments=attachments)
            session.add(new_msg)
            await session.commit()
            await session.refresh(new_msg)
            return new_msg

    async def get_messages(self, conv_id: str, user_id: int) -> List[Message]:
        async with AsyncSessionLocal() as session:
            # First check if conversation belongs to user
            conv_check = await session.execute(
                select(Conversation.id).where(Conversation.id == conv_id, Conversation.user_id == user_id)
            )
            if not conv_check.scalar_one_or_none():
                return []
                
            result = await session.execute(
                select(Message)
                .where(Message.conversation_id == conv_id)
                .order_by(Message.created_at.asc())
            )
            return result.scalars().all()

memory_service = MemoryService()
