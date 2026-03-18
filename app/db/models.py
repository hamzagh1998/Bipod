import datetime
from typing import List, Optional
from sqlalchemy import String, DateTime, ForeignKey, Boolean, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from app.db.database import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    conversations: Mapped[List["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    studio_projects: Mapped[List["StudioProject"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    coach_sessions: Mapped[List["CoachSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    coach_turns: Mapped[List["CoachTurn"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    coach_mistakes: Mapped[List["CoachMistake"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True) # Optional for now to avoid breaking existing? Better to be strict if starting fresh.
    title: Mapped[str] = mapped_column(String, default="New Conversation")
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    archive_password_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    
    user: Mapped[Optional["User"]] = relationship(back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String) # user, assistant, system
    content: Mapped[str] = mapped_column(Text)
    attachments: Mapped[Optional[List[dict]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class StudioProject(Base):
    __tablename__ = "studio_projects"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String, default="New Project")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="studio_projects")
    images: Mapped[List["StudioImage"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )


class StudioImage(Base):
    __tablename__ = "studio_images"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("studio_projects.id"), index=True)
    filename: Mapped[str] = mapped_column(String)
    relative_path: Mapped[str] = mapped_column(String, unique=True)
    mime_type: Mapped[str] = mapped_column(String)
    file_extension: Mapped[str] = mapped_column(String)
    width: Mapped[Optional[int]] = mapped_column(nullable=True)
    height: Mapped[Optional[int]] = mapped_column(nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped["StudioProject"] = relationship(back_populates="images")


class CoachSession(Base):
    __tablename__ = "coach_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String, default="Coach Session")
    target_language: Mapped[str] = mapped_column(String, default="English")
    native_language: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cefr_level: Mapped[str] = mapped_column(String, default="A2")
    audio_retention_opt_in: Mapped[bool] = mapped_column(Boolean, default=False)
    focus_area: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user: Mapped["User"] = relationship(back_populates="coach_sessions")
    turns: Mapped[List["CoachTurn"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="CoachTurn.turn_index",
    )


class CoachTurn(Base):
    __tablename__ = "coach_turns"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("coach_sessions.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    turn_index: Mapped[int] = mapped_column(index=True)
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    model_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    transcript: Mapped[str] = mapped_column(Text, default="")
    reply: Mapped[str] = mapped_column(Text, default="")
    correction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    score: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped["CoachSession"] = relationship(back_populates="turns")
    user: Mapped["User"] = relationship(back_populates="coach_turns")
    mistakes: Mapped[List["CoachMistake"]] = relationship(
        back_populates="turn",
        cascade="all, delete-orphan",
        order_by="CoachMistake.created_at",
    )


class CoachMistake(Base):
    __tablename__ = "coach_mistakes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("coach_sessions.id"), index=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("coach_turns.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    category: Mapped[str] = mapped_column(String)
    detail: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String, default="medium")
    suggestion: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped["CoachSession"] = relationship()
    turn: Mapped["CoachTurn"] = relationship(back_populates="mistakes")
    user: Mapped["User"] = relationship(back_populates="coach_mistakes")
