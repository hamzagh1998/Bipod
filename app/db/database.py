import logging

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

logger = logging.getLogger("bipod.db")

# engine = create_async_engine(settings.DATABASE_URL, echo=False)
# Since DATABASE_URL is "sqlite:///...", we need to ensure it's "sqlite+aiosqlite:///..."
async_db_url = settings.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///")
engine = create_async_engine(async_db_url, echo=False)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

class Base(DeclarativeBase):
    pass


def _table_columns(conn, table_name: str) -> set[str]:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _ensure_sqlite_column(conn, table_name: str, column_name: str, column_definition: str) -> None:
    existing = _table_columns(conn, table_name)
    if column_name in existing:
        return
    conn.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")
    logger.info("Applied SQLite schema patch: added %s.%s", table_name, column_name)


def _apply_sqlite_schema_patches(conn) -> None:
    """
    Lightweight, additive schema patches for persisted SQLite databases.
    Needed because create_all() won't alter existing tables.
    """
    dialect_name = conn.dialect.name
    if dialect_name != "sqlite":
        return

    tables = {
        row[0]
        for row in conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    if "coach_sessions" in tables:
        _ensure_sqlite_column(
            conn,
            "coach_sessions",
            "target_language",
            "target_language VARCHAR NOT NULL DEFAULT 'English'",
        )
        _ensure_sqlite_column(
            conn,
            "coach_sessions",
            "native_language",
            "native_language VARCHAR",
        )
        _ensure_sqlite_column(
            conn,
            "coach_sessions",
            "cefr_level",
            "cefr_level VARCHAR NOT NULL DEFAULT 'A2'",
        )
        _ensure_sqlite_column(
            conn,
            "coach_sessions",
            "audio_retention_opt_in",
            "audio_retention_opt_in BOOLEAN NOT NULL DEFAULT 0",
        )
        _ensure_sqlite_column(
            conn,
            "coach_sessions",
            "voice_profile_id",
            "voice_profile_id VARCHAR",
        )

    if "coach_turns" in tables:
        _ensure_sqlite_column(
            conn,
            "coach_turns",
            "transcript",
            "transcript TEXT NOT NULL DEFAULT ''",
        )
        _ensure_sqlite_column(
            conn,
            "coach_turns",
            "reply",
            "reply TEXT NOT NULL DEFAULT ''",
        )
        _ensure_sqlite_column(
            conn,
            "coach_turns",
            "correction",
            "correction TEXT",
        )
        _ensure_sqlite_column(
            conn,
            "coach_turns",
            "explanation",
            "explanation TEXT",
        )
        _ensure_sqlite_column(
            conn,
            "coach_turns",
            "score",
            "score INTEGER",
        )


async def init_db():
    from app.db import models # Ensure models are loaded
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_apply_sqlite_schema_patches)
