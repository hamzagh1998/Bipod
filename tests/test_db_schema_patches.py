from sqlalchemy import create_engine

from app.db.database import _apply_sqlite_schema_patches


def _columns(conn, table_name: str) -> set[str]:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def test_sqlite_schema_patches_adds_missing_coach_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        # Simulate older schema before coach column expansion.
        conn.exec_driver_sql(
            """
            CREATE TABLE coach_sessions (
                id VARCHAR PRIMARY KEY,
                user_id INTEGER,
                title VARCHAR,
                focus_area VARCHAR,
                model_id VARCHAR,
                status VARCHAR,
                created_at DATETIME,
                updated_at DATETIME
            )
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE coach_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR,
                user_id INTEGER,
                turn_index INTEGER,
                role VARCHAR,
                content TEXT,
                model_id VARCHAR,
                latency_ms INTEGER,
                created_at DATETIME
            )
            """
        )

        _apply_sqlite_schema_patches(conn)
        # Ensure idempotency.
        _apply_sqlite_schema_patches(conn)

        session_cols = _columns(conn, "coach_sessions")
        turn_cols = _columns(conn, "coach_turns")

        assert {"target_language", "native_language", "cefr_level", "audio_retention_opt_in", "voice_profile_id"}.issubset(session_cols)
        assert {"transcript", "reply", "correction", "explanation", "score"}.issubset(turn_cols)

    engine.dispose()
