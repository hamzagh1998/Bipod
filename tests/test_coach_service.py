import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.services.coach_service as coach_service_module
from app.db.database import Base
from app.db.models import User
from app.services.coach_service import coach_service


class _AsyncSessionAdapter:
    def __init__(self, session):
        self._session = session

    async def execute(self, *args, **kwargs):
        return self._session.execute(*args, **kwargs)

    def add(self, obj):
        self._session.add(obj)

    async def commit(self):
        self._session.commit()

    async def refresh(self, obj):
        self._session.refresh(obj)

    async def flush(self):
        self._session.flush()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._session.close()


class _AsyncSessionFactory:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def __call__(self):
        return _AsyncSessionAdapter(self._session_factory())


@pytest.fixture
def coach_db(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'coach.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(coach_service_module, "AsyncSessionLocal", _AsyncSessionFactory(session_factory))
    yield engine
    engine.dispose()


async def _seed_user(username: str = "coach-user") -> User:
    async with coach_service_module.AsyncSessionLocal() as session:
        user = User(username=username, hashed_password="hash")
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


def test_coach_session_turn_mistake_flow(coach_db):
    async def scenario():
        user = await _seed_user()

        session = await coach_service.create_session(
            user_id=user.id,
            title="Pair Programming",
            target_language="English",
            native_language="Arabic",
            cefr_level="B1",
            audio_retention_opt_in=False,
            focus_area="debugging",
            model_id="coach-smart",
        )
        assert session.user_id == user.id
        assert session.title == "Pair Programming"
        assert session.target_language == "English"
        assert session.cefr_level == "B1"
        assert session.focus_area == "debugging"

        first_turn = await coach_service.save_turn_with_mistakes(
            session.id,
            user.id,
            transcript="I try iterative approach.",
            reply="Try an iterative approach.",
            score=81,
            correction="Use 'an iterative approach'.",
            explanation="Target: fix article usage. Native: add 'an' before iterative approach.",
            model_id="coach-smart",
            latency_ms=180,
            mistakes=[
                {
                    "category": "logic",
                    "detail": "Skipped the edge case for empty input.",
                    "severity": "high",
                    "suggestion": "Handle the empty list before the main loop.",
                },
                {
                    "category": "style",
                    "detail": "The explanation was too verbose.",
                },
            ],
        )
        assert first_turn is not None
        assert first_turn.turn_index == 1
        assert first_turn.reply == "Try an iterative approach."
        assert first_turn.score == 81
        assert len(first_turn.mistakes) == 2

        second_turn = await coach_service.save_turn_with_mistakes(
            session.id,
            user.id,
            transcript="What if the input is empty?",
            reply="Great question. Handle empty input before the loop.",
            score=88,
            model_id="coach-light",
            latency_ms=55,
            mistakes=[
                {
                    "category": "coverage",
                    "detail": "Did not address the empty-input branch.",
                }
            ],
        )
        assert second_turn is not None
        assert second_turn.turn_index == 2
        assert len(second_turn.mistakes) == 1

        loaded_session = await coach_service.get_session(session.id, user.id)
        assert loaded_session is not None
        assert len(loaded_session.turns) == 2

        sessions = await coach_service.list_sessions(user.id)
        assert len(sessions) == 1
        assert sessions[0]["target_language"] == "English"
        assert sessions[0]["native_language"] == "Arabic"
        assert sessions[0]["cefr_level"] == "B1"
        assert sessions[0]["turn_count"] == 2
        assert sessions[0]["mistake_count"] == 3

        turns = await coach_service.list_turns(session.id, user.id)
        assert [turn.turn_index for turn in turns] == [1, 2]
        assert turns[0].transcript == "I try iterative approach."
        assert turns[1].reply == "Great question. Handle empty input before the loop."
        assert [len(turn.mistakes) for turn in turns] == [2, 1]

        mistakes = await coach_service.list_mistakes(session.id, user.id)
        assert [mistake.category for mistake in mistakes] == ["logic", "style", "coverage"]

        summary = await coach_service.progress_summary(user.id)
        assert summary == {
            "user_id": user.id,
            "total_sessions": 1,
            "total_turns": 2,
            "total_mistakes": 3,
            "mistake_counts_by_category": {
                "logic": 1,
                "style": 1,
                "coverage": 1,
            },
            "turn_counts_by_model": {
                "coach-smart": 1,
                "coach-light": 1,
            },
            "active_sessions": 1,
            "latest_session_id": session.id,
            "latest_session_title": "Pair Programming",
            "latest_session_turns": 2,
        }

        session_progress = await coach_service.progress(user.id, session.id)
        assert session_progress == {
            "session_id": session.id,
            "turn_count": 2,
            "mistake_count": 3,
            "average_score": 84.5,
            "latest_score": 88,
        }

    asyncio.run(scenario())


def test_coach_save_turn_rejects_unknown_session(coach_db):
    async def scenario():
        user = await _seed_user("other-user")

        result = await coach_service.save_turn_with_mistakes(
            "missing-session",
            user.id,
            transcript="No session here.",
            reply="No session here.",
            score=0,
            mistakes=[{"category": "logic", "detail": "Missing session"}],
        )
        assert result is None

    asyncio.run(scenario())


def test_model_selection_uses_quality_order_and_latency_hooks():
    order = coach_service.get_quality_first_model_order(
        ["quality", "balanced", "fast"]
    )
    assert order == ["quality", "balanced", "fast"]

    selected = coach_service.select_model(
        candidate_order=["quality", "balanced", "fast"],
        latency_budget_ms=100.0,
        latency_probe=lambda model: {"quality": 240.0, "balanced": 125.0, "fast": 35.0}[model],
    )
    assert selected == "fast"

    fallback_selected = coach_service.select_model(
        candidate_order=["quality", "balanced"],
        latency_budget_ms=10.0,
        latency_probe=lambda model: 200.0,
        latency_fallback=lambda candidates: candidates[-1],
    )
    assert fallback_selected == "balanced"
