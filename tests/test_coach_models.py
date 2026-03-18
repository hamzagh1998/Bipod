from app.db.database import Base
from app.db.models import CoachMistake, CoachSession, CoachTurn, User


def test_coach_tables_are_registered_on_metadata():
    assert {"coach_sessions", "coach_turns", "coach_mistakes"}.issubset(
        Base.metadata.tables
    )


def test_user_has_coach_relationships():
    assert hasattr(User, "coach_sessions")
    assert hasattr(User, "coach_turns")
    assert hasattr(User, "coach_mistakes")
    assert CoachSession.user.property.mapper.class_ is User
    assert CoachTurn.user.property.mapper.class_ is User
    assert CoachMistake.user.property.mapper.class_ is User
