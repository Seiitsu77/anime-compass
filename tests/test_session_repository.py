from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.repositories.session_repository import AnonymousSession, SQLiteSessionRepository, empty_profile


def repository(tmp_path: Path) -> SQLiteSessionRepository:
    return SQLiteSessionRepository(f"sqlite:///{(tmp_path / 'sessions.db').as_posix()}", retention_days=30)


def test_get_missing_session_is_read_only(tmp_path: Path) -> None:
    sessions = repository(tmp_path)

    assert sessions.get("missing-session") == empty_profile()
    with sessions.session_factory() as database:
        assert database.get(AnonymousSession, "missing-session") is None

    sessions.engine.dispose()


def test_cleanup_removes_expired_sessions(tmp_path: Path) -> None:
    sessions = repository(tmp_path)
    sessions.update("expired-session", {"liked_titles": ["Death Note"]})
    with sessions.session_factory() as database:
        record = database.get(AnonymousSession, "expired-session")
        assert record is not None
        record.updated_at = datetime.now(timezone.utc) - timedelta(days=31)
        database.commit()

    assert sessions.cleanup_expired() == 1
    with sessions.session_factory() as database:
        assert database.get(AnonymousSession, "expired-session") is None

    sessions.engine.dispose()


def test_replacing_preferences_keeps_the_agent_conversation_context(tmp_path: Path) -> None:
    """A browser profile sync must not erase what a follow-up refers to."""
    sessions = repository(tmp_path)
    sessions.update(
        "chat-session",
        {
            "liked_titles": ["Death Note"],
            "last_recommendations": ["Death Parade", "Ghost Hunt"],
            "last_recommendation_intent": {"include_genres": ["Supernatural"], "top_k": 2},
        },
    )

    profile = sessions.update("chat-session", {"replace": True, "liked_titles": ["Ghost Hunt"]})

    assert profile["liked_titles"] == ["Ghost Hunt"], "replace still swaps the preferences"
    assert profile["last_recommendations"] == ["Death Parade", "Ghost Hunt"]
    assert profile["last_recommendation_intent"] == {"include_genres": ["Supernatural"], "top_k": 2}

    assert sessions.update("chat-session", {"reset": True}) == empty_profile()
    sessions.engine.dispose()
