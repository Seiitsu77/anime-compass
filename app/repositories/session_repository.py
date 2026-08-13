from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, delete, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

SESSION_LIST_FIELDS = {
    "liked_titles",
    "disliked_titles",
    "seen_titles",
    "excluded_titles",
    "preferred_genres",
    "excluded_genres",
    "preferred_studios",
    "preferred_staff",
    "preferred_characters",
    "preferred_voice_actors",
    "previous_reference_titles",
}
SESSION_CONTEXT_FIELDS = {"last_recommendation_intent", "last_recommendations"}


class Base(DeclarativeBase):
    pass


class AnonymousSession(Base):
    __tablename__ = "anonymous_sessions"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        index=True,
    )
    profile_json: Mapped[str] = mapped_column(Text, default="{}")
    events: Mapped[list[SessionEvent]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class SessionEvent(Base):
    __tablename__ = "session_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("anonymous_sessions.session_id", ondelete="CASCADE"),
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    event_category: Mapped[str] = mapped_column(String(32), index=True)
    event_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    session: Mapped[AnonymousSession] = relationship(back_populates="events")


def empty_profile() -> dict[str, Any]:
    return {field: [] for field in SESSION_LIST_FIELDS} | {
        "temporary_ratings": {},
        "last_recommendation_intent": {},
        "last_recommendations": [],
    }


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def merge_profiles(*profiles: dict[str, Any]) -> dict[str, Any]:
    merged = empty_profile()
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        for field in SESSION_LIST_FIELDS:
            existing = [str(value) for value in merged.get(field, []) if value]
            seen = {value.casefold() for value in existing}
            for value in as_list(profile.get(field)):
                text_value = str(value).strip()
                if text_value and text_value.casefold() not in seen:
                    existing.append(text_value)
                    seen.add(text_value.casefold())
            merged[field] = existing
        if isinstance(profile.get("temporary_ratings"), dict):
            merged["temporary_ratings"].update(profile["temporary_ratings"])
        for field in SESSION_CONTEXT_FIELDS:
            if field in profile:
                merged[field] = json.loads(json.dumps(profile[field]))
    return merged


class SQLiteSessionRepository:
    def __init__(self, database_url: str, retention_days: int = 30):
        if database_url.startswith("sqlite:///"):
            raw_path = database_url.removeprefix("sqlite:///")
            path = Path(raw_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{path.as_posix()}"
        self.engine = create_engine(
            database_url,
            future=True,
            connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
        )
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        self.retention_days = retention_days
        Base.metadata.create_all(self.engine)

    def get(self, session_id: str | None) -> dict[str, Any]:
        if not session_id:
            return empty_profile()
        with self.session_factory() as database:
            record = database.get(AnonymousSession, session_id)
            if record is None:
                return empty_profile()
            return self._decode_profile(record.profile_json)

    def update(self, session_id: str | None, patch: dict[str, Any]) -> dict[str, Any]:
        if not session_id:
            return empty_profile()
        if patch.get("reset"):
            return self.reset(session_id)
        with self.session_factory() as database:
            record = database.get(AnonymousSession, session_id)
            if record is None:
                record = AnonymousSession(session_id=session_id, profile_json=json.dumps(empty_profile()))
                database.add(record)
            current = empty_profile() if patch.get("replace") else self._decode_profile(record.profile_json)
            profile = self._apply_patch(current, patch)
            record.profile_json = json.dumps(profile, ensure_ascii=False, separators=(",", ":"))
            record.updated_at = datetime.now(timezone.utc)
            database.commit()
            return profile

    def reset(self, session_id: str) -> dict[str, Any]:
        profile = empty_profile()
        with self.session_factory() as database:
            record = database.get(AnonymousSession, session_id)
            if record is None:
                record = AnonymousSession(session_id=session_id)
                database.add(record)
            record.profile_json = json.dumps(profile, separators=(",", ":"))
            record.updated_at = datetime.now(timezone.utc)
            database.execute(delete(SessionEvent).where(SessionEvent.session_id == session_id))
            database.commit()
        return profile

    def delete(self, session_id: str) -> bool:
        with self.session_factory() as database:
            record = database.get(AnonymousSession, session_id)
            if record is None:
                return False
            database.delete(record)
            database.commit()
            return True

    def log_event(
        self,
        session_id: str | None,
        category: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not session_id:
            return
        safe_payload = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))[:20_000]
        with self.session_factory() as database:
            record = database.get(AnonymousSession, session_id)
            if record is None:
                record = AnonymousSession(session_id=session_id, profile_json=json.dumps(empty_profile()))
                database.add(record)
                database.flush()
            database.add(
                SessionEvent(
                    session_id=session_id,
                    event_category=category[:32],
                    event_type=event_type[:64],
                    payload_json=safe_payload,
                )
            )
            database.commit()

    def cleanup_expired(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        with self.session_factory() as database:
            records = database.scalars(select(AnonymousSession).where(AnonymousSession.updated_at < cutoff)).all()
            for record in records:
                database.delete(record)
            database.commit()
            return len(records)

    def health(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @staticmethod
    def _decode_profile(value: str) -> dict[str, Any]:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        return merge_profiles(parsed)

    @staticmethod
    def _apply_patch(profile: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        for field in SESSION_LIST_FIELDS:
            values = patch.get(field)
            if values is None:
                continue
            existing = [str(value) for value in profile.get(field, []) if value]
            seen = {value.casefold() for value in existing}
            for value in as_list(values):
                text_value = str(value).strip()
                if text_value and text_value.casefold() not in seen:
                    existing.append(text_value)
                    seen.add(text_value.casefold())
            profile[field] = existing
        if patch.get("watched_titles"):
            profile["seen_titles"] = list(
                dict.fromkeys([*profile.get("seen_titles", []), *as_list(patch["watched_titles"])])
            )
        if isinstance(patch.get("temporary_ratings"), dict):
            profile.setdefault("temporary_ratings", {}).update(patch["temporary_ratings"])
        for field in SESSION_CONTEXT_FIELDS:
            if field in patch:
                profile[field] = json.loads(json.dumps(patch[field]))
        return profile
