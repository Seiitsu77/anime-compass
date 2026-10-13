"""An in-memory stand-in for the session repository, for tests of the agent.

It applies patches with the SQLite repository's own code, so the two cannot
drift apart; the only thing it leaves out is the database.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.repositories.session_repository import SESSION_CONTEXT_FIELDS, SQLiteSessionRepository, empty_profile


class SessionStore:
    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, Any]] = {}

    def get(self, session_id: str | None) -> dict[str, Any]:
        if not session_id:
            return empty_profile()
        return deepcopy(self._profiles.get(session_id, empty_profile()))

    def update(self, session_id: str | None, patch: dict[str, Any]) -> dict[str, Any]:
        if not session_id:
            return empty_profile()
        if patch.get("reset"):
            self._profiles[session_id] = empty_profile()
            return self.get(session_id)
        stored = self.get(session_id)
        if patch.get("replace"):
            current = empty_profile()
            for name in SESSION_CONTEXT_FIELDS:
                current[name] = stored[name]
        else:
            current = stored
        self._profiles[session_id] = SQLiteSessionRepository._apply_patch(current, patch)
        return self.get(session_id)
