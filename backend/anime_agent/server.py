from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .agent import AnimeAgent
from .data_pipeline import load_or_create_catalog
from .recommender import AnimeRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
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


class ValidationError(ValueError):
    pass


class SessionStore:
    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, Any]] = {}

    def get(self, session_id: str | None) -> dict[str, Any]:
        if not session_id:
            return self._empty_profile()
        return json.loads(json.dumps(self._profiles.setdefault(session_id, self._empty_profile())))

    def reset(self, session_id: str) -> dict[str, Any]:
        self._profiles[session_id] = self._empty_profile()
        return self.get(session_id)

    def update(self, session_id: str | None, patch: dict[str, Any]) -> dict[str, Any]:
        if not session_id:
            return self._empty_profile()
        if patch.get("reset"):
            return self.reset(session_id)

        profile = (
            self._empty_profile()
            if patch.get("replace")
            else self._profiles.setdefault(session_id, self._empty_profile())
        )
        for field in SESSION_LIST_FIELDS:
            values = patch.get(field)
            if values is None:
                continue
            existing = [str(value) for value in profile.get(field, []) if value]
            seen = {value.casefold() for value in existing}
            for value in as_list(values):
                text = str(value).strip()
                if text and text.casefold() not in seen:
                    existing.append(text)
                    seen.add(text.casefold())
            profile[field] = existing

        ratings = patch.get("temporary_ratings")
        if isinstance(ratings, dict):
            profile.setdefault("temporary_ratings", {}).update(ratings)

        for field in SESSION_CONTEXT_FIELDS:
            if field in patch:
                profile[field] = json.loads(json.dumps(patch[field]))

        self._profiles[session_id] = profile
        return self.get(session_id)

    def _empty_profile(self) -> dict[str, Any]:
        return {field: [] for field in SESSION_LIST_FIELDS} | {
            "temporary_ratings": {},
            "last_recommendation_intent": {},
            "last_recommendations": [],
        }


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def merge_profiles(*profiles: dict[str, Any]) -> dict[str, Any]:
    merged = SessionStore()._empty_profile()
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        for field in SESSION_LIST_FIELDS:
            merged[field] = list(
                dict.fromkeys(
                    [str(value) for value in merged.get(field, []) if value]
                    + [str(value) for value in as_list(profile.get(field)) if value]
                )
            )
        if isinstance(profile.get("temporary_ratings"), dict):
            merged["temporary_ratings"].update(profile["temporary_ratings"])
        for field in SESSION_CONTEXT_FIELDS:
            if field in profile:
                merged[field] = json.loads(json.dumps(profile[field]))
    return merged


class AppState:
    def __init__(self, project_root: Path):
        catalog = load_or_create_catalog(project_root)
        self.recommender = AnimeRecommender(catalog)
        self.sessions = SessionStore()
        self.agent = AnimeAgent(
            self.recommender,
            get_session_profile=self.sessions.get,
            update_session_preferences=self.sessions.update,
        )


class AnimeRequestHandler(BaseHTTPRequestHandler):
    state: AppState
    frontend_root: Path

    server_version = "AnimeCompass/0.1"

    def parse_limit(self, raw_value: Any, default: int) -> int:
        catalog_size = max(1, len(self.state.recommender.catalog))
        if raw_value in (None, ""):
            return default
        if str(raw_value).casefold() == "all":
            return catalog_size
        return max(1, min(int(raw_value), catalog_size))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/api/health":
                self.send_json(
                    {
                        "ok": True,
                        "catalog": self.state.recommender.meta(),
                        "agent": self.state.agent.status(),
                    }
                )
                return

            if parsed.path == "/api/meta":
                self.send_json(self.state.recommender.meta())
                return

            if parsed.path == "/api/session/preferences":
                params = parse_qs(parsed.query)
                session_id = params.get("session_id", [""])[0]
                self.send_json({"session_id": session_id, "profile": self.state.sessions.get(session_id)})
                return

            if parsed.path == "/api/anime/search":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                limit = self.parse_limit(params.get("limit", ["10"])[0], default=10)
                self.send_json(
                    {
                        "results": self.state.recommender.search(
                            query,
                            limit=limit,
                            genres=[str(value) for value in params.get("genres", [])],
                            media_type=params.get("media_type", [""])[0] or None,
                            min_score=float(params["min_score"][0]) if params.get("min_score", [""])[0] else None,
                            max_episodes=int(params["max_episodes"][0])
                            if params.get("max_episodes", [""])[0]
                            else None,
                        )
                    }
                )
                return

            if parsed.path.startswith("/api/anime/"):
                anime_id = int(parsed.path.rsplit("/", 1)[-1])
                details = self.state.recommender.details(anime_id)
                if details is None:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Anime not found")
                    return
                self.send_json({"result": details})
                return

            self.serve_static(parsed.path)

        except Exception as exc:  # Keep local dev errors visible to the browser.
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            body = self.read_json()

            if parsed.path == "/api/recommend":
                started = time.perf_counter()
                request = self.parse_recommendation_request(body)
                session_profile = merge_profiles(
                    self.state.sessions.get(request["session_id"]),
                    body.get("session_profile") or {},
                )
                diagnostics: dict[str, Any] = {}
                results = self.state.recommender.recommend(
                    reference_titles=request["reference_titles"],
                    liked_ids=request["liked_ids"],
                    liked_titles=request["liked_titles"],
                    excluded_ids=request["excluded_ids"],
                    excluded_titles=request["excluded_titles"],
                    seen_titles=request["seen_titles"],
                    include_genres=request["include_genres"],
                    exclude_genres=request["exclude_genres"],
                    formats=request["formats"],
                    min_score=request["min_score"],
                    min_year=request["min_year"],
                    max_year=request["max_year"],
                    max_episodes=request["max_episodes"],
                    query=request["query"],
                    free_text_preferences=request["free_text_preferences"],
                    preferred_studios=request["preferred_studios"],
                    preferred_staff=request["preferred_staff"],
                    preferred_characters=request["preferred_characters"],
                    required_voice_actors=request["required_voice_actors"],
                    preferred_voice_actors=request["preferred_voice_actors"],
                    novelty_preference=request["novelty_preference"],
                    exclude_related_series=request["exclude_related_series"],
                    one_per_series=request["one_per_series"],
                    session_profile=session_profile,
                    diversity_strength=request["diversity_strength"],
                    weights=request["weights"],
                    limit=request["top_k"],
                    diagnostics=diagnostics,
                )
                resolved_titles = self.state.recommender.resolve_title_details(
                    request["reference_titles"] + request["liked_titles"]
                )
                response = {
                    "query": request["public_query"],
                    "resolved_titles": resolved_titles,
                    "recommendations": results,
                    "results": results,
                    "model_info": self.state.recommender.model_info(),
                    "diagnostics": diagnostics,
                    "timing_ms": round((time.perf_counter() - started) * 1000, 2),
                }
                if not results:
                    response["message"] = "No catalog titles matched all hard constraints."
                self.send_json(response)
                return

            if parsed.path == "/api/agent":
                message = str(body.get("message") or "")
                history = body.get("history") or []
                session_id = str(body.get("session_id") or "")
                if not message.strip():
                    self.send_error_json(HTTPStatus.BAD_REQUEST, "Message is required")
                    return
                self.send_json(
                    self.state.agent.respond(
                        message,
                        history=history,
                        session_id=session_id,
                        debug=bool(body.get("debug")),
                    )
                )
                return

            if parsed.path == "/api/session/preferences":
                session_id = str(body.get("session_id") or "")
                if not session_id.strip():
                    self.send_error_json(HTTPStatus.BAD_REQUEST, "session_id is required")
                    return
                profile = self.state.sessions.update(session_id, body)
                self.send_json({"session_id": session_id, "profile": profile})
                return

            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")

        except ValidationError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def parse_recommendation_request(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ValidationError("Request body must be a JSON object")

        include_genres = self.string_list(body, "include_genres")
        if not include_genres:
            include_genres = self.string_list(body, "genres")

        formats = self.string_list(body, "formats")
        media_type = str(body.get("media_type") or "").strip()
        if media_type:
            formats.append(media_type)
        formats = list(dict.fromkeys(formats))

        top_k_raw = body.get("top_k", body.get("limit", 12))
        top_k = self.parse_limit(top_k_raw, default=12)
        public_query = {
            "reference_titles": self.string_list(body, "reference_titles"),
            "liked_titles": self.string_list(body, "liked_titles"),
            "include_genres": include_genres,
            "exclude_genres": self.string_list(body, "exclude_genres"),
            "formats": formats,
            "min_score": self.optional_float(body, "min_score", minimum=0.0, maximum=10.0),
            "min_year": self.optional_int(body, "min_year", minimum=1900, maximum=2100),
            "max_year": self.optional_int(body, "max_year", minimum=1900, maximum=2100),
            "max_episodes": self.optional_int(body, "max_episodes", minimum=1),
            "excluded_titles": self.string_list(body, "excluded_titles"),
            "seen_titles": self.string_list(body, "seen_titles"),
            "preferred_studios": self.string_list(body, "preferred_studios"),
            "preferred_staff": self.string_list(body, "preferred_staff"),
            "preferred_characters": self.string_list(body, "preferred_characters"),
            "required_voice_actors": self.string_list(body, "required_voice_actors"),
            "preferred_voice_actors": self.string_list(body, "preferred_voice_actors"),
            "free_text_preferences": str(body.get("free_text_preferences") or "").strip(),
            "novelty_preference": str(body.get("novelty_preference") or "neutral").strip().casefold(),
            "exclude_related_series": bool(body.get("exclude_related_series", True)),
            "one_per_series": bool(body.get("one_per_series")),
            "top_k": top_k,
            "session_id": str(body.get("session_id") or ""),
        }

        if (
            public_query["min_year"] is not None
            and public_query["max_year"] is not None
            and public_query["min_year"] > public_query["max_year"]
        ):
            raise ValidationError("min_year cannot be greater than max_year")
        if public_query["novelty_preference"] not in {"neutral", "less_famous", "mainstream"}:
            raise ValidationError("novelty_preference must be neutral, less_famous, or mainstream")

        weights = body.get("weights") or None
        if weights is not None and not isinstance(weights, dict):
            raise ValidationError("weights must be an object keyed by channel name")

        diversity_strength = self.optional_float(
            body,
            "diversity_strength",
            minimum=0.0,
            maximum=1.0,
            default=0.12,
        )

        return {
            **public_query,
            "public_query": public_query,
            "liked_ids": self.int_list(body, "liked_ids"),
            "excluded_ids": self.int_list(body, "excluded_ids"),
            "query": str(body.get("query") or "").strip() or None,
            "weights": weights,
            "diversity_strength": diversity_strength,
        }

    def string_list(self, body: dict[str, Any], field: str) -> list[str]:
        values = body.get(field)
        if values in (None, ""):
            return []
        if not isinstance(values, list):
            values = [values]
        result = []
        for value in values:
            text = str(value).strip()
            if text:
                result.append(text)
        return result

    def int_list(self, body: dict[str, Any], field: str) -> list[int]:
        result = []
        for value in as_list(body.get(field)):
            if value in (None, ""):
                continue
            try:
                result.append(int(value))
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{field} must contain integers") from exc
        return result

    def optional_int(
        self,
        body: dict[str, Any],
        field: str,
        minimum: int | None = None,
        maximum: int | None = None,
        default: int | None = None,
    ) -> int | None:
        value = body.get(field)
        if value in (None, ""):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{field} must be an integer") from exc
        if minimum is not None and parsed < minimum:
            raise ValidationError(f"{field} must be at least {minimum}")
        if maximum is not None and parsed > maximum:
            raise ValidationError(f"{field} must be at most {maximum}")
        return parsed

    def optional_float(
        self,
        body: dict[str, Any],
        field: str,
        minimum: float | None = None,
        maximum: float | None = None,
        default: float | None = None,
    ) -> float | None:
        value = body.get(field)
        if value in (None, ""):
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{field} must be a number") from exc
        if minimum is not None and parsed < minimum:
            raise ValidationError(f"{field} must be at least {minimum:g}")
        if maximum is not None and parsed > maximum:
            raise ValidationError(f"{field} must be at most {maximum:g}")
        return parsed

    def serve_static(self, url_path: str) -> None:
        if url_path in {"", "/"}:
            relative = Path("index.html")
        else:
            relative = Path(unquote(url_path.lstrip("/")))

        target = (self.frontend_root / relative).resolve()
        root = self.frontend_root.resolve()

        if root not in target.parents and target != root:
            self.send_error_json(HTTPStatus.FORBIDDEN, "Invalid static path")
            return
        if not target.exists() or not target.is_file():
            self.send_error_json(HTTPStatus.NOT_FOUND, "File not found")
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"ok": False, "error": message}, status)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def make_handler(state: AppState, frontend_root: Path) -> type[AnimeRequestHandler]:
    class ConfiguredHandler(AnimeRequestHandler):
        pass

    ConfiguredHandler.state = state
    ConfiguredHandler.frontend_root = frontend_root
    return ConfiguredHandler


def run(host: str = "127.0.0.1", port: int = 8000, project_root: Path = PROJECT_ROOT) -> None:
    state = AppState(project_root)
    handler = make_handler(state, project_root / "frontend")
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Anime Compass is running at http://{host}:{port}")
    print(f"Loaded {state.recommender.meta()['count']} anime records")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Anime Compass web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
