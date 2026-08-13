from __future__ import annotations

import math
import re
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

try:
    import numpy as np
except ImportError:  # The recommender still works without the dense hybrid channels.
    np = None


TOKEN_RE = re.compile(r"[a-z0-9]+")
EMBEDDING_DIMENSIONS = 192
SVD_FEATURES = 384
SVD_DIMENSIONS = 48
MODEL_VERSION = "hybrid-collaborative-v4"
MIN_ENTITY_FREQUENCY_LARGE_CATALOG = 2
DEFAULT_CHANNEL_WEIGHTS = {
    "metadata": 0.16,
    "synopsis": 0.10,
    "lsa": 0.04,
    "semantic_embedding": 0.14,
    "dense": 0.08,
    "creator": 0.05,
    "collaborative": 0.22,
    "quality": 0.13,
    "session": 0.05,
    "novelty": 0.03,
}
SELECTED_CREATOR_ROLES = {
    "director",
    "original creator",
    "creator",
    "series composition",
    "script",
    "music",
    "character design",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "its",
    "anime",
    "like",
    "liked",
    "next",
    "of",
    "on",
    "or",
    "recommend",
    "recommendation",
    "recommendations",
    "series",
    "show",
    "shows",
    "that",
    "the",
    "their",
    "this",
    "to",
    "watch",
    "with",
}

SERIES_STOP_TOKENS = {
    "movie",
    "season",
    "special",
    "specials",
    "ova",
    "ona",
    "tv",
    "recap",
    "remix",
    "part",
    "chapter",
    "episode",
    "episodes",
}

STORY_THEMES = {
    "romantic tension": {
        "affection",
        "attraction",
        "couple",
        "crush",
        "dating",
        "feelings",
        "love",
        "relationship",
        "romance",
        "romantic",
    },
    "close friendship": {
        "bond",
        "friend",
        "friends",
        "friendship",
        "together",
        "trust",
    },
    "creative passion": {
        "art",
        "artist",
        "cosplay",
        "craft",
        "create",
        "design",
        "doll",
        "hobby",
        "outfit",
        "outfits",
        "passion",
        "sewing",
    },
    "self-expression": {
        "accept",
        "confidence",
        "dream",
        "identity",
        "passions",
        "ridicule",
        "ridiculed",
        "secret",
        "unique",
    },
    "school life": {
        "class",
        "classmate",
        "club",
        "high",
        "school",
        "student",
        "students",
    },
    "opposites attract": {
        "contrast",
        "different",
        "friendless",
        "meek",
        "popular",
        "shy",
        "worlds",
    },
    "supernatural connection": {
        "destiny",
        "dream",
        "fate",
        "miracle",
        "mysterious",
        "phenomenon",
        "spirit",
        "supernatural",
    },
    "coming of age": {
        "adolescence",
        "childhood",
        "grow",
        "student",
        "teen",
        "teenage",
        "teenager",
        "youth",
    },
    "separation and reunion": {
        "apart",
        "distance",
        "missing",
        "reunite",
        "reunion",
        "separate",
        "separated",
    },
    "weather and the sky": {
        "cloud",
        "rain",
        "sky",
        "storm",
        "sunlight",
        "weather",
    },
    "city and countryside": {
        "city",
        "countryside",
        "rural",
        "tokyo",
        "village",
    },
}


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def normalize_label(text: str | None) -> str:
    return " ".join(tokenize(text))


def entity_name_variants(text: str | None) -> set[str]:
    value = str(text or "").strip()
    if not value:
        return set()
    key = " ".join("".join(character if character.isalnum() else " " for character in value.casefold()).split())
    variants = {key}
    if "," in value:
        last, first = (part.strip() for part in value.split(",", 1))
        reversed_key = " ".join(
            "".join(character if character.isalnum() else " " for character in f"{first} {last}".casefold()).split()
        )
        variants.add(reversed_key)
    return {variant for variant in variants if variant}


def creator_roles(value: str | None) -> list[str]:
    roles = set()
    for part in re.split(r"[,/;|]+", str(value or "")):
        role = normalize_label(part)
        if role in SELECTED_CREATOR_ROLES:
            roles.add(role)
    return sorted(roles)


def natural_join(values: list[str]) -> str:
    cleaned = [value for value in values if value]
    if len(cleaned) <= 1:
        return "".join(cleaned)
    if len(cleaned) == 2:
        return " and ".join(cleaned)
    return ", ".join(cleaned[:-1]) + ", and " + cleaned[-1]


def story_text(item: dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("synopsis") or ""),
            " ".join(str(genre) for genre in item.get("genres", [])),
        ]
    )


def story_themes(text: str | None) -> list[str]:
    tokens = set(tokenize(text))
    return [theme for theme, keywords in STORY_THEMES.items() if tokens.intersection(keywords)]


def cosine(a: Counter[str], b: Counter[str], norm_a: float | None = None, norm_b: float | None = None) -> float:
    if not a or not b:
        return 0.0

    if len(a) > len(b):
        a, b = b, a
        norm_a, norm_b = norm_b, norm_a

    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = norm_a if norm_a is not None else math.sqrt(sum(value * value for value in a.values()))
    norm_b = norm_b if norm_b is not None else math.sqrt(sum(value * value for value in b.values()))

    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def dense_embedding(text: str | None, dimensions: int):
    vector = np.zeros(dimensions, dtype=np.float32)
    tokens = tokenize(text)

    for token in tokens:
        add_hashed_feature(vector, token, 1.0)
        if len(token) >= 5:
            for index in range(len(token) - 2):
                add_hashed_feature(vector, f"char:{token[index : index + 3]}", 0.18)

    for left, right in zip(tokens, tokens[1:], strict=False):
        add_hashed_feature(vector, f"{left}_{right}", 0.65)

    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


def add_hashed_feature(vector, feature: str, weight: float) -> None:
    bucket_hash = zlib.crc32(feature.encode("utf-8"))
    bucket = bucket_hash % vector.shape[0]
    sign = 1.0 if zlib.crc32(("sign:" + feature).encode("utf-8")) % 2 == 0 else -1.0
    vector[bucket] += weight * sign


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def dense_cosine(a, b) -> float:
    if np is None or a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def normalize_vector(vector):
    if np is None or vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if not norm:
        return None
    return (vector / norm).astype(np.float32)


def positive_score(value: float | None) -> float:
    if value is None:
        return 0.0
    if math.isnan(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {channel: max(0.0, float(weights.get(channel, 0.0))) for channel in DEFAULT_CHANNEL_WEIGHTS}
    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_CHANNEL_WEIGHTS.copy()
    return {channel: value / total for channel, value in cleaned.items()}


class AnimeRecommender:
    def __init__(
        self,
        catalog: list[dict[str, Any]],
        weights: dict[str, float] | None = None,
        semantic_index: Any | None = None,
        collaborative_index: Any | None = None,
    ):
        self.catalog = catalog
        self.semantic_index = semantic_index
        self.collaborative_index = collaborative_index
        self.weights = normalize_weights({**DEFAULT_CHANNEL_WEIGHTS, **(weights or {})})
        self.item_ids = [int(item["id"]) for item in catalog]
        self.index_by_id = {anime_id: index for index, anime_id in enumerate(self.item_ids)}
        self.by_id = {int(item["id"]): item for item in catalog}
        self.voice_actor_roles_by_anime = {int(item["id"]): self._voice_actor_roles(item) for item in catalog}
        self.voice_actor_by_id, self.voice_actor_by_key = self._build_voice_actor_index()
        self.max_members = max((int(item.get("members") or 0) for item in catalog), default=1)
        self.max_popularity = max((int(item.get("popularity") or 0) for item in catalog), default=1)
        scores = [float(item["score"]) / 10.0 for item in catalog if item.get("score") is not None]
        self.global_score_mean = sum(scores) / len(scores) if scores else 0.7
        self.min_entity_frequency = MIN_ENTITY_FREQUENCY_LARGE_CATALOG if len(catalog) >= 100 else 1
        self.studio_counts = Counter(studio for item in catalog for studio in item.get("studios", []))
        self.creator_counts = Counter(
            person.get("name", "") for item in catalog for person in self._selected_creators(item) if person.get("name")
        )
        raw_vectors = {int(item["id"]): self._build_metadata_vector(item) for item in catalog}
        self.idf = self._build_idf(raw_vectors.values())
        self.vectors = {anime_id: self._apply_idf(vector) for anime_id, vector in raw_vectors.items()}
        self.norms = {
            anime_id: math.sqrt(sum(value * value for value in vector.values()))
            for anime_id, vector in self.vectors.items()
        }
        raw_story_vectors = {int(item["id"]): self._build_story_vector(item) for item in catalog}
        self.story_idf = self._build_idf(raw_story_vectors.values())
        self.story_vectors = {
            anime_id: self._apply_idf(vector, self.story_idf) for anime_id, vector in raw_story_vectors.items()
        }
        self.story_norms = {
            anime_id: math.sqrt(sum(value * value for value in vector.values()))
            for anime_id, vector in self.story_vectors.items()
        }
        raw_creator_vectors = {int(item["id"]): self._build_creator_vector(item) for item in catalog}
        self.creator_idf = self._build_idf(raw_creator_vectors.values())
        self.creator_vectors = {
            anime_id: self._apply_idf(vector, self.creator_idf) for anime_id, vector in raw_creator_vectors.items()
        }
        self.creator_norms = {
            anime_id: math.sqrt(sum(value * value for value in vector.values()))
            for anime_id, vector in self.creator_vectors.items()
        }
        self.embedding_vectors = self._build_embedding_index()
        self.svd_features: list[str] = []
        self.svd_feature_index: dict[str, int] = {}
        self.svd_components = None
        self.svd_vectors = self._build_svd_index()

    def model_info(self) -> dict[str, Any]:
        return {
            "name": MODEL_VERSION,
            "channels": list(DEFAULT_CHANNEL_WEIGHTS),
            "weights": self.weights,
            "uses_collaborative_filtering": self.collaborative_index is not None,
            "architecture": "multi-channel hybrid collaborative/content recommender with session personalization",
            "notes": (
                "Uses manual TF-IDF, NumPy LSA, an optional pretrained semantic embedding index, "
                "deterministic hashed dense text vectors, metadata, "
                "creator/entity features, quality, novelty, session signals, and optional item embeddings "
                "learned from anonymous ratings. It does not require scikit-learn."
            ),
            "semantic_embedding": self.semantic_index.model_info()
            if self.semantic_index is not None
            else {"available": False},
            "collaborative": self.collaborative_index.model_info()
            if self.collaborative_index is not None
            else {"available": False},
        }

    def meta(self) -> dict[str, Any]:
        genres = sorted({genre for item in self.catalog for genre in item.get("genres", [])})
        media_types = sorted({item.get("type") for item in self.catalog if item.get("type")})
        years = [item["start_year"] for item in self.catalog if item.get("start_year")]
        genre_covers: dict[str, str] = {}

        for item in self.catalog:
            image_url = item.get("image_url")
            if not image_url:
                continue
            for genre in item.get("genres", []):
                genre_covers.setdefault(genre, image_url)

        return {
            "count": len(self.catalog),
            "genres": genres,
            "genre_covers": genre_covers,
            "types": media_types,
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
        }

    def search(
        self,
        query: str,
        limit: int = 10,
        genres: list[str] | None = None,
        media_type: str | None = None,
        min_score: float | None = None,
        max_episodes: int | None = None,
    ) -> list[dict[str, Any]]:
        results, _ = self.search_page(
            query,
            limit=limit,
            genres=genres,
            media_type=media_type,
            min_score=min_score,
            max_episodes=max_episodes,
        )
        return results

    def search_page(
        self,
        query: str,
        *,
        limit: int = 24,
        offset: int = 0,
        genres: list[str] | None = None,
        include_genres: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        formats: list[str] | None = None,
        required_studios: list[str] | None = None,
        media_type: str | None = None,
        min_score: float | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        max_episodes: int | None = None,
        sort_by: str = "relevance",
        sort_order: str | None = None,
        semantic: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        query = (query or "").strip()
        selected_genres = {genre.casefold() for genre in include_genres or genres or [] if genre}
        blocked_genres = {genre.casefold() for genre in exclude_genres or [] if genre}
        format_keys = {value.casefold() for value in formats or [] if value}
        if media_type:
            format_keys.add(media_type.casefold())
        studio_keys = {normalize_label(value) for value in required_studios or [] if normalize_label(value)}
        allowed_sort_fields = {"relevance", "score", "popularity", "rank", "members", "start_year", "title"}
        if sort_by not in allowed_sort_fields:
            raise ValueError(f"Unsupported search sort field: {sort_by}")
        natural_order = "asc" if sort_by in {"popularity", "rank", "title"} else "desc"
        order = sort_order or natural_order
        if order not in {"asc", "desc"}:
            raise ValueError(f"Unsupported search sort order: {order}")
        query_tokens = tokenize(query)
        query_text = normalize_label(query)
        semantic_scores: dict[int, float] = {}
        if semantic and query and np is not None and self.semantic_index is not None:
            try:
                query_vector = self.semantic_index.encode_query(query)
                similarities = self.semantic_index.matrix @ query_vector
                candidate_count = min(200, len(similarities))
                if candidate_count:
                    indexes = np.argpartition(similarities, -candidate_count)[-candidate_count:]
                    semantic_scores = {
                        int(self.semantic_index.anime_ids[index]): max(0.0, float(similarities[index]))
                        for index in indexes
                        if float(similarities[index]) > 0.0
                    }
            except (OSError, RuntimeError, ValueError):
                semantic_scores = {}

        scored: list[tuple[float, dict[str, Any], set[str]]] = []

        for item in self.catalog:
            item_studio_keys = {normalize_label(value) for value in item.get("studios", []) if value}
            if studio_keys and not studio_keys.issubset(item_studio_keys):
                continue
            if not self._passes_filters(
                item,
                selected_genres=selected_genres,
                blocked_genres=blocked_genres,
                format_keys=format_keys,
                min_score=min_score,
                min_year=min_year,
                max_year=max_year,
                max_episodes=max_episodes,
            ):
                continue

            if not query:
                scored.append((self._quality_bonus(item), item, {"quality"}))
                continue

            titles = [str(value) for value in [item.get("title"), *(item.get("aliases") or [])] if value]
            normalized_titles = [normalize_label(value) for value in titles]
            title_tokens = {token for value in titles for token in tokenize(value)}
            item_genres = {genre.casefold() for genre in item.get("genres", [])}
            people = [
                normalize_label(person.get("name", ""))
                for person in item.get("characters", [])[:8]
                + item.get("staff", [])[:8]
                + item.get("voice_actors", [])[:8]
            ]
            studios = [normalize_label(value) for value in item.get("studios", [])]
            story_vector = self.story_vectors.get(int(item["id"]), {})

            score = 0.0
            matched_fields: set[str] = set()
            if query_text in normalized_titles:
                score += 100.0
                matched_fields.add("title_exact")
            elif any(query_text in value for value in normalized_titles):
                score += 35.0
                matched_fields.add("title")
            elif any(query_text and query_text in name for name in people):
                score += 20.0
                matched_fields.add("person")
            elif any(query_text and query_text in studio for studio in studios):
                score += 18.0
                matched_fields.add("studio")

            for token in query_tokens:
                if token in title_tokens:
                    score += 12.0
                    matched_fields.add("title")
                if any(token in genre for genre in item_genres):
                    score += 5.0
                    matched_fields.add("genre")
                if any(token in name for name in people):
                    score += 4.0
                    matched_fields.add("person")
                if any(token in studio for studio in studios):
                    score += 4.0
                    matched_fields.add("studio")
                if token in story_vector:
                    score += min(3.0, float(story_vector[token]) * 0.35)
                    matched_fields.add("synopsis")

            semantic_score = semantic_scores.get(int(item["id"]), 0.0)
            if semantic_score > 0:
                score += semantic_score * 18.0
                matched_fields.add("semantic")

            if score > 0:
                score += self._quality_bonus(item) * 3
                scored.append((score, item, matched_fields))

        if query and not scored:
            for item in self.catalog:
                item_studio_keys = {normalize_label(value) for value in item.get("studios", []) if value}
                if studio_keys and not studio_keys.issubset(item_studio_keys):
                    continue
                if not self._passes_filters(
                    item,
                    selected_genres=selected_genres,
                    blocked_genres=blocked_genres,
                    format_keys=format_keys,
                    min_score=min_score,
                    min_year=min_year,
                    max_year=max_year,
                    max_episodes=max_episodes,
                ):
                    continue
                confidence = max(
                    (
                        SequenceMatcher(None, query_text, normalize_label(value)).ratio()
                        for value in [item.get("title"), *(item.get("aliases") or [])]
                        if value
                    ),
                    default=0.0,
                )
                if confidence >= 0.55:
                    scored.append((confidence * 30.0 + self._quality_bonus(item) * 3, item, {"fuzzy_title"}))

        if sort_by == "relevance":
            scored.sort(key=lambda row: (row[0], self._quality_bonus(row[1])), reverse=order == "desc")
        else:
            present = [row for row in scored if row[1].get(sort_by) not in (None, "")]
            missing = [row for row in scored if row[1].get(sort_by) in (None, "")]
            present.sort(key=lambda row: (str(row[1].get("title") or "").casefold(), int(row[1]["id"])))
            present.sort(key=lambda row: row[1][sort_by], reverse=order == "desc")
            missing.sort(key=lambda row: (str(row[1].get("title") or "").casefold(), int(row[1]["id"])))
            scored = present + missing

        total = len(scored)
        results = []
        for score, item, matched_fields in scored[max(0, offset) : max(0, offset) + max(1, limit)]:
            public = self.public_item(item)
            public["search_score"] = round(score, 6)
            public["matched_fields"] = sorted(matched_fields)
            results.append(public)
        return results, total

    def rank_catalog(
        self,
        *,
        query: str = "",
        include_genres: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        formats: list[str] | None = None,
        required_studios: list[str] | None = None,
        min_score: float | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        max_episodes: int | None = None,
        excluded_titles: list[str] | None = None,
        sort_by: str = "score",
        sort_order: str | None = None,
        limit: int = 10,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Filter the catalog and sort by one explicit field without preference signals."""
        allowed_fields = {"score", "popularity", "rank", "members", "start_year", "title"}
        if sort_by not in allowed_fields:
            raise ValueError(f"Unsupported catalog sort field: {sort_by}")
        natural_order = "asc" if sort_by in {"popularity", "rank", "title"} else "desc"
        order = sort_order or natural_order
        if order not in {"asc", "desc"}:
            raise ValueError(f"Unsupported catalog sort order: {order}")

        query_key = normalize_label(query)
        query_tokens = set(tokenize(query_key))
        selected_genres = {genre.casefold() for genre in include_genres or [] if genre}
        blocked_genres = {genre.casefold() for genre in exclude_genres or [] if genre}
        format_keys = {value.casefold() for value in formats or [] if value}
        studio_keys = {normalize_label(value) for value in required_studios or [] if value}
        excluded_keys = {normalize_label(value) for value in excluded_titles or [] if value}
        candidates = []
        for item in self.catalog:
            title_key = normalize_label(item.get("title"))
            title_tokens = {
                token
                for value in [item.get("title"), *(item.get("aliases") or [])]
                if value
                for token in tokenize(str(value))
            }
            if query_tokens and not query_tokens.issubset(title_tokens):
                continue
            if title_key in excluded_keys:
                continue
            item_studio_keys = {normalize_label(value) for value in item.get("studios", []) if value}
            if studio_keys and not studio_keys.issubset(item_studio_keys):
                continue
            if not self._passes_filters(
                item,
                selected_genres=selected_genres,
                blocked_genres=blocked_genres,
                format_keys=format_keys,
                min_score=min_score,
                min_year=min_year,
                max_year=max_year,
                max_episodes=max_episodes,
            ):
                continue
            candidates.append(item)

        present = [item for item in candidates if item.get(sort_by) not in (None, "")]
        missing = [item for item in candidates if item.get(sort_by) in (None, "")]
        present.sort(key=lambda item: (str(item.get("title") or "").casefold(), int(item["id"])))
        present.sort(key=lambda item: item[sort_by], reverse=order == "desc")
        missing.sort(key=lambda item: (str(item.get("title") or "").casefold(), int(item["id"])))

        label = {
            "score": "catalog score",
            "popularity": "catalog popularity rank",
            "rank": "catalog rank",
            "members": "member count",
            "start_year": "start year",
            "title": "title",
        }[sort_by]
        direction = "highest first" if order == "desc" else "lowest first"
        results = []
        for position, item in enumerate((present + missing)[: max(1, limit)], start=1):
            public = self.public_item(item)
            public["ranking"] = {
                "position": position,
                "sort_by": sort_by,
                "sort_order": order,
                "value": item.get(sort_by),
            }
            value = item.get(sort_by)
            public["reasons"] = [f"Ranked #{position} by {label} ({value}); {direction}."]
            results.append(public)
        return results, {
            "operation": "rank_catalog",
            "query": query,
            "required_studios": list(required_studios or []),
            "sort_by": sort_by,
            "sort_order": order,
            "candidate_count": len(candidates),
            "missing_sort_values": len(missing),
            "session_preferences_used": False,
        }

    def resolve_titles(self, titles: list[str]) -> list[int]:
        ids: list[int] = []
        seen: set[int] = set()

        for title in titles:
            matches = self.search(title, limit=1)
            if not matches:
                continue
            anime_id = int(matches[0]["id"])
            if anime_id not in seen:
                ids.append(anime_id)
                seen.add(anime_id)

        return ids

    def resolve_title_details(self, titles: list[str]) -> list[dict[str, Any]]:
        resolved = []
        for title in titles:
            matches = self.search(title, limit=1)
            if not matches:
                resolved.append({"input": title, "anime_id": None, "matched_title": None})
                continue
            match = matches[0]
            resolved.append(
                {
                    "input": title,
                    "anime_id": match["id"],
                    "matched_title": match["title"],
                }
            )
        return resolved

    def recommend(
        self,
        reference_titles: list[str] | None = None,
        liked_ids: list[int] | None = None,
        liked_titles: list[str] | None = None,
        excluded_ids: list[int] | None = None,
        excluded_titles: list[str] | None = None,
        seen_titles: list[str] | None = None,
        genres: list[str] | None = None,
        include_genres: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        media_type: str | None = None,
        formats: list[str] | None = None,
        min_score: float | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        max_episodes: int | None = None,
        query: str | None = None,
        free_text_preferences: str | None = None,
        required_studios: list[str] | None = None,
        preferred_studios: list[str] | None = None,
        required_staff: list[str] | None = None,
        preferred_staff: list[str] | None = None,
        required_characters: list[str] | None = None,
        preferred_characters: list[str] | None = None,
        required_voice_actors: list[str] | None = None,
        required_voice_actor_ids: list[int] | None = None,
        preferred_voice_actors: list[str] | None = None,
        required_entity_constraints: list[dict[str, Any]] | None = None,
        novelty_preference: str = "neutral",
        exclude_related_series: bool = True,
        one_per_series: bool = False,
        session_profile: dict[str, Any] | None = None,
        diversity_strength: float = 0.12,
        weights: dict[str, float] | None = None,
        limit: int = 12,
        top_k: int | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        recommend_started = time.perf_counter()
        session_profile = session_profile or {}
        if top_k is not None:
            limit = top_k
        required_actor_input = [str(value) for value in required_voice_actors or [] if str(value).strip()]
        required_actor_ids = [int(value) for value in required_voice_actor_ids or []]
        entity_resolution_started = time.perf_counter()
        resolved_required_actors, actor_resolution_errors = self.resolve_voice_actor_constraints(
            required_actor_input,
            required_actor_ids,
        )
        entity_resolution_ms = (time.perf_counter() - entity_resolution_started) * 1000
        required_studio_keys = {normalize_label(value) for value in required_studios or [] if normalize_label(value)}
        required_staff_keys = {normalize_label(value) for value in required_staff or [] if normalize_label(value)}
        required_character_keys = {
            normalize_label(value) for value in required_characters or [] if normalize_label(value)
        }
        normalized_entity_constraints = self._normalize_required_entity_constraints(required_entity_constraints)
        has_required_voice_actor = bool(required_actor_input or required_actor_ids)
        has_required_entity = bool(
            has_required_voice_actor
            or required_studio_keys
            or required_staff_keys
            or required_character_keys
            or normalized_entity_constraints
        )
        if has_required_entity:
            session_profile = {}
        diagnostic_target = diagnostics if diagnostics is not None else {}
        diagnostic_target.update(
            {
                "required_voice_actors": required_actor_input,
                "resolved_voice_actors": [
                    {
                        "entity_id": record.get("entity_id"),
                        "entity_type": record.get("entity_type"),
                        "matched_name": record.get("matched_name"),
                        "related_anime_count": len(record.get("related_anime_ids", [])),
                    }
                    for record in resolved_required_actors
                ],
                "voice_actor_resolution_errors": actor_resolution_errors,
                "required_entity_constraints": [
                    {
                        "entity_type": constraint["entity_type"],
                        "entity_id": constraint.get("entity_id"),
                        "matched_name": constraint["matched_name"],
                        "related_anime_count": len(constraint["related_anime_ids"]),
                    }
                    for constraint in normalized_entity_constraints
                ],
                "candidate_count_before_filter": 0,
                "candidate_count_after_entity_filter": 0,
                "candidate_count_after_voice_actor_filter": 0,
                "verified_voice_actor_matches": 0,
                "verified_entity_matches": 0,
                "timing_ms": {"entity_resolution": round(entity_resolution_ms, 3)},
            }
        )
        if has_required_voice_actor and actor_resolution_errors:
            return []
        configured_weights = normalize_weights({**self.weights, **(weights or {})})
        all_reference_titles = list(reference_titles or []) + list(liked_titles or [])
        all_reference_titles.extend(str(value) for value in session_profile.get("liked_titles", []) or [])
        all_excluded_titles = list(excluded_titles or []) + list(seen_titles or [])
        all_excluded_titles.extend(str(value) for value in session_profile.get("excluded_titles", []) or [])
        all_excluded_titles.extend(str(value) for value in session_profile.get("seen_titles", []) or [])
        session_disliked_titles = [str(value) for value in session_profile.get("disliked_titles", []) or []]
        all_excluded_titles.extend(session_disliked_titles)
        all_include_genres = list(include_genres or genres or [])
        session_preferred_genres = [str(value) for value in session_profile.get("preferred_genres", []) or []]
        all_exclude_genres = list(exclude_genres or [])
        all_exclude_genres.extend(str(value) for value in session_profile.get("excluded_genres", []) or [])
        all_preferred_studios = list(preferred_studios or [])
        all_preferred_studios.extend(str(value) for value in session_profile.get("preferred_studios", []) or [])
        all_preferred_staff = list(preferred_staff or [])
        all_preferred_staff.extend(str(value) for value in session_profile.get("preferred_staff", []) or [])
        all_preferred_characters = list(preferred_characters or [])
        all_preferred_characters.extend(str(value) for value in session_profile.get("preferred_characters", []) or [])
        all_preferred_voice_actors = list(preferred_voice_actors or [])
        all_preferred_voice_actors.extend(
            str(value) for value in session_profile.get("preferred_voice_actors", []) or []
        )
        preference_text = " ".join(
            dict.fromkeys(value.strip() for value in (query or "", free_text_preferences or "") if value.strip())
        )
        novelty_preference = (
            novelty_preference if novelty_preference in {"neutral", "less_famous", "mainstream"} else "neutral"
        )

        liked_ids = [int(value) for value in liked_ids or [] if int(value) in self.by_id]
        liked_ids.extend(self.resolve_titles(all_reference_titles))
        liked_ids = list(dict.fromkeys(liked_ids))
        excluded_ids = [int(value) for value in excluded_ids or [] if int(value) in self.by_id]
        excluded_ids.extend(self.resolve_titles(all_excluded_titles))
        excluded_ids = list(dict.fromkeys(excluded_ids))
        disliked_ids = self.resolve_titles(session_disliked_titles)
        explicit_rating_ids: dict[int, float] = {}
        for title, rating in (session_profile.get("temporary_ratings", {}) or {}).items():
            resolved = self.resolve_titles([str(title)])
            if resolved:
                explicit_rating_ids[resolved[0]] = float(rating)
        excluded_series = {
            series_key(self.by_id[anime_id]["title"]) for anime_id in excluded_ids if anime_id in self.by_id
        }
        if excluded_series and exclude_related_series:
            excluded_ids.extend(
                int(item["id"]) for item in self.catalog if series_key(item["title"]) in excluded_series
            )
            excluded_ids = list(dict.fromkeys(excluded_ids))

        selected_genres = {genre.casefold() for genre in all_include_genres if genre}
        blocked_genres = {genre.casefold() for genre in all_exclude_genres if genre}
        format_keys = {value.casefold() for value in formats or [] if value}
        if media_type:
            format_keys.add(media_type.casefold())
        profile = Counter()
        story_profile = Counter()
        creator_profile = Counter()
        embedding_profile = None
        svd_profile = None
        semantic_profile = None

        for anime_id in liked_ids:
            profile.update(self.vectors[anime_id])
            story_profile.update(self.story_vectors[anime_id])
            creator_profile.update(self.creator_vectors[anime_id])

        for genre in selected_genres:
            genre_key = normalize_label(genre)
            self._add_profile_feature(profile, f"genre:{genre_key}", 12.0)
            for token in tokenize(genre):
                self._add_profile_feature(profile, token, 2.4)
        for genre in session_preferred_genres:
            genre_key = normalize_label(genre)
            self._add_profile_feature(profile, f"genre:{genre_key}", 4.0)
            for token in tokenize(genre):
                self._add_profile_feature(profile, token, 0.8)

        for studio in all_preferred_studios:
            self._add_profile_feature(profile, f"studio:{normalize_label(studio)}", 8.0)
            self._add_profile_feature(creator_profile, f"studio:{normalize_label(studio)}", 8.0, self.creator_idf)
        for person in all_preferred_staff:
            key = normalize_label(person)
            self._add_profile_feature(profile, f"staff:{key}", 8.0)
            for role in SELECTED_CREATOR_ROLES:
                self._add_profile_feature(creator_profile, f"{role}:{key}", 4.0, self.creator_idf)
            self._add_profile_feature(creator_profile, f"creator:{key}", 4.0, self.creator_idf)
            self._add_profile_feature(creator_profile, f"producer:{key}", 4.0, self.creator_idf)
        for character in all_preferred_characters:
            self._add_profile_feature(profile, f"character:{normalize_label(character)}", 8.0)
        for actor in all_preferred_voice_actors:
            self._add_profile_feature(profile, f"voice_actor:{normalize_label(actor)}", 8.0)

        if preference_text:
            for token in tokenize(preference_text):
                self._add_profile_feature(profile, token, 1.2)
                self._add_profile_feature(story_profile, token, 4.0, self.story_idf)
            for theme in story_themes(preference_text):
                self._add_profile_feature(story_profile, f"theme:{theme}", 8.0, self.story_idf)

        profile_norm = math.sqrt(sum(value * value for value in profile.values()))
        story_profile_norm = math.sqrt(sum(value * value for value in story_profile.values()))
        creator_profile_norm = math.sqrt(sum(value * value for value in creator_profile.values()))
        embedding_profile = self._embedding_profile(liked_ids, preference_text, all_include_genres)
        svd_profile = self._svd_profile(liked_ids, story_profile)
        semantic_profile = self._semantic_profile(liked_ids, preference_text, all_include_genres)
        collaborative_scores = (
            self.collaborative_index.profile_scores(
                positive_ids=[anime_id for anime_id in liked_ids if anime_id not in explicit_rating_ids],
                negative_ids=[anime_id for anime_id in disliked_ids if anime_id not in explicit_rating_ids],
                explicit_ratings=explicit_rating_ids,
            )
            if self.collaborative_index is not None
            else {}
        )
        liked_set = set(liked_ids)
        blocked_set = liked_set.union(excluded_ids)
        session_history = self._session_history(liked_ids)
        active_channels = {
            "metadata": bool(profile),
            "synopsis": bool(story_profile),
            "lsa": svd_profile is not None,
            "semantic_embedding": semantic_profile is not None,
            "dense": embedding_profile is not None,
            "creator": bool(creator_profile),
            "collaborative": bool(collaborative_scores),
            "quality": True,
            "session": self._has_session_signal(session_profile),
            "novelty": novelty_preference != "neutral",
        }
        inactive_reasons = {
            "metadata": "No reference title, catalog label, or named entity preference was supplied.",
            "synopsis": "No reference synopsis or free-text story preference was supplied.",
            "lsa": "No usable text profile was available, or NumPy LSA is unavailable.",
            "semantic_embedding": "No valid pretrained embedding artifact or semantic query profile is available.",
            "dense": "No usable text profile was available, or NumPy hashed vectors are unavailable.",
            "creator": "No reference creator, studio, staff, or producer preference was supplied.",
            "collaborative": (
                "No rated or liked title has a usable collaborative embedding, "
                "or the collaborative artifact is unavailable."
            ),
            "session": "The session has no saved preference signals.",
            "novelty": "No less-famous or mainstream preference was requested.",
        }
        has_personalization = any(active_channels[channel] for channel in active_channels if channel != "quality")
        recommendation_mode = "hybrid" if has_personalization else "quality_fallback"
        if recommendation_mode == "quality_fallback":
            effective_weights = {channel: (1.0 if channel == "quality" else 0.0) for channel in configured_weights}
        else:
            active_weight_total = sum(
                weight for channel, weight in configured_weights.items() if active_channels[channel]
            )
            effective_weights = {
                channel: (weight / active_weight_total if active_channels[channel] and active_weight_total else 0.0)
                for channel, weight in configured_weights.items()
            }
        scored: list[tuple[float, dict[str, Any], list[str], dict[str, float], dict[str, Any]]] = []
        verified_roles_by_anime: dict[int, list[dict[str, Any]]] = {}
        verified_entities_by_anime: dict[int, list[dict[str, Any]]] = {}
        candidate_count_before_actor_filter = 0
        candidate_count_after_actor_filter = 0
        scoring_started = time.perf_counter()

        for item in self.catalog:
            anime_id = int(item["id"])
            if anime_id in blocked_set:
                continue
            if not self._passes_filters(
                item,
                selected_genres=selected_genres,
                blocked_genres=blocked_genres,
                format_keys=format_keys,
                min_score=min_score,
                min_year=min_year,
                max_year=max_year,
                max_episodes=max_episodes,
            ):
                continue
            candidate_count_before_actor_filter += 1
            item_studio_keys = {normalize_label(value) for value in item.get("studios", [])}
            item_staff_keys = {
                normalize_label(person.get("name"))
                for person in item.get("staff_relationships", item.get("staff", []))
                if person.get("name")
            }
            item_character_keys = {normalize_label(value) for value in item.get("character_names", []) if value} or {
                normalize_label(person.get("name")) for person in item.get("characters", []) if person.get("name")
            }
            if not (
                required_studio_keys.issubset(item_studio_keys)
                and required_staff_keys.issubset(item_staff_keys)
                and required_character_keys.issubset(item_character_keys)
            ):
                continue
            if any(anime_id not in constraint["related_anime_ids"] for constraint in normalized_entity_constraints):
                continue
            matched_actor_roles = self._matched_required_voice_actor_roles(
                anime_id,
                resolved_required_actors,
            )
            if has_required_voice_actor and not matched_actor_roles:
                continue
            candidate_count_after_actor_filter += 1
            if matched_actor_roles:
                verified_roles_by_anime[anime_id] = matched_actor_roles
            matched_entity_relationships = self._required_entity_evidence(
                item,
                required_studio_keys=required_studio_keys,
                required_staff_keys=required_staff_keys,
                required_character_keys=required_character_keys,
                matched_voice_actor_roles=matched_actor_roles,
                generic_constraints=normalized_entity_constraints,
            )
            if has_required_entity:
                verified_entities_by_anime[anime_id] = matched_entity_relationships

            vector = self.vectors[anime_id]
            similarity = cosine(profile, vector, profile_norm, self.norms[anime_id]) if profile else 0.0
            story_vector = self.story_vectors[anime_id]
            story_similarity = (
                cosine(story_profile, story_vector, story_profile_norm, self.story_norms[anime_id])
                if story_profile
                else 0.0
            )
            row_index = self.index_by_id[anime_id]
            embedding_similarity = (
                dense_cosine(embedding_profile, self.embedding_vectors[row_index])
                if self.embedding_vectors is not None and embedding_profile is not None
                else 0.0
            )
            svd_similarity = (
                dense_cosine(svd_profile, self.svd_vectors[row_index])
                if self.svd_vectors is not None and svd_profile is not None
                else 0.0
            )
            semantic_vector = self.semantic_index.document_vector(anime_id) if self.semantic_index is not None else None
            semantic_similarity = (
                float(np.dot(semantic_profile, semantic_vector))
                if np is not None and semantic_profile is not None and semantic_vector is not None
                else 0.0
            )
            creator_similarity = (
                cosine(
                    creator_profile, self.creator_vectors[anime_id], creator_profile_norm, self.creator_norms[anime_id]
                )
                if creator_profile
                else 0.0
            )
            collaborative_similarity = collaborative_scores.get(anime_id, 0.0)
            quality = self._quality_bonus(item)
            session_score = self._session_score(
                item,
                liked_ids,
                session_profile,
                selected_genres,
                blocked_genres,
                history=session_history,
            )
            signals = {
                "metadata": positive_score(similarity),
                "synopsis": positive_score(story_similarity),
                "lsa": positive_score(svd_similarity),
                "semantic_embedding": positive_score(semantic_similarity),
                "dense": positive_score(embedding_similarity),
                "creator": positive_score(creator_similarity),
                "collaborative": positive_score(collaborative_similarity),
                "quality": positive_score(quality),
                "session": positive_score(session_score),
                "novelty": positive_score(self._novelty_score(item, novelty_preference)),
            }
            contributions = {channel: effective_weights[channel] * signals[channel] for channel in configured_weights}
            total = sum(contributions.values())
            breakdown = {
                channel: {
                    "raw_score": round(signals[channel], 6),
                    "normalized_score": round(signals[channel], 6),
                    "configured_weight": round(configured_weights[channel], 6),
                    "effective_weight": round(effective_weights[channel], 6),
                    "weighted_contribution": round(contributions[channel], 6),
                    "active": active_channels[channel],
                    "inactive_reason": None if active_channels[channel] else inactive_reasons.get(channel),
                }
                for channel in configured_weights
            }
            reasons = self.explain(
                item,
                liked_ids=liked_ids,
                selected_genres=all_include_genres,
                query=preference_text,
                signals=signals,
                matched_voice_actor_roles=matched_actor_roles,
                matched_required_studios=[
                    studio for studio in item.get("studios", []) if normalize_label(studio) in required_studio_keys
                ],
                matched_required_staff=[
                    person.get("name")
                    for person in item.get("staff_relationships", item.get("staff", []))
                    if normalize_label(person.get("name")) in required_staff_keys
                ],
                matched_required_characters=[
                    name for name in item.get("character_names", []) if normalize_label(name) in required_character_keys
                ],
                matched_required_entities=matched_entity_relationships,
            )
            scored.append((total, item, reasons, signals, breakdown))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        diagnostic_target["timing_ms"]["candidate_generation_and_channel_scoring"] = round(
            (time.perf_counter() - scoring_started) * 1000,
            3,
        )

        results = []
        selected_items: list[dict[str, Any]] = []
        seen_series: set[str] = (
            {series_key(self.by_id[anime_id]["title"]) for anime_id in blocked_set if anime_id in self.by_id}
            if one_per_series
            else set()
        )

        reranking_started = time.perf_counter()
        while scored and len(results) < limit:
            best_index = 0
            best_score = -1.0
            for index, (score, item, _reasons, _signals, _breakdown) in enumerate(scored[: max(limit * 8, 40)]):
                adjusted = score - diversity_strength * self._diversity_penalty(item, selected_items)
                if adjusted > best_score:
                    best_score = adjusted
                    best_index = index

            score, item, reasons, signals, breakdown = scored.pop(best_index)
            if one_per_series:
                series = series_key(item["title"])
                if series in seen_series:
                    continue
                seen_series.add(series)

            public = self.public_item(item)
            diversity_penalty = self._diversity_penalty(item, selected_items)
            diversity_adjustment = -min(score, diversity_strength * diversity_penalty)
            final_score = score + diversity_adjustment
            public["match_score"] = round(final_score, 4)
            public["hybrid_scores"] = {key: round(value, 4) for key, value in signals.items()}
            public["hybrid_scores"]["final"] = round(final_score, 4)
            public["recommendation_mode"] = recommendation_mode
            public["active_channels"] = [channel for channel, active in active_channels.items() if active]
            public["inactive_channel_reasons"] = {
                channel: inactive_reasons[channel]
                for channel, active in active_channels.items()
                if not active and channel in inactive_reasons
            }
            public["configured_weights"] = {channel: round(value, 6) for channel, value in configured_weights.items()}
            public["effective_weights"] = {channel: round(value, 6) for channel, value in effective_weights.items()}
            public["weighted_contributions"] = {
                channel: round(breakdown[channel]["weighted_contribution"], 6) for channel in configured_weights
            }
            public["pre_diversity_score"] = round(score, 6)
            public["diversity_adjustment"] = round(diversity_adjustment, 6)
            public["final_score"] = round(final_score, 6)
            public["score_breakdown"] = {
                "recommendation_mode": recommendation_mode,
                "channels": breakdown,
                "pre_diversity_score": public["pre_diversity_score"],
                "diversity_adjustment": public["diversity_adjustment"],
                "final_score": public["final_score"],
            }
            public["reasons"] = reasons
            matched_actor_roles = verified_roles_by_anime.get(int(item["id"]), [])
            public["matched_voice_actors"] = list(dict.fromkeys(role["voice_actor"] for role in matched_actor_roles))
            public["voice_actor_roles"] = [
                {
                    "voice_actor": role["voice_actor"],
                    "character": role.get("character") or "Unknown character",
                    "language": role.get("language") or "Unknown",
                }
                for role in matched_actor_roles
            ]
            public["matched_required_studios"] = [
                studio for studio in item.get("studios", []) if normalize_label(studio) in required_studio_keys
            ]
            public["matched_required_staff"] = [
                person.get("name")
                for person in item.get("staff_relationships", item.get("staff", []))
                if normalize_label(person.get("name")) in required_staff_keys
            ]
            public["matched_required_characters"] = [
                name for name in item.get("character_names", []) if normalize_label(name) in required_character_keys
            ]
            public["entity_relationships"] = verified_entities_by_anime.get(int(item["id"]), [])
            public["matched_entities"] = public["entity_relationships"]
            public["matched_producers"] = self._matched_entity_names(public["entity_relationships"], "producer")
            public["matched_directors"] = self._matched_entity_names(public["entity_relationships"], "director")
            public["matched_original_creators"] = self._matched_entity_names(
                public["entity_relationships"], "original_creator"
            )
            public["matched_themes"] = self._matched_entity_names(public["entity_relationships"], "theme")
            public["matched_demographics"] = self._matched_entity_names(public["entity_relationships"], "demographic")
            public["explanation_data"] = self.explanation_data(
                item,
                liked_ids=liked_ids,
                selected_genres=all_include_genres,
                min_score=min_score,
                min_year=min_year,
                max_year=max_year,
                max_episodes=max_episodes,
                formats=list(format_keys),
                signals=signals,
                contributions=public["weighted_contributions"],
            )
            public["explanation_data"]["matched_voice_actors"] = public["matched_voice_actors"]
            public["explanation_data"]["voice_actor_roles"] = public["voice_actor_roles"]
            public["explanation_data"]["matched_required_studios"] = public["matched_required_studios"]
            public["explanation_data"]["matched_required_staff"] = public["matched_required_staff"]
            public["explanation_data"]["matched_required_characters"] = public["matched_required_characters"]
            public["explanation_data"]["entity_relationships"] = public["entity_relationships"]
            public["explanation_data"]["matched_producers"] = public["matched_producers"]
            public["explanation_data"]["matched_directors"] = public["matched_directors"]
            public["explanation_data"]["matched_original_creators"] = public["matched_original_creators"]
            public["explanation_data"]["matched_themes"] = public["matched_themes"]
            public["explanation_data"]["matched_demographics"] = public["matched_demographics"]
            public["explanation_data"]["session_feedback_influence"] = round(signals["session"], 6)
            public["explanation_data"]["quality_prior"] = round(signals["quality"], 6)
            public["explanation_data"]["diversity_reason"] = (
                "Reduced slightly to avoid a repetitive result list."
                if diversity_adjustment < 0
                else "No diversity penalty was needed."
            )
            relationship_constraints = [
                *(f"studio: {value}" for value in public["matched_required_studios"]),
                *(f"staff: {value}" for value in public["matched_required_staff"]),
                *(f"character: {value}" for value in public["matched_required_characters"]),
                *(f"voice actor: {value}" for value in public["matched_voice_actors"]),
                *(
                    f"{relationship['entity_type']}: {relationship['name']}"
                    for relationship in public["entity_relationships"]
                ),
            ]
            public["explanation_data"]["matched_constraints"] = list(
                dict.fromkeys(
                    [
                        *public["explanation_data"].get("matched_constraints", []),
                        *relationship_constraints,
                    ]
                )
            )
            results.append(public)
            selected_items.append(item)
        diagnostic_target.update(
            {
                "candidate_count_before_filter": candidate_count_before_actor_filter,
                "candidate_count_after_entity_filter": candidate_count_after_actor_filter,
                "candidate_count_after_voice_actor_filter": candidate_count_after_actor_filter,
                "verified_voice_actor_matches": candidate_count_after_actor_filter,
                "verified_entity_matches": candidate_count_after_actor_filter if has_required_entity else 0,
                "returned_verified_match_count": len(results) if has_required_voice_actor else 0,
                "returned_verified_entity_match_count": len(results) if has_required_entity else 0,
            }
        )
        diagnostic_target["timing_ms"]["diversity_reranking"] = round(
            (time.perf_counter() - reranking_started) * 1000,
            3,
        )
        diagnostic_target["timing_ms"]["total"] = round(
            (time.perf_counter() - recommend_started) * 1000,
            3,
        )
        return results

    def resolve_voice_actor_constraints(
        self,
        names: list[str],
        entity_ids: list[int] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        resolved: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        seen: set[tuple[int | None, str]] = set()

        for entity_id in entity_ids or []:
            record = self.voice_actor_by_id.get(int(entity_id))
            if not record:
                errors.append({"type": "unknown_voice_actor_id", "entity_id": int(entity_id)})
                continue
            key = (record.get("entity_id"), normalize_label(record.get("matched_name")))
            if key not in seen:
                resolved.append(dict(record))
                seen.add(key)

        for name in names:
            candidates: dict[tuple[int | None, str], dict[str, Any]] = {}
            for variant in entity_name_variants(name):
                for record in self.voice_actor_by_key.get(variant, []):
                    key = (record.get("entity_id"), normalize_label(record.get("matched_name")))
                    candidates[key] = record
            unresolved = [record for key, record in candidates.items() if key not in seen]
            if not unresolved and any(
                entity_name_variants(name).intersection(entity_name_variants(record.get("matched_name")))
                for record in resolved
            ):
                continue
            if not unresolved:
                errors.append({"type": "unknown_voice_actor", "input": name})
                continue
            if len(unresolved) > 1:
                errors.append(
                    {
                        "type": "ambiguous_voice_actor",
                        "input": name,
                        "matches": [
                            {"entity_id": record.get("entity_id"), "matched_name": record.get("matched_name")}
                            for record in unresolved
                        ],
                    }
                )
                continue
            record = dict(unresolved[0])
            key = (record.get("entity_id"), normalize_label(record.get("matched_name")))
            resolved.append(record)
            seen.add(key)

        return resolved, errors

    def _build_voice_actor_index(
        self,
    ) -> tuple[dict[int, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        grouped: dict[tuple[int | None, str], dict[str, Any]] = {}
        for anime_id, roles in self.voice_actor_roles_by_anime.items():
            for role in roles:
                name = str(role.get("voice_actor") or "").strip()
                if not name:
                    continue
                entity_id = role.get("voice_actor_id")
                key = (int(entity_id) if entity_id is not None else None, normalize_label(name))
                record = grouped.setdefault(
                    key,
                    {
                        "entity_id": key[0],
                        "entity_type": "voice_actor",
                        "matched_name": name,
                        "related_anime_ids": set(),
                    },
                )
                record["related_anime_ids"].add(anime_id)

        by_id: dict[int, dict[str, Any]] = {}
        by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in grouped.values():
            public = {
                **record,
                "related_anime_ids": sorted(record["related_anime_ids"]),
            }
            entity_id = public.get("entity_id")
            if entity_id is not None:
                by_id[int(entity_id)] = public
            for variant in entity_name_variants(public["matched_name"]):
                by_key[variant].append(public)
        return by_id, dict(by_key)

    def _voice_actor_roles(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        roles: list[dict[str, Any]] = []
        seen: set[tuple[int | None, str, int | None, str, str]] = set()

        def add(
            actor_id: Any,
            actor_name: Any,
            character_id: Any,
            character_name: Any,
            language: Any,
        ) -> None:
            name = str(actor_name or "").strip()
            if not name:
                return
            try:
                normalized_actor_id = int(actor_id) if actor_id is not None else None
            except (TypeError, ValueError):
                normalized_actor_id = None
            try:
                normalized_character_id = int(character_id) if character_id is not None else None
            except (TypeError, ValueError):
                normalized_character_id = None
            character = str(character_name or "").strip()
            spoken_language = str(language or "Unknown").strip() or "Unknown"
            key = (
                normalized_actor_id,
                normalize_label(name),
                normalized_character_id,
                normalize_label(character),
                spoken_language.casefold(),
            )
            if key in seen:
                return
            seen.add(key)
            roles.append(
                {
                    "voice_actor_id": normalized_actor_id,
                    "voice_actor": name,
                    "character_id": normalized_character_id,
                    "character": character or None,
                    "language": spoken_language,
                }
            )

        for role in item.get("voice_actor_roles", []):
            add(
                role.get("voice_actor_id"),
                role.get("voice_actor"),
                role.get("character_id"),
                role.get("character"),
                role.get("language"),
            )
        for character in item.get("characters", []):
            for actor in character.get("voice_actors", []):
                add(
                    actor.get("id"),
                    actor.get("name"),
                    character.get("id"),
                    character.get("name"),
                    actor.get("language"),
                )
        actor_keys_with_roles = {
            (role.get("voice_actor_id"), normalize_label(role.get("voice_actor"))) for role in roles
        }
        for actor in item.get("voice_actors", []):
            try:
                actor_id = int(actor.get("id")) if actor.get("id") is not None else None
            except (TypeError, ValueError):
                actor_id = None
            actor_key = (actor_id, normalize_label(actor.get("name")))
            if actor_key not in actor_keys_with_roles:
                add(actor_id, actor.get("name"), None, None, actor.get("language"))
        return roles

    def _matched_required_voice_actor_roles(
        self,
        anime_id: int,
        required_actors: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not required_actors:
            return []
        anime_roles = self.voice_actor_roles_by_anime.get(int(anime_id), [])
        matched: list[dict[str, Any]] = []
        for required in required_actors:
            entity_id = required.get("entity_id")
            required_variants = entity_name_variants(required.get("matched_name"))
            actor_matches = [
                role
                for role in anime_roles
                if (entity_id is not None and role.get("voice_actor_id") == int(entity_id))
                or (entity_id is None and required_variants.intersection(entity_name_variants(role.get("voice_actor"))))
            ]
            if not actor_matches:
                return []
            matched.extend(actor_matches)
        return matched

    @staticmethod
    def _normalize_required_entity_constraints(
        constraints: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, int | None, str]] = set()
        for constraint in constraints or []:
            entity_type = str(constraint.get("entity_type") or "").strip().casefold()
            matched_name = str(constraint.get("matched_name") or constraint.get("name") or "").strip()
            if not entity_type or not matched_name:
                continue
            try:
                entity_id = int(constraint["entity_id"]) if constraint.get("entity_id") is not None else None
            except (TypeError, ValueError):
                entity_id = None
            related_anime_ids: set[int] = set()
            for value in constraint.get("related_anime_ids", []):
                try:
                    related_anime_ids.add(int(value))
                except (TypeError, ValueError):
                    continue
            key = (entity_type, entity_id, normalize_label(matched_name))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "matched_name": matched_name,
                    "related_anime_ids": related_anime_ids,
                }
            )
        return normalized

    def _required_entity_evidence(
        self,
        item: dict[str, Any],
        *,
        required_studio_keys: set[str],
        required_staff_keys: set[str],
        required_character_keys: set[str],
        matched_voice_actor_roles: list[dict[str, Any]],
        generic_constraints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        seen: set[tuple[str, int | None, str, str]] = set()

        def add(
            entity_type: str,
            name: Any,
            entity_id: Any = None,
            *,
            relationship: str,
            role: Any = None,
            character: Any = None,
            language: Any = None,
        ) -> None:
            text = str(name or "").strip()
            if not text:
                return
            try:
                normalized_id = int(entity_id) if entity_id is not None else None
            except (TypeError, ValueError):
                normalized_id = None
            key = (entity_type, normalized_id, normalize_label(text), str(role or "").casefold())
            if key in seen:
                return
            seen.add(key)
            record: dict[str, Any] = {
                "entity_type": entity_type,
                "name": text,
                "entity_id": normalized_id,
                "relationship": relationship,
            }
            if role:
                record["role"] = str(role)
            if character:
                record["character"] = str(character)
            if language:
                record["language"] = str(language)
            evidence.append(record)

        studio_relationships = item.get("studio_relationships", [])
        for studio in item.get("studios", []):
            if normalize_label(studio) not in required_studio_keys:
                continue
            company = next(
                (
                    value
                    for value in studio_relationships
                    if normalize_label(value.get("name")) == normalize_label(studio)
                ),
                {},
            )
            add("studio", studio, company.get("id"), relationship="studio")

        staff_relationships = item.get("staff_relationships", item.get("staff", []))
        for person in staff_relationships:
            if normalize_label(person.get("name")) in required_staff_keys:
                add(
                    "staff",
                    person.get("name"),
                    person.get("id"),
                    relationship="staff_credit",
                    role=person.get("role"),
                )

        character_relationships = item.get("character_relationships", item.get("characters", []))
        for character_record in character_relationships:
            if normalize_label(character_record.get("name")) in required_character_keys:
                add(
                    "character",
                    character_record.get("name"),
                    character_record.get("id"),
                    relationship="character_appearance",
                    role=character_record.get("role"),
                )

        for actor_role in matched_voice_actor_roles:
            add(
                "voice_actor",
                actor_role.get("voice_actor"),
                actor_role.get("voice_actor_id"),
                relationship="voice_role",
                character=actor_role.get("character"),
                language=actor_role.get("language"),
            )

        for constraint in generic_constraints:
            entity_type = constraint["entity_type"]
            name = constraint["matched_name"]
            name_key = normalize_label(name)
            entity_id = constraint.get("entity_id")
            if entity_type == "studio":
                add("studio", name, entity_id, relationship="studio")
            elif entity_type == "producer":
                producer = next(
                    (
                        value
                        for value in item.get("producer_relationships", [])
                        if normalize_label(value.get("name")) == name_key
                    ),
                    {},
                )
                add(
                    "producer",
                    name,
                    entity_id or producer.get("id"),
                    relationship="production_credit",
                    role=producer.get("role") or "Producer",
                )
            elif entity_type in {"staff", "director", "original_creator"}:
                person = next(
                    (value for value in staff_relationships if normalize_label(value.get("name")) == name_key),
                    {},
                )
                add(
                    entity_type,
                    name,
                    entity_id or person.get("id"),
                    relationship="staff_credit",
                    role=person.get("role") or entity_type.replace("_", " ").title(),
                )
            elif entity_type == "character":
                character_record = next(
                    (value for value in character_relationships if normalize_label(value.get("name")) == name_key),
                    {},
                )
                add(
                    "character",
                    name,
                    entity_id or character_record.get("id"),
                    relationship="character_appearance",
                    role=character_record.get("role"),
                )
            elif entity_type == "voice_actor":
                for actor_role in matched_voice_actor_roles:
                    if normalize_label(actor_role.get("voice_actor")) == name_key:
                        add(
                            "voice_actor",
                            name,
                            entity_id or actor_role.get("voice_actor_id"),
                            relationship="voice_role",
                            character=actor_role.get("character"),
                            language=actor_role.get("language"),
                        )
            elif entity_type in {"genre", "theme", "demographic"}:
                add(entity_type, name, entity_id, relationship="catalog_label")
            elif entity_type == "anime":
                add("anime", name, entity_id, relationship="catalog_relation")
        return evidence

    @staticmethod
    def _matched_entity_names(evidence: list[dict[str, Any]], entity_type: str) -> list[str]:
        return list(
            dict.fromkeys(
                str(value["name"])
                for value in evidence
                if value.get("entity_type") == entity_type and value.get("name")
            )
        )

    def details(self, anime_id: int) -> dict[str, Any] | None:
        item = self.by_id.get(int(anime_id))
        return self.public_item(item, include_synopsis=True, include_people=True) if item else None

    def explain(
        self,
        item: dict[str, Any],
        liked_ids: list[int] | None = None,
        selected_genres: list[str] | None = None,
        query: str | None = None,
        signals: dict[str, float] | None = None,
        matched_voice_actor_roles: list[dict[str, Any]] | None = None,
        matched_required_studios: list[str] | None = None,
        matched_required_staff: list[str] | None = None,
        matched_required_characters: list[str] | None = None,
        matched_required_entities: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        reasons: list[str] = []
        signals = signals or {}
        item_genres = set(item.get("genres", []))
        selected = {genre for genre in selected_genres or [] if genre in item_genres}

        if matched_voice_actor_roles:
            role = matched_voice_actor_roles[0]
            character = role.get("character") or "a catalog character"
            language = role.get("language") or "Unknown"
            reasons.append(f"Verified voice-actor relationship: {role['voice_actor']} voices {character} ({language}).")
        if matched_required_studios:
            reasons.append("Verified studio relationship: " + ", ".join(matched_required_studios[:2]) + ".")
        if matched_required_staff:
            reasons.append("Verified staff relationship: " + ", ".join(matched_required_staff[:2]) + ".")
        if matched_required_characters:
            reasons.append("Verified character relationship: " + ", ".join(matched_required_characters[:2]) + ".")
        specialized_types = {"studio", "staff", "character", "voice_actor"}
        for relationship in matched_required_entities or []:
            entity_type = str(relationship.get("entity_type") or "")
            if entity_type in specialized_types:
                continue
            label = entity_type.replace("_", " ")
            detail = str(relationship.get("name") or "")
            role = relationship.get("role")
            if role:
                detail += f" ({role})"
            reasons.append(f"Verified {label} relationship: {detail}.")

        if selected:
            reasons.append("Matches your requested genres: " + ", ".join(sorted(selected)[:3]))

        liked_genres: Counter[str] = Counter()
        liked_studios: Counter[str] = Counter()
        liked_creators: Counter[int] = Counter()
        liked_staff: Counter[int] = Counter()
        liked_voice_actors: Counter[int] = Counter()
        liked_types: Counter[str] = Counter()
        liked_themes: Counter[str] = Counter()
        liked_story_tokens: Counter[str] = Counter()

        for liked_id in liked_ids or []:
            liked = self.by_id.get(int(liked_id))
            if not liked:
                continue
            liked_genres.update(liked.get("genres", []))
            liked_studios.update(liked.get("studios", []))
            liked_creators.update(int(person["id"]) for person in self._selected_creators(liked))
            liked_staff.update(int(person["id"]) for person in liked.get("staff", []))
            liked_voice_actors.update(int(person["id"]) for person in liked.get("voice_actors", []))
            if liked.get("type"):
                liked_types.update([liked["type"]])
            liked_themes.update(story_themes(story_text(liked)))
            liked_story_tokens.update(tokenize(liked.get("synopsis")))

        item_themes = story_themes(story_text(item))
        query_themes = story_themes(query)
        shared_genres = [genre for genre in item.get("genres", []) if liked_genres[genre]]
        shared_studios = [studio for studio in item.get("studios", []) if liked_studios[studio]]
        shared_creators = [
            person["name"] for person in self._selected_creators(item) if liked_creators[int(person["id"])]
        ]
        shared_staff = [person["name"] for person in item.get("staff", []) if liked_staff[int(person["id"])]]
        shared_voice_actors = [
            person["name"] for person in item.get("voice_actors", []) if liked_voice_actors[int(person["id"])]
        ]
        shared_themes = [theme for theme in item_themes if liked_themes[theme]]
        requested_themes = [theme for theme in item_themes if theme in query_themes]
        generic_story_words = {
            "anime",
            "becomes",
            "however",
            "life",
            "people",
            "story",
            "their",
            "world",
            "young",
        }
        shared_story_words = [
            token
            for token in set(tokenize(item.get("synopsis")))
            if liked_story_tokens[token] and len(token) > 3 and token not in generic_story_words
        ]
        shared_story_words.sort(key=lambda token: self.story_idf.get(token, 0.0), reverse=True)

        if requested_themes:
            reasons.append("Matches requested story themes: " + ", ".join(requested_themes[:2]))

        connections = []
        if shared_creators:
            connections.append("creator or director " + ", ".join(shared_creators[:2]))
        if shared_studios:
            connections.append("studio " + ", ".join(shared_studios[:2]))
        if shared_themes:
            connections.append("themes of " + ", ".join(shared_themes[:2]))
        if shared_genres:
            connections.append("genres including " + ", ".join(shared_genres[:3]))
        if item.get("type") and liked_types[item["type"]]:
            connections.append(f"the same {item['type']} format")
        if shared_story_words and len(connections) < 2:
            connections.append("story elements such as " + ", ".join(shared_story_words[:3]))

        if connections:
            reasons.append("Connects to your examples through " + natural_join(connections[:4]) + ".")
        if shared_staff and not shared_creators:
            reasons.append("Shares credited staff: " + ", ".join(shared_staff[:2]))
        if shared_voice_actors:
            reasons.append("Shares Japanese voice cast: " + ", ".join(shared_voice_actors[:2]))
        if signals.get("collaborative", 0.0) >= 0.20:
            reasons.append("People with similar rating patterns also connected this title to your examples.")
        if item.get("score"):
            reasons.append(f"Strong audience score: {item['score']:.2f}")

        if not reasons:
            reasons.append("High-quality catalog match based on score and popularity")

        return reasons[:3]

    def explanation_data(
        self,
        item: dict[str, Any],
        liked_ids: list[int],
        selected_genres: list[str],
        min_score: float | None,
        min_year: int | None,
        max_year: int | None,
        max_episodes: int | None,
        formats: list[str],
        signals: dict[str, float],
        contributions: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        item_genres = set(item.get("genres", []))
        reference_items = [self.by_id[anime_id] for anime_id in liked_ids if anime_id in self.by_id]
        liked_genres = {genre for reference in reference_items for genre in reference.get("genres", [])}
        liked_themes = {theme for reference in reference_items for theme in story_themes(story_text(reference))}
        item_themes = set(story_themes(story_text(item)))
        liked_creators = {
            person["name"]
            for reference in reference_items
            for person in self._selected_creators(reference)
            if person.get("name")
        }
        matched_creators = [
            person["name"] for person in self._selected_creators(item) if person.get("name") in liked_creators
        ]
        matched_studios = [
            studio
            for studio in item.get("studios", [])
            if any(studio in reference.get("studios", []) for reference in reference_items)
        ]
        constraints = []
        if min_score is not None:
            constraints.append(f"score at least {min_score:g}")
        if min_year is not None:
            constraints.append(f"released in or after {min_year}")
        if max_year is not None:
            constraints.append(f"released in or before {max_year}")
        if max_episodes is not None:
            constraints.append(f"{max_episodes} episodes or fewer")
        if formats:
            constraints.append("format: " + ", ".join(sorted(formats)))

        channel_values = contributions or signals
        ranked_channels = [
            channel
            for channel, _value in sorted(
                ((key, value) for key, value in channel_values.items() if value > 0),
                key=lambda pair: pair[1],
                reverse=True,
            )[:3]
        ]

        return {
            "matched_genres": [genre for genre in selected_genres if genre in item_genres],
            "shared_genres": sorted(item_genres.intersection(liked_genres))[:5],
            "shared_story_themes": sorted(item_themes.intersection(liked_themes))[:5],
            "matched_studios": matched_studios[:3],
            "matched_creators": matched_creators[:3],
            "reference_titles": [reference["title"] for reference in reference_items],
            "reference_context": [
                {
                    "title": reference["title"],
                    "type": reference.get("type"),
                    "genres": reference.get("genres", []),
                    "studios": reference.get("studios", []),
                    "creators": [person.get("name") for person in self._selected_creators(reference)[:3]],
                    "synopsis": (reference.get("synopsis") or "")[:260],
                }
                for reference in reference_items[:4]
            ],
            "matched_constraints": constraints,
            "strongest_channels": ranked_channels,
        }

    def public_item(
        self,
        item: dict[str, Any],
        include_synopsis: bool = False,
        include_people: bool = False,
    ) -> dict[str, Any]:
        synopsis = item.get("synopsis") or ""
        if not include_synopsis and len(synopsis) > 220:
            synopsis = synopsis[:217].rstrip() + "..."

        people_limit = 12 if include_people else 4

        return {
            "id": item["id"],
            "anime_id": item["id"],
            "title": item["title"],
            "score": item.get("score"),
            "rank": item.get("rank"),
            "popularity": item.get("popularity"),
            "members": item.get("members"),
            "synopsis": synopsis,
            "start_year": item.get("start_year"),
            "year": item.get("start_year"),
            "type": item.get("type"),
            "format": item.get("type"),
            "episodes": item.get("episodes"),
            "image_url": item.get("image_url"),
            "genres": item.get("genres", []),
            "genre_groups": item.get("genre_groups", {}),
            "studios": item.get("studios", []),
            "studio": (item.get("studios") or [None])[0],
            "producers": item.get("producers", []),
            "characters": item.get("characters", [])[:people_limit],
            "staff": item.get("staff", [])[:people_limit],
            "creators": item.get("creators", [])[:people_limit],
            "voice_actors": item.get("voice_actors", [])[:people_limit],
        }

    def _build_metadata_vector(self, item: dict[str, Any]) -> Counter[str]:
        vector: Counter[str] = Counter()

        for token in item.get("metadata_tokens", []):
            token_key = normalize_label(str(token).replace("_", " "))
            if not token_key:
                continue
            if str(token).startswith(("genre_", "theme_", "demographic_")):
                vector[f"metadata:{token_key}"] += 8.0
            elif str(token).startswith("type_"):
                vector[f"metadata:{token_key}"] += 3.0
            else:
                vector[f"metadata:{token_key}"] += 2.0

        for group, values in item.get("genre_groups", {}).items():
            group_key = normalize_label(group) or "genre"
            for value in values:
                label_key = normalize_label(value)
                if label_key:
                    vector[f"{group_key}:{label_key}"] += 7.0

        for genre in item.get("genres", []):
            key = normalize_label(genre)
            if key:
                vector[f"genre:{key}"] += 8.0
            for token in tokenize(genre):
                vector[token] += 2.0

        if item.get("type"):
            vector[f"type:{normalize_label(item['type'])}"] += 3.0

        for studio in item.get("studios", []):
            key = normalize_label(studio)
            if key:
                vector[f"studio:{key}"] += 3.0
        for person in item.get("staff", [])[:16]:
            key = normalize_label(person.get("name"))
            if key:
                vector[f"staff:{key}"] += 3.0
        for person in item.get("characters", [])[:16]:
            key = normalize_label(person.get("name"))
            if key:
                vector[f"character:{key}"] += 3.0
        for person in item.get("voice_actors", [])[:16]:
            key = normalize_label(person.get("name"))
            if key:
                vector[f"voice_actor:{key}"] += 3.0

        return vector

    def _build_creator_vector(self, item: dict[str, Any]) -> Counter[str]:
        vector: Counter[str] = Counter()

        for studio in item.get("studios", []):
            if self.studio_counts[studio] < self.min_entity_frequency:
                continue
            key = normalize_label(studio)
            if key:
                vector[f"studio:{key}"] += 4.0

        for person in self._selected_creators(item):
            name = person.get("name", "")
            if self.creator_counts[name] < self.min_entity_frequency:
                continue
            key = normalize_label(person.get("name"))
            roles = creator_roles(person.get("role"))
            if key and roles:
                for role in roles:
                    vector[f"{role}:{key}"] += 4.0
            elif key:
                vector[f"creator:{key}"] += 3.4

        for producer in item.get("producers", [])[:4]:
            if isinstance(producer, dict):
                producer = producer.get("name")
            key = normalize_label(producer)
            if key:
                vector[f"producer:{key}"] += 0.45

        return vector

    def _selected_creators(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        selected = []
        for person in item.get("creators", []) or item.get("staff", []):
            if creator_roles(person.get("role")):
                selected.append(person)
        return selected

    def _build_story_vector(self, item: dict[str, Any]) -> Counter[str]:
        vector: Counter[str] = Counter()

        for token in tokenize(item.get("title")):
            vector[token] += 1.6

        for token in tokenize(item.get("synopsis"))[:360]:
            vector[token] += 1.2

        for theme in story_themes(story_text(item)):
            vector[f"theme:{theme}"] += 6.0

        return vector

    def _build_embedding_index(self):
        if np is None:
            return None

        matrix = np.zeros((len(self.catalog), EMBEDDING_DIMENSIONS), dtype=np.float32)
        for index, item in enumerate(self.catalog):
            matrix[index] = dense_embedding(story_text(item), EMBEDDING_DIMENSIONS)

        return normalize_rows(matrix)

    def _build_svd_index(self):
        if np is None or not self.story_vectors:
            return None

        feature_scores: Counter[str] = Counter()
        for vector in self.story_vectors.values():
            feature_scores.update(vector)

        self.svd_features = [feature for feature, _ in feature_scores.most_common(SVD_FEATURES)]
        if not self.svd_features:
            return None

        matrix = np.zeros((len(self.catalog), len(self.svd_features)), dtype=np.float32)
        self.svd_feature_index = {feature: index for index, feature in enumerate(self.svd_features)}

        for row, anime_id in enumerate(self.item_ids):
            vector = self.story_vectors[anime_id]
            for feature, value in vector.items():
                column = self.svd_feature_index.get(feature)
                if column is not None:
                    matrix[row, column] = float(value)

        matrix = normalize_rows(matrix)
        if not matrix.size:
            return None

        covariance = matrix.T @ matrix
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        dimensions = min(SVD_DIMENSIONS, len(order))
        if dimensions <= 0:
            return None

        self.svd_components = vectors[:, order[:dimensions]].astype(np.float32)
        latent = matrix @ self.svd_components
        return normalize_rows(latent)

    def _story_dense_vector(self, vector: Counter[str]):
        if np is None or not self.svd_features:
            return None

        dense = np.zeros(len(self.svd_features), dtype=np.float32)
        for feature, value in vector.items():
            column = self.svd_feature_index.get(feature)
            if column is not None:
                dense[column] = float(value)
        return dense

    def _embedding_profile(self, liked_ids: list[int], query: str | None, genres: list[str]):
        if np is None or self.embedding_vectors is None:
            return None

        vectors = []
        for anime_id in liked_ids:
            index = self.index_by_id.get(int(anime_id))
            if index is not None:
                vectors.append(self.embedding_vectors[index] * 1.4)

        query_text = " ".join([query or "", " ".join(genres or [])]).strip()
        if query_text:
            vectors.append(dense_embedding(query_text, EMBEDDING_DIMENSIONS))

        if not vectors:
            return None

        return normalize_vector(np.sum(np.vstack(vectors), axis=0))

    def _semantic_profile(self, liked_ids: list[int], query: str | None, genres: list[str]):
        if np is None or self.semantic_index is None:
            return None

        vectors = []
        for anime_id in liked_ids:
            vector = self.semantic_index.document_vector(int(anime_id))
            if vector is not None:
                vectors.append(vector * 1.4)

        query_text = " ".join([query or "", " ".join(genres or [])]).strip()
        if query_text:
            vector = self.semantic_index.encode_query(query_text)
            if vector is not None:
                vectors.append(vector)

        if not vectors:
            return None

        return normalize_vector(np.sum(np.vstack(vectors), axis=0))

    def _svd_profile(self, liked_ids: list[int], story_profile: Counter[str]):
        if np is None or self.svd_vectors is None or self.svd_components is None:
            return None

        vectors = []
        for anime_id in liked_ids:
            index = self.index_by_id.get(int(anime_id))
            if index is not None:
                vectors.append(self.svd_vectors[index] * 1.4)

        dense_story = self._story_dense_vector(story_profile)
        if dense_story is not None:
            dense_story = normalize_vector(dense_story)
            if dense_story is not None:
                vectors.append(dense_story @ self.svd_components)

        if not vectors:
            return None

        return normalize_vector(np.sum(np.vstack(vectors), axis=0))

    def _build_idf(self, vectors: Iterable[Counter[str]]) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        total = max(len(self.catalog), 1)

        for vector in vectors:
            document_frequency.update(vector.keys())

        return {key: math.log((1 + total) / (1 + frequency)) + 1.0 for key, frequency in document_frequency.items()}

    def _apply_idf(self, vector: Counter[str], idf: dict[str, float] | None = None) -> Counter[str]:
        idf = idf or self.idf
        return Counter({key: value * idf.get(key, 1.0) for key, value in vector.items()})

    def _add_profile_feature(
        self,
        profile: Counter[str],
        key: str,
        weight: float,
        idf: dict[str, float] | None = None,
    ) -> None:
        if key.strip(":"):
            idf = idf or self.idf
            profile[key] += weight * idf.get(key, 1.0)

    def _passes_filters(
        self,
        item: dict[str, Any],
        selected_genres: set[str] | None = None,
        blocked_genres: set[str] | None = None,
        format_keys: set[str] | None = None,
        min_score: float | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        max_episodes: int | None = None,
    ) -> bool:
        selected_genres = selected_genres or set()
        blocked_genres = blocked_genres or set()
        format_keys = format_keys or set()
        item_genres = {genre.casefold() for genre in item.get("genres", [])}
        item_type = (item.get("type") or "").casefold()

        if selected_genres and not selected_genres.intersection(item_genres):
            return False
        if blocked_genres and blocked_genres.intersection(item_genres):
            return False
        if format_keys and item_type not in format_keys:
            return False
        if min_score is not None and item.get("score") is not None and float(item["score"]) < min_score:
            return False
        if min_score is not None and item.get("score") is None:
            return False
        if min_year is not None and item.get("start_year") is not None and int(item["start_year"]) < min_year:
            return False
        if min_year is not None and item.get("start_year") is None:
            return False
        if max_year is not None and item.get("start_year") is not None and int(item["start_year"]) > max_year:
            return False
        if max_year is not None and item.get("start_year") is None:
            return False
        if max_episodes is not None and item.get("episodes") is not None and int(item["episodes"]) > max_episodes:
            return False
        if max_episodes is not None and item.get("episodes") is None:
            return False

        return True

    def _genre_match_score(self, item: dict[str, Any], selected_genres: set[str]) -> float:
        if not selected_genres:
            return 0.0

        item_genres = {genre.casefold() for genre in item.get("genres", [])}
        return len(selected_genres.intersection(item_genres)) / max(len(selected_genres), 1)

    def _quality_bonus(self, item: dict[str, Any]) -> float:
        raw_score = float(item.get("score") or 0) / 10.0
        members = math.log1p(int(item.get("members") or 0)) / math.log1p(max(self.max_members, 1))
        confidence = members / (members + 0.35) if members else 0.0
        score = self.global_score_mean * (1.0 - confidence) + raw_score * confidence

        popularity = item.get("popularity")
        if popularity:
            popularity_score = 1.0 - (math.log1p(int(popularity)) / math.log1p(max(self.max_popularity, 1)))
        else:
            popularity_score = 0.0

        catalog_quality = positive_score(score * 0.70 + members * 0.18 + popularity_score * 0.12)
        collaborative_quality = (
            self.collaborative_index.quality_score(int(item["id"])) if self.collaborative_index is not None else None
        )
        if collaborative_quality is None:
            return catalog_quality
        return positive_score(catalog_quality * 0.65 + collaborative_quality * 0.35)

    def _novelty_score(self, item: dict[str, Any], preference: str) -> float:
        if preference == "neutral":
            return 0.0
        members = math.log1p(int(item.get("members") or 0)) / math.log1p(max(self.max_members, 1))
        popularity = item.get("popularity")
        popularity_fame = (
            1.0 - math.log1p(int(popularity)) / math.log1p(max(self.max_popularity, 1)) if popularity else 0.0
        )
        fame = positive_score(members * 0.70 + popularity_fame * 0.30)
        return 1.0 - fame if preference == "less_famous" else fame

    def _has_session_signal(self, session_profile: dict[str, Any]) -> bool:
        fields = (
            "liked_titles",
            "preferred_genres",
            "preferred_studios",
            "preferred_staff",
            "preferred_characters",
            "preferred_voice_actors",
        )
        return any(session_profile.get(field) for field in fields) or bool(session_profile.get("temporary_ratings"))

    def _session_score(
        self,
        item: dict[str, Any],
        liked_ids: list[int],
        session_profile: dict[str, Any],
        selected_genres: set[str],
        blocked_genres: set[str],
        *,
        history: tuple[Counter[str], Counter[str], Counter[str], Counter[str]] | None = None,
    ) -> float:
        item_genres = {genre.casefold() for genre in item.get("genres", [])}
        if blocked_genres and item_genres.intersection(blocked_genres):
            return 0.0

        liked_genres, liked_studios, liked_creators, liked_types = history or self._session_history(liked_ids)

        preferred_genres = {
            str(value).casefold() for value in session_profile.get("preferred_genres", []) or [] if value
        }
        excluded_genres = {str(value).casefold() for value in session_profile.get("excluded_genres", []) or [] if value}
        if item_genres.intersection(excluded_genres):
            return 0.0

        score = 0.0
        if preferred_genres:
            score += 0.30 * len(item_genres.intersection(preferred_genres)) / max(len(preferred_genres), 1)
        if liked_genres:
            shared = sum(liked_genres[genre] for genre in item_genres)
            score += 0.28 * min(shared / max(sum(liked_genres.values()), 1), 1.0)

        item_studios = {studio.casefold() for studio in item.get("studios", [])}
        if item_studios and liked_studios:
            score += 0.12 * min(sum(liked_studios[studio] for studio in item_studios), 2) / 2

        item_creators = {
            person.get("name", "").casefold() for person in self._selected_creators(item) if person.get("name")
        }
        if item_creators and liked_creators:
            score += 0.12 * min(sum(liked_creators[person] for person in item_creators), 2) / 2

        item_type = (item.get("type") or "").casefold()
        if item_type and liked_types[item_type]:
            score += 0.08

        temporary_ratings = session_profile.get("temporary_ratings", {}) or {}
        title_key = item.get("title", "").casefold()
        for title, rating in temporary_ratings.items():
            if str(title).casefold() == title_key:
                score += 0.10 * max(-1.0, min(1.0, float(rating)))

        return positive_score(score)

    def _session_history(
        self,
        liked_ids: list[int],
    ) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str]]:
        liked_genres: Counter[str] = Counter()
        liked_studios: Counter[str] = Counter()
        liked_creators: Counter[str] = Counter()
        liked_types: Counter[str] = Counter()

        for anime_id in liked_ids:
            liked = self.by_id.get(int(anime_id))
            if not liked:
                continue
            liked_genres.update(genre.casefold() for genre in liked.get("genres", []))
            liked_studios.update(studio.casefold() for studio in liked.get("studios", []))
            liked_creators.update(
                person.get("name", "").casefold() for person in self._selected_creators(liked) if person.get("name")
            )
            if liked.get("type"):
                liked_types.update([(liked["type"] or "").casefold()])
        return liked_genres, liked_studios, liked_creators, liked_types

    def _diversity_penalty(self, item: dict[str, Any], selected_items: list[dict[str, Any]]) -> float:
        if not selected_items:
            return 0.0

        item_genres = set(item.get("genres", []))
        item_studios = set(item.get("studios", []))
        item_series = series_key(item.get("title", ""))
        penalties = []

        for selected in selected_items:
            selected_genres = set(selected.get("genres", []))
            selected_studios = set(selected.get("studios", []))
            genre_union = len(item_genres.union(selected_genres))
            genre_overlap = len(item_genres.intersection(selected_genres)) / genre_union if genre_union else 0.0
            studio_overlap = 1.0 if item_studios.intersection(selected_studios) else 0.0
            series_overlap = 1.0 if item_series and item_series == series_key(selected.get("title", "")) else 0.0
            semantic_overlap = 0.0

            item_index = self.index_by_id.get(int(item["id"]))
            selected_index = self.index_by_id.get(int(selected["id"]))
            if item_index is not None and selected_index is not None and self.svd_vectors is not None:
                semantic_overlap = positive_score(
                    dense_cosine(self.svd_vectors[item_index], self.svd_vectors[selected_index])
                )

            penalties.append(
                series_overlap * 0.65 + genre_overlap * 0.20 + studio_overlap * 0.10 + semantic_overlap * 0.05
            )

        return min(1.0, sum(penalties) / max(len(penalties), 1))


def series_key(title: str) -> str:
    base = re.split(
        r"\s*[:：]\s*|\s+-\s+|\b(?:movie|season|specials?|ova|ona|recap|remix|part|chapter|episode|episodes)\b",
        title.casefold(),
        maxsplit=1,
    )[0]
    tokens = tokenize(base)
    key: list[str] = []

    for token in tokens:
        if token in SERIES_STOP_TOKENS or re.fullmatch(r"\d+(?:st|nd|rd|th)?", token):
            break
        key.append(token)

    if len(key) >= 2:
        return " ".join(key)
    return " ".join(tokens[:2]) if tokens else title.casefold()
