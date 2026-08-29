from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from .data_pipeline import clean_label, dedupe, fix_text, metadata_token, to_float, to_int

ARCHIVE_FILES = {
    "anime": "anime.csv",
    "synopsis": "anime_with_synopsis.csv",
    "ratings": "rating_complete.csv",
    "lists": "animelist.csv",
    "statuses": "watching_status.csv",
}
RELATIONSHIP_FIELDS = (
    "characters",
    "character_relationships",
    "character_names",
    "staff",
    "staff_relationships",
    "creators",
    "voice_actors",
    "voice_actor_roles",
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def is_archive_dataset(path: Path) -> bool:
    return all((path / filename).exists() for filename in ARCHIVE_FILES.values())


def find_archive_dir(project_root: Path) -> Path:
    candidates = (
        project_root / "archive",
        project_root / "data" / "raw" / "archive",
        project_root,
    )
    for candidate in candidates:
        if is_archive_dataset(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find the Anime Recommendation Database 2020 files. Expected: {', '.join(ARCHIVE_FILES.values())}"
    )


def _rows(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        yield from csv.DictReader(file)


def _split_labels(value: object) -> list[str]:
    text = fix_text(value)
    if not text or text.casefold() == "unknown":
        return []
    return dedupe([clean_label(part) for part in text.split(",") if clean_label(part)])


def _parse_year(*values: object) -> int | None:
    for value in values:
        match = YEAR_RE.search(str(value or ""))
        if match:
            return int(match.group())
    return None


def _score_distribution(row: dict[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for score in range(1, 11):
        count = to_int(row.get(f"Score-{score}"))
        if count is not None and count >= 0:
            result[str(score)] = count
    return result


def _anonymous_relationships(names: list[str], role: str) -> list[dict[str, Any]]:
    return [{"id": None, "name": name, "role": role} for name in names]


def _enrichment_relationships(
    legacy: dict[str, Any] | None,
    field: str,
) -> list[Any]:
    if not legacy:
        return []
    values = legacy.get(field)
    return list(values) if isinstance(values, list) else []


def build_archive_catalog(
    archive_dir: Path,
    *,
    enrichment_catalog: list[dict[str, Any]] | None = None,
    include_adult: bool = False,
    include_legacy_only: bool = True,
) -> list[dict[str, Any]]:
    """Build the application catalog from the 2020 archive.

    The archive is the primary metadata and collaborative-data source. Existing
    application records are used only to retain fields the archive does not
    provide (posters, characters, cast, staff, and post-2020 titles).
    """

    synopsis_by_id: dict[int, str] = {}
    for row in _rows(archive_dir / ARCHIVE_FILES["synopsis"]):
        anime_id = to_int(row.get("MAL_ID"))
        if anime_id is not None:
            synopsis_by_id[anime_id] = fix_text(row.get("sypnopsis"))

    legacy_by_id = {int(item["id"]): item for item in enrichment_catalog or [] if item.get("id") is not None}
    archive_ids: set[int] = set()
    catalog: list[dict[str, Any]] = []

    for row in _rows(archive_dir / ARCHIVE_FILES["anime"]):
        anime_id = to_int(row.get("MAL_ID"))
        title = fix_text(row.get("Name"))
        if anime_id is None or not title or anime_id in archive_ids:
            continue
        archive_ids.add(anime_id)

        genres = _split_labels(row.get("Genres"))
        content_rating = fix_text(row.get("Rating"))
        adult_content = "hentai" in {genre.casefold() for genre in genres} or content_rating.casefold().startswith("rx")
        if adult_content and not include_adult:
            continue

        legacy = legacy_by_id.get(anime_id)
        studios = _split_labels(row.get("Studios"))
        producers = _split_labels(row.get("Producers"))
        licensors = _split_labels(row.get("Licensors"))
        media_type = clean_label(row.get("Type"))
        score_distribution = _score_distribution(row)
        archive_score = to_float(row.get("Score"))
        score = to_float(legacy.get("score")) if legacy else None
        score = score if score is not None else archive_score
        rank = to_int(legacy.get("rank")) if legacy else None
        rank = rank if rank is not None else to_int(row.get("Ranked"))
        popularity = to_int(legacy.get("popularity")) if legacy else None
        popularity = popularity if popularity is not None else to_int(row.get("Popularity"))
        members = to_int(legacy.get("members")) if legacy else None
        members = max(members or 0, to_int(row.get("Members")) or 0)
        start_year = _parse_year(row.get("Aired"), row.get("Premiered"))
        if legacy and legacy.get("start_year"):
            start_year = int(legacy["start_year"])

        aliases = dedupe(
            [
                title,
                fix_text(row.get("English name")),
                fix_text(row.get("Japanese name")),
                str(legacy.get("title") or "") if legacy else "",
            ]
        )
        metadata_tokens = dedupe(
            [
                *(metadata_token("genre", genre) for genre in genres),
                metadata_token("type", media_type),
                *(metadata_token("studio", studio) for studio in studios),
                metadata_token("source", clean_label(row.get("Source"))),
            ]
        )
        legacy_studio_relationships = _enrichment_relationships(legacy, "studio_relationships")
        legacy_producer_relationships = _enrichment_relationships(legacy, "producer_relationships")

        item: dict[str, Any] = {
            "id": anime_id,
            "title": title,
            "aliases": aliases,
            "score": score,
            "archive_score": archive_score,
            "rank": rank,
            "popularity": popularity,
            "members": members,
            "favorites": to_int(row.get("Favorites")) or 0,
            "synopsis": synopsis_by_id.get(anime_id, ""),
            "start_date": fix_text(row.get("Aired")).split(" to ", 1)[0],
            "end_date": (
                fix_text(row.get("Aired")).split(" to ", 1)[1] if " to " in fix_text(row.get("Aired")) else ""
            ),
            "start_year": start_year,
            "premiered": fix_text(row.get("Premiered")),
            "type": media_type,
            "episodes": to_int(row.get("Episodes")),
            "duration": fix_text(row.get("Duration")),
            "content_rating": content_rating,
            "source": clean_label(row.get("Source")),
            "image_url": str(legacy.get("image_url") or "") if legacy else "",
            "genres": genres,
            "genre_groups": (
                dict(legacy.get("genre_groups") or {}) if legacy and legacy.get("genre_groups") else {"genre": genres}
            ),
            "metadata_tokens": metadata_tokens,
            "studios": studios,
            "studio_relationships": (
                legacy_studio_relationships
                if legacy_studio_relationships
                else _anonymous_relationships(studios, "Studio")
            ),
            "producers": producers,
            "producer_relationships": (
                legacy_producer_relationships
                if legacy_producer_relationships
                else _anonymous_relationships(producers, "Producer")
            ),
            "licensors": licensors,
            "characters": _enrichment_relationships(legacy, "characters"),
            "character_relationships": _enrichment_relationships(legacy, "character_relationships"),
            "character_names": _enrichment_relationships(legacy, "character_names"),
            "staff": _enrichment_relationships(legacy, "staff"),
            "staff_relationships": _enrichment_relationships(legacy, "staff_relationships"),
            "creators": _enrichment_relationships(legacy, "creators"),
            "voice_actors": _enrichment_relationships(legacy, "voice_actors"),
            "voice_actor_roles": _enrichment_relationships(legacy, "voice_actor_roles"),
            "rating_count": sum(score_distribution.values()),
            "score_distribution": score_distribution,
            "watching_stats": {
                "watching": to_int(row.get("Watching")) or 0,
                "completed": to_int(row.get("Completed")) or 0,
                "on_hold": to_int(row.get("On-Hold")) or 0,
                "dropped": to_int(row.get("Dropped")) or 0,
                "plan_to_watch": to_int(row.get("Plan to Watch")) or 0,
            },
            "adult_content": adult_content,
            "collaborative_available": True,
            "relationship_enriched": any(_enrichment_relationships(legacy, field) for field in RELATIONSHIP_FIELDS),
            "data_source": "archive_2020+legacy_relationships" if legacy else "archive_2020",
        }
        catalog.append(item)

    if include_legacy_only:
        for anime_id, legacy in legacy_by_id.items():
            if anime_id in archive_ids:
                continue
            item = dict(legacy)
            item.setdefault("aliases", [str(item.get("title") or "")])
            item.setdefault("archive_score", None)
            item.setdefault("favorites", 0)
            item.setdefault("duration", "")
            item.setdefault("content_rating", "")
            item.setdefault("source", "")
            item.setdefault("licensors", [])
            item.setdefault("rating_count", 0)
            item.setdefault("score_distribution", {})
            item.setdefault("watching_stats", {})
            item["collaborative_available"] = False
            item["relationship_enriched"] = any(item.get(field) for field in RELATIONSHIP_FIELDS)
            item["data_source"] = "legacy_recent_extension"
            catalog.append(item)

    catalog.sort(
        key=lambda item: (
            item.get("rank") is None,
            item.get("rank") if item.get("rank") is not None else 10**9,
            -float(item.get("score") or 0),
            -int(item.get("members") or 0),
            int(item["id"]),
        )
    )
    return catalog
