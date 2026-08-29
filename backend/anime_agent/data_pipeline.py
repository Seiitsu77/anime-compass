from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
PROCESSED_ARTIFACTS = PROJECT_ROOT / "data" / "processed" / "recommendation_artifacts.json"

CSV_NAMES = {
    "anime": "anime.csv",
    "characters": "anime_characters.csv",
    "genres": "anime_genres.csv",
    "companies": "anime_companies.csv",
    "entities": "entities.csv",
    "staff": "anime_staff.csv",
    "voice_actors": "anime_voice_actors.csv",
}

MOJIBAKE_MARKERS = ("\u00c3", "\u00c2", "\u00e2", "\u00f0", "\ufffd")
VALID_CHARACTER_ROLES = {"main": "Main", "supporting": "Supporting", "unknown": "Unknown", "": "Unknown"}
STAFF_LIKE_CHARACTER_ROLES = {
    "director",
    "producer",
    "key animation",
    "music",
    "script",
    "series composition",
    "character design",
    "original creator",
}
PRIORITY_STAFF_ROLES = {
    "director": "Director",
    "original creator": "Original Creator",
    "creator": "Original Creator",
    "series composition": "Series Composition",
    "script": "Script",
    "music": "Music",
    "character design": "Character Design",
}
PLACEHOLDER_ENTITY_NAMES = {"none found", "add some"}


def fix_text(value: Any) -> str:
    """Clean whitespace and common UTF-8-as-Latin-1 mojibake found in CSV exports."""
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    # Some MAL exports were decoded as either Latin-1 or Windows-1252, and a
    # small subset was decoded twice. Keep a candidate only when it reduces the
    # number of characteristic mojibake markers.
    for _ in range(2):
        marker_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
        if not marker_count:
            break
        candidates = [text]
        for encoding in ("latin1", "cp1252"):
            try:
                candidates.append(text.encode(encoding).decode("utf-8"))
            except UnicodeError:
                continue
        repaired = min(
            candidates,
            key=lambda candidate: (
                sum(candidate.count(marker) for marker in MOJIBAKE_MARKERS),
                candidate.count("\ufffd"),
            ),
        )
        if repaired == text:
            break
        text = repaired

    return re.sub(r"\s+", " ", text).strip()


def clean_label(value: Any) -> str:
    """Normalize labels such as 'Theme::School School'."""
    text = fix_text(value)
    if "::" in text:
        text = text.rsplit("::", 1)[-1].strip()

    parts = text.split()
    half = len(parts) // 2

    if parts and len(parts) % 2 == 0 and parts[:half] == parts[half:]:
        text = " ".join(parts[:half])

    return text


def slug_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", clean_label(value).casefold()).strip("_")


def split_category_label(value: Any) -> tuple[str, str]:
    text = fix_text(value)
    if "::" not in text:
        return "genre", clean_label(text)

    prefix, label = text.split("::", 1)
    category = slug_label(prefix) or "genre"
    return category, clean_label(label)


def metadata_token(category: str, label: str) -> str:
    token = slug_label(label)
    return f"{category}_{token}" if category and token else ""


def normalize_staff_role(value: Any) -> str:
    role = clean_label(value)
    role_key = role.casefold()
    return PRIORITY_STAFF_ROLES.get(role_key, role)


def selected_staff_role(value: Any) -> str | None:
    role_parts = set()
    for part in re.split(r"[,/;|]+", fix_text(value)):
        role = clean_label(part).casefold()
        if role:
            role_parts.add(role)
    for role_key, normalized_role in PRIORITY_STAFF_ROLES.items():
        if role_key in role_parts:
            return normalized_role
    return None


def normalize_character_role(value: Any) -> str | None:
    role = clean_label(value)
    role_key = role.casefold()
    if role_key in STAFF_LIKE_CHARACTER_ROLES:
        return None
    return VALID_CHARACTER_ROLES.get(role_key, "Unknown")


def to_int(value: Any) -> int | None:
    text = fix_text(value)
    if not text or text.lower() in {"unknown", "nan", "none"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def to_float(value: Any) -> float | None:
    text = fix_text(value)
    if not text or text.lower() in {"unknown", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def year_from_date(value: Any) -> int | None:
    text = fix_text(value)
    if len(text) < 4:
        return None
    try:
        return int(text[:4])
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        return list(csv.DictReader(file))


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        cleaned = clean_label(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)

    return result


def entity_card(entity_id: int, entities: dict[int, dict[str, str]]) -> dict[str, Any] | None:
    entity = entities.get(entity_id)
    if not entity or not entity["name"]:
        return None

    return {
        "id": entity_id,
        "name": entity["name"],
        "image_url": entity["image_url"],
    }


def dedupe_people(values: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seen: set[int] = set()
    result: list[dict[str, Any]] = []

    for value in values:
        entity_id = int(value["id"])
        if entity_id in seen:
            continue
        seen.add(entity_id)
        result.append(value)
        if len(result) >= limit:
            break

    return result


def dedupe_voice_actor_roles(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    result: list[dict[str, Any]] = []

    for value in values:
        key = (
            int(value["voice_actor_id"]),
            int(value["character_id"]),
            str(value.get("language") or "Unknown").casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(value)

    return result


def character_priority(role: str) -> int:
    role_key = role.casefold()
    if role_key == "main":
        return 0
    if role_key == "supporting":
        return 1
    return 2


def voice_priority(language: str) -> int:
    language_key = language.casefold()
    if language_key == "japanese":
        return 0
    if language_key == "english":
        return 1
    return 2


def find_raw_dir(project_root: Path = PROJECT_ROOT) -> Path:
    candidates = [
        project_root / "Anime Data" / "Anime Data",
        project_root / "Anime Data",
        project_root / "data" / "raw",
    ]

    for candidate in candidates:
        if (candidate / CSV_NAMES["anime"]).exists():
            return candidate

    for anime_csv in project_root.rglob(CSV_NAMES["anime"]):
        candidate = anime_csv.parent
        if all((candidate / name).exists() for name in CSV_NAMES.values()):
            return candidate

    expected = ", ".join(CSV_NAMES.values())
    raise FileNotFoundError(f"Could not find the raw anime CSV files under {project_root}. Expected files: {expected}.")


def build_catalog(raw_dir: Path) -> list[dict[str, Any]]:
    anime_rows = read_csv(raw_dir / CSV_NAMES["anime"])
    character_rows = read_csv(raw_dir / CSV_NAMES["characters"])
    genre_rows = read_csv(raw_dir / CSV_NAMES["genres"])
    company_rows = read_csv(raw_dir / CSV_NAMES["companies"])
    entity_rows = read_csv(raw_dir / CSV_NAMES["entities"])
    staff_rows = read_csv(raw_dir / CSV_NAMES["staff"])
    voice_actor_rows = read_csv(raw_dir / CSV_NAMES["voice_actors"])

    genres_by_anime: dict[int, list[str]] = defaultdict(list)
    genre_groups_by_anime: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    metadata_tokens_by_anime: dict[int, list[str]] = defaultdict(list)
    for row in genre_rows:
        anime_id = to_int(row.get("anime_id"))
        category, label = split_category_label(row.get("genre"))
        if anime_id is not None and label:
            genres_by_anime[anime_id].append(label)
            genre_groups_by_anime[anime_id][category].append(label)
            token = metadata_token(category, label)
            if token:
                metadata_tokens_by_anime[anime_id].append(token)

    entities: dict[int, dict[str, str]] = {}
    for row in entity_rows:
        entity_id = to_int(row.get("entity_id"))
        if entity_id is None:
            continue

        name = clean_label(row.get("name"))
        if name.casefold() in PLACEHOLDER_ENTITY_NAMES:
            continue

        entities[entity_id] = {
            "type": clean_label(row.get("entity_type")).lower(),
            "name": name,
            "image_url": fix_text(row.get("image_url")),
        }

    studios_by_anime: dict[int, list[str]] = defaultdict(list)
    producers_by_anime: dict[int, list[str]] = defaultdict(list)
    studio_relationships_by_anime: dict[int, list[dict[str, Any]]] = defaultdict(list)
    producer_relationships_by_anime: dict[int, list[dict[str, Any]]] = defaultdict(list)
    voice_actors_by_character: dict[int, list[dict[str, Any]]] = defaultdict(list)
    characters_by_anime: dict[int, list[dict[str, Any]]] = defaultdict(list)
    staff_by_anime: dict[int, list[dict[str, Any]]] = defaultdict(list)
    creators_by_anime: dict[int, list[dict[str, Any]]] = defaultdict(list)
    voice_actors_by_anime: dict[int, list[dict[str, Any]]] = defaultdict(list)
    voice_actor_roles_by_anime: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in company_rows:
        anime_id = to_int(row.get("anime_id"))
        company_id = to_int(row.get("company_id"))
        role = clean_label(row.get("role")).lower()

        if anime_id is None or company_id is None:
            continue

        entity = entities.get(company_id)
        if not entity or not entity["name"]:
            continue

        if role == "studio" or entity["type"] == "studio":
            studios_by_anime[anime_id].append(entity["name"])
            studio_relationships_by_anime[anime_id].append({"id": company_id, "name": entity["name"], "role": "Studio"})
        else:
            producers_by_anime[anime_id].append(entity["name"])
            producer_relationships_by_anime[anime_id].append(
                {"id": company_id, "name": entity["name"], "role": clean_label(row.get("role")) or "Producer"}
            )

    for row in voice_actor_rows:
        character_id = to_int(row.get("character_id"))
        person_id = to_int(row.get("person_id"))
        language = clean_label(row.get("language")) or "Unknown"

        if character_id is None or person_id is None:
            continue

        card = entity_card(person_id, entities)
        if not card:
            continue

        card["language"] = language
        voice_actors_by_character[character_id].append(card)

    for values in voice_actors_by_character.values():
        values.sort(key=lambda item: (voice_priority(item.get("language", "")), item["name"]))

    for row in character_rows:
        anime_id = to_int(row.get("anime_id"))
        character_id = to_int(row.get("character_id"))
        character_role = normalize_character_role(row.get("role"))

        if anime_id is None or character_id is None or character_role is None:
            continue

        card = entity_card(character_id, entities)
        if not card:
            continue

        card["role"] = role
        all_voice_actors = dedupe_people(
            voice_actors_by_character.get(character_id, []),
            limit=max(1, len(voice_actors_by_character.get(character_id, []))),
        )
        card["voice_actors"] = all_voice_actors[:3]
        characters_by_anime[anime_id].append(card)
        voice_actors_by_anime[anime_id].extend(all_voice_actors)
        for actor in all_voice_actors:
            voice_actor_roles_by_anime[anime_id].append(
                {
                    "voice_actor_id": actor["id"],
                    "voice_actor": actor["name"],
                    "character_id": card["id"],
                    "character": card["name"],
                    "language": actor.get("language") or "Unknown",
                }
            )

    for values in characters_by_anime.values():
        values.sort(key=lambda item: (character_priority(item.get("role", "")), item["name"]))

    for row in staff_rows:
        anime_id = to_int(row.get("anime_id"))
        person_id = to_int(row.get("person_id"))
        role = normalize_staff_role(row.get("role")) or "Unknown"

        if anime_id is None or person_id is None:
            continue

        card = entity_card(person_id, entities)
        if not card:
            continue

        card["role"] = role
        staff_by_anime[anime_id].append(card)
        if selected_staff_role(role):
            creators_by_anime[anime_id].append(card)

    catalog: list[dict[str, Any]] = []
    seen_anime_ids: set[int] = set()

    for row in anime_rows:
        anime_id = to_int(row.get("anime_id"))
        title = fix_text(row.get("title"))

        if anime_id is None or not title or anime_id in seen_anime_ids:
            continue
        seen_anime_ids.add(anime_id)

        media_type = clean_label(row.get("type"))
        type_token = metadata_token("type", media_type)
        tokens = dedupe(metadata_tokens_by_anime.get(anime_id, []))
        if type_token:
            tokens.append(type_token)

        catalog.append(
            {
                "id": anime_id,
                "title": title,
                "score": to_float(row.get("score")),
                "rank": to_int(row.get("rank")),
                "popularity": to_int(row.get("popularity")),
                "members": to_int(row.get("members")) or 0,
                "synopsis": fix_text(row.get("synopsis")),
                "start_date": fix_text(row.get("start_date")),
                "end_date": fix_text(row.get("end_date")),
                "start_year": year_from_date(row.get("start_date")),
                "type": media_type,
                "episodes": to_int(row.get("episodes")),
                "image_url": fix_text(row.get("image_url")),
                "genres": dedupe(genres_by_anime.get(anime_id, [])),
                "genre_groups": {
                    category: dedupe(values) for category, values in genre_groups_by_anime.get(anime_id, {}).items()
                },
                "metadata_tokens": dedupe(tokens),
                "studios": dedupe(studios_by_anime.get(anime_id, [])),
                "studio_relationships": dedupe_people(
                    studio_relationships_by_anime.get(anime_id, []),
                    limit=max(1, len(studio_relationships_by_anime.get(anime_id, []))),
                ),
                "producers": dedupe(producers_by_anime.get(anime_id, []))[:6],
                "producer_relationships": dedupe_people(
                    producer_relationships_by_anime.get(anime_id, []),
                    limit=max(1, len(producer_relationships_by_anime.get(anime_id, []))),
                ),
                "characters": characters_by_anime.get(anime_id, [])[:12],
                "character_relationships": dedupe_people(
                    characters_by_anime.get(anime_id, []),
                    limit=max(1, len(characters_by_anime.get(anime_id, []))),
                ),
                "character_names": dedupe(
                    [person["name"] for person in characters_by_anime.get(anime_id, []) if person.get("name")]
                ),
                "staff": dedupe_people(staff_by_anime.get(anime_id, []), limit=12),
                "staff_relationships": dedupe_people(
                    staff_by_anime.get(anime_id, []),
                    limit=max(1, len(staff_by_anime.get(anime_id, []))),
                ),
                "creators": dedupe_people(creators_by_anime.get(anime_id, []), limit=8),
                "voice_actors": dedupe_people(voice_actors_by_anime.get(anime_id, []), limit=12),
                "voice_actor_roles": dedupe_voice_actor_roles(voice_actor_roles_by_anime.get(anime_id, [])),
            }
        )

    catalog.sort(
        key=lambda item: (
            item["rank"] is None,
            item["rank"] if item["rank"] is not None else 10**9,
            -float(item["score"] or 0),
            -int(item["members"] or 0),
        )
    )
    return catalog


def build_artifact_cache(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "anime_id_to_row": {str(item["id"]): index for index, item in enumerate(catalog)},
        "title_by_id": {str(item["id"]): item["title"] for item in catalog},
        "metadata_tokens_by_id": {str(item["id"]): item.get("metadata_tokens", []) for item in catalog},
        "studios_by_id": {str(item["id"]): item.get("studios", []) for item in catalog},
        "creators_by_id": {
            str(item["id"]): [
                {
                    "id": person.get("id"),
                    "name": person.get("name"),
                    "role": person.get("role"),
                }
                for person in item.get("creators", [])
            ]
            for item in catalog
        },
        "notes": (
            "Dense vectors, TF-IDF maps, and LSA components are built once in memory "
            "when the app starts to avoid binary cache files and extra dependencies."
        ),
    }


def write_artifact_cache(catalog: list[dict[str, Any]], output_path: Path = PROCESSED_ARTIFACTS) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(build_artifact_cache(catalog), file, ensure_ascii=False, separators=(",", ":"))
    return output_path


def write_catalog(catalog: list[dict[str, Any]], output_path: Path = PROCESSED_CATALOG) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(catalog, file, ensure_ascii=False, separators=(",", ":"))
    write_artifact_cache(catalog, output_path.parent / PROCESSED_ARTIFACTS.name)
    return output_path


def load_or_create_catalog(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    output_path = project_root / "data" / "processed" / "anime_catalog.json"

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    from .archive_pipeline import build_archive_catalog, find_archive_dir

    try:
        archive_dir = find_archive_dir(project_root)
    except FileNotFoundError:
        archive_dir = None
    if archive_dir is not None:
        catalog = build_archive_catalog(archive_dir)
        write_catalog(catalog, output_path)
        return catalog

    raw_dir = find_raw_dir(project_root)
    catalog = build_catalog(raw_dir)
    write_catalog(catalog, output_path)
    return catalog


def main() -> None:
    raw_dir = find_raw_dir(PROJECT_ROOT)
    catalog = build_catalog(raw_dir)
    output_path = write_catalog(catalog, PROCESSED_CATALOG)
    print(f"Wrote {len(catalog)} anime records to {output_path}")


if __name__ == "__main__":
    main()
