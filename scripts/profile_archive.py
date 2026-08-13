from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "archive"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "archive_quality.json"
MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "�")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _reader(path: Path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        yield from csv.DictReader(file)


def _unknown(value: object) -> bool:
    return not str(value or "").strip() or str(value).strip().casefold() in {"unknown", "nan", "none"}


def _contains_mojibake(value: object) -> bool:
    text = str(value or "")
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def _year(value: object) -> int | None:
    match = YEAR_RE.search(str(value or ""))
    return int(match.group()) if match else None


def profile_metadata(archive_dir: Path) -> tuple[dict[str, Any], set[int]]:
    anime_path = archive_dir / "anime.csv"
    synopsis_path = archive_dir / "anime_with_synopsis.csv"
    ids: set[int] = set()
    duplicate_ids = 0
    invalid_ids = 0
    missing: Counter[str] = Counter()
    media_types: Counter[str] = Counter()
    studios_missing = 0
    adult_titles = 0
    mojibake_rows = 0
    years: list[int] = []

    for row in _reader(anime_path):
        try:
            anime_id = int(row.get("MAL_ID") or "")
        except ValueError:
            invalid_ids += 1
            continue
        if anime_id in ids:
            duplicate_ids += 1
        ids.add(anime_id)
        for field in ("Name", "Score", "Genres", "Type", "Episodes", "Aired", "Studios"):
            if _unknown(row.get(field)):
                missing[field] += 1
        media_types[str(row.get("Type") or "Unknown").strip()] += 1
        if _unknown(row.get("Studios")):
            studios_missing += 1
        genres = str(row.get("Genres") or "").casefold()
        rating = str(row.get("Rating") or "").casefold()
        if "hentai" in genres or rating.startswith("rx"):
            adult_titles += 1
        if any(_contains_mojibake(value) for value in row.values()):
            mojibake_rows += 1
        parsed_year = _year(row.get("Aired")) or _year(row.get("Premiered"))
        if parsed_year:
            years.append(parsed_year)

    synopsis_ids: set[int] = set()
    synopsis_duplicate_ids = 0
    synopsis_mojibake_rows = 0
    synopsis_missing_text = 0
    for row in _reader(synopsis_path):
        try:
            anime_id = int(row.get("MAL_ID") or "")
        except ValueError:
            continue
        if anime_id in synopsis_ids:
            synopsis_duplicate_ids += 1
        synopsis_ids.add(anime_id)
        synopsis = row.get("sypnopsis")
        if _unknown(synopsis):
            synopsis_missing_text += 1
        if _contains_mojibake(synopsis):
            synopsis_mojibake_rows += 1

    row_count = len(ids)
    return (
        {
            "rows": row_count,
            "unique_anime_ids": len(ids),
            "duplicate_anime_ids": duplicate_ids,
            "invalid_anime_ids": invalid_ids,
            "missing": dict(sorted(missing.items())),
            "missing_rates": {
                field: round(count / row_count, 6) if row_count else 0.0 for field, count in sorted(missing.items())
            },
            "media_types": dict(media_types.most_common()),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "missing_studios": studios_missing,
            "adult_titles_excluded_by_default": adult_titles,
            "mojibake_rows_before_cleaning": mojibake_rows,
            "synopsis": {
                "rows": len(synopsis_ids),
                "duplicate_anime_ids": synopsis_duplicate_ids,
                "missing_text": synopsis_missing_text,
                "mojibake_rows_before_cleaning": synopsis_mojibake_rows,
                "catalog_join_coverage": round(len(ids.intersection(synopsis_ids)) / row_count, 6)
                if row_count
                else 0.0,
                "orphan_ids": len(synopsis_ids.difference(ids)),
            },
        },
        ids,
    )


def profile_ratings(
    archive_dir: Path,
    catalog_ids: set[int],
    *,
    row_limit: int | None = None,
) -> dict[str, Any]:
    path = archive_dir / "rating_complete.csv"
    rows = 0
    invalid_rows = 0
    orphan_rows = 0
    duplicate_pairs = 0
    sorted_by_user = True
    previous_user = -1
    users = 0
    current_user: int | None = None
    current_items: set[int] = set()
    rated_ids: set[int] = set()
    rating_counts: Counter[int] = Counter()
    rating_sum = 0
    rating_min = 11
    rating_max = 0

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        header = file.readline().strip().split(",")
        if header != ["user_id", "anime_id", "rating"]:
            raise ValueError(f"Unexpected rating_complete.csv header: {header}")
        for line in file:
            if row_limit is not None and rows >= row_limit:
                break
            try:
                user_text, anime_text, rating_text = line.rstrip("\r\n").split(",")
                user_id = int(user_text)
                anime_id = int(anime_text)
                rating = int(rating_text)
            except (ValueError, TypeError):
                invalid_rows += 1
                continue
            rows += 1
            if user_id < previous_user:
                sorted_by_user = False
            previous_user = user_id
            if current_user != user_id:
                users += 1
                current_user = user_id
                current_items.clear()
            if anime_id in current_items:
                duplicate_pairs += 1
            current_items.add(anime_id)
            if rating < 1 or rating > 10:
                invalid_rows += 1
            if anime_id not in catalog_ids:
                orphan_rows += 1
            rated_ids.add(anime_id)
            rating_counts[rating] += 1
            rating_sum += rating
            rating_min = min(rating_min, rating)
            rating_max = max(rating_max, rating)

    return {
        "rows_scanned": rows,
        "row_limit": row_limit,
        "users": users,
        "rated_anime_ids": len(rated_ids),
        "catalog_rating_coverage": round(len(rated_ids.intersection(catalog_ids)) / len(catalog_ids), 6)
        if catalog_ids
        else 0.0,
        "invalid_rows": invalid_rows,
        "invalid_rate": round(invalid_rows / rows, 8) if rows else 0.0,
        "orphan_rows": orphan_rows,
        "orphan_rate": round(orphan_rows / rows, 8) if rows else 0.0,
        "duplicate_user_anime_pairs": duplicate_pairs,
        "sorted_by_user": sorted_by_user,
        "rating_min": rating_min if rows else None,
        "rating_max": rating_max if rows else None,
        "rating_mean": round(rating_sum / rows, 6) if rows else None,
        "rating_distribution": {str(key): rating_counts[key] for key in sorted(rating_counts)},
    }


def build_report(
    archive_dir: Path,
    *,
    rating_limit: int | None = None,
) -> dict[str, Any]:
    metadata, catalog_ids = profile_metadata(archive_dir)
    ratings = profile_ratings(archive_dir, catalog_ids, row_limit=rating_limit)
    try:
        archive_display_path = archive_dir.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        archive_display_path = archive_dir.name
    gates = {
        "metadata_primary_key_unique": metadata["duplicate_anime_ids"] == 0,
        "metadata_primary_key_valid": metadata["invalid_anime_ids"] == 0,
        "synopsis_join_coverage_at_least_90_percent": metadata["synopsis"]["catalog_join_coverage"] >= 0.90,
        "ratings_valid": ratings["invalid_rows"] == 0,
        "ratings_orphan_rate_below_0_1_percent": ratings["orphan_rate"] < 0.001,
        "ratings_unique_user_anime_pairs": ratings["duplicate_user_anime_pairs"] == 0,
        "ratings_sorted_by_user": ratings["sorted_by_user"],
    }
    return {
        "dataset": {
            "name": "Anime Recommendation Database 2020",
            "source": "https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020",
            "license": "CC0-1.0",
            "archive_path": archive_display_path,
        },
        "grain": {
            "anime.csv": "one row per MAL anime ID",
            "rating_complete.csv": "one completed-and-rated anime per anonymous user",
        },
        "metadata": metadata,
        "ratings": ratings,
        "quality_gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the Anime Recommendation Database 2020 archive.")
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rating-limit", type=int, default=None)
    args = parser.parse_args()

    report = build_report(args.archive_dir, rating_limit=args.rating_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
