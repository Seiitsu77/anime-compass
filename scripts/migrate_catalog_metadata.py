"""Merge a newer MAL metadata export into the serving catalog.

**No such dataset is present in this repository.** This script is the interface
for one, not a downloader: it will not fetch anything from the internet, and it
refuses to run without an explicit local path.

What it is for
--------------
The catalog powering search, filters, and display is derived from the CC0 Anime
Recommendation Database 2020. It currently reaches 2025 through retained
enrichment, but its poster coverage is 55% and it has no English titles. A
newer export improves *display and discovery*.

What it is NOT for
------------------
Retraining ALS. The two datasets play different roles and must not be merged
conceptually:

* **historical user-item interactions** -> ALS personalization
* **newer metadata**                     -> catalog freshness, search, filters, display

A title that appears only in the new metadata has no trained ALS factor. It is
marked ``cold_start`` and stays searchable and displayable, but it is never
given a fabricated personalized score.

Expected input
--------------
CSV or JSON keyed by ``mal_id``, with any of::

    mal_id, title, title_english, title_japanese, type, source, episodes,
    status, score, scored_by, rank, popularity, members, favorites, season,
    year, studios, genres, themes, demographics, synopsis, rating, duration

``mal_id`` is the canonical join key; the existing catalog's ``id`` is the same
MAL identifier.

    python scripts/migrate_catalog_metadata.py --source path/to/anime_2026.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "anime_catalog_refreshed.json"

# Fields the newer export may refresh. Interaction-derived fields are absent on
# purpose: overwriting them would silently change what the model was measured
# against.
REFRESHABLE = (
    "title_english",
    "title_japanese",
    "image_url",
    "synopsis",
    "type",
    "source",
    "episodes",
    "status",
    "duration",
    "season",
    "content_rating",
)
DISPLAY_ONLY_SCORES = ("score", "scored_by", "rank", "popularity", "members", "favorites")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Local CSV or JSON metadata export.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--refresh-scores",
        action="store_true",
        help="Also refresh score/members/rank. Off by default: these feed the quality prior.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the join only; write nothing.")
    return parser.parse_args()


def read_source(path: Path) -> Iterator[Mapping[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"Metadata source not found: {path}\n"
            "This script never downloads anything. Provide a local export keyed by mal_id."
        )
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("data") or []
        yield from rows
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def coerce_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    by_id = {int(item["id"]): dict(item) for item in catalog}
    old_size = len(by_id)

    incoming: dict[int, Mapping[str, Any]] = {}
    malformed = 0
    for row in read_source(args.source):
        mal_id = coerce_int(row.get("mal_id") or row.get("id"))
        if mal_id is None:
            malformed += 1
            continue
        incoming[mal_id] = row

    matched = sorted(set(by_id) & set(incoming))
    new_only = sorted(set(incoming) - set(by_id))
    missing_from_source = sorted(set(by_id) - set(incoming))

    refreshed_fields = 0
    for mal_id in matched:
        row = incoming[mal_id]
        target = by_id[mal_id]
        fields = REFRESHABLE + (DISPLAY_ONLY_SCORES if args.refresh_scores else ())
        for field in fields:
            value = row.get(field)
            if value not in (None, "", []) and target.get(field) != value:
                target[field] = value
                refreshed_fields += 1

    added = 0
    recent_added = 0
    for mal_id in new_only:
        row = incoming[mal_id]
        year = coerce_int(row.get("year")) or coerce_int(
            str(row.get("season", "")).split()[-1] if row.get("season") else None
        )
        entry: dict[str, Any] = {
            "id": mal_id,
            "title": row.get("title") or f"#{mal_id}",
            "start_year": year,
            # No trained ALS factor exists for these. They are searchable and
            # displayable; they must never receive a fabricated model score.
            "cold_start": True,
            "collaborative_available": False,
            "data_source": "metadata_refresh",
        }
        for field in REFRESHABLE + DISPLAY_ONLY_SCORES:
            if row.get(field) not in (None, "", []):
                entry[field] = row[field]
        for field in ("genres", "themes", "demographics", "studios"):
            raw = row.get(field)
            if isinstance(raw, str) and raw:
                entry[field] = [part.strip() for part in raw.split(",") if part.strip()]
            elif isinstance(raw, list):
                entry[field] = raw
        by_id[mal_id] = entry
        added += 1
        if year and year >= 2021:
            recent_added += 1

    existing_recent = sum(1 for item in catalog if (item.get("start_year") or 0) >= 2021)
    print("=== catalog metadata migration ===")
    print(f"  source                : {args.source}")
    print(f"  old catalog size      : {old_size:,}")
    print(f"  source rows           : {len(incoming):,} ({malformed} skipped without a usable mal_id)")
    print(f"  MAL-ID overlap        : {len(matched):,}")
    print(f"  in catalog only       : {len(missing_from_source):,} (retained)")
    print(f"  new titles added      : {added:,}")
    print(f"  2021+ titles added    : {recent_added:,} (catalog already had {existing_recent:,})")
    print(f"  fields refreshed      : {refreshed_fields:,}")
    print(f"  new catalog size      : {len(by_id):,}")
    print(f"  ALS-covered items     : {old_size:,} (unchanged; ALS is not retrained)")
    print(f"  cold-start items      : {added:,}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    merged = [by_id[key] for key in sorted(by_id)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output)
    print(f"\nwrote {args.output}")
    print(
        "\nThe ALS artifact was trained against the OLD catalog ID set. Serving this\n"
        "refreshed catalog will change the catalog digest, so either re-pin\n"
        "ALS_EXPECTED_CATALOG_IDS_SHA256 or leave it unset. Cold-start items are\n"
        "flagged and must stay out of personalized ranking."
    )


if __name__ == "__main__":
    main()
