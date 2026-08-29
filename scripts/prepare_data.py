from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from backend.anime_agent.archive_pipeline import build_archive_catalog, find_archive_dir  # noqa: E402
from backend.anime_agent.data_pipeline import build_catalog, find_raw_dir, write_catalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Anime Compass catalog artifacts.")
    parser.add_argument("--dataset", choices=("archive", "legacy"), default="archive")
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--enrichment-catalog", type=Path, default=None)
    parser.add_argument("--include-adult", action="store_true")
    parser.add_argument("--drop-legacy-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "anime_catalog.json",
    )
    args = parser.parse_args()

    if args.dataset == "legacy":
        raw_dir = find_raw_dir(PROJECT_ROOT)
        catalog = build_catalog(raw_dir)
        source = raw_dir
    else:
        archive_dir = args.archive_dir or find_archive_dir(PROJECT_ROOT)
        enrichment_path = args.enrichment_catalog
        if enrichment_path is None and args.output.exists():
            enrichment_path = args.output
        enrichment_catalog = None
        if enrichment_path and enrichment_path.exists():
            enrichment_catalog = json.loads(enrichment_path.read_text(encoding="utf-8"))
        catalog = build_archive_catalog(
            archive_dir,
            enrichment_catalog=enrichment_catalog,
            include_adult=args.include_adult,
            include_legacy_only=not args.drop_legacy_only,
        )
        source = archive_dir

    ids = [int(item["id"]) for item in catalog]
    if not catalog or len(ids) != len(set(ids)):
        raise ValueError("Catalog quality gate failed: records are empty or MAL IDs are duplicated")
    if any(not str(item.get("title") or "").strip() for item in catalog):
        raise ValueError("Catalog quality gate failed: one or more titles are empty")

    output_path = write_catalog(catalog, args.output)
    print(f"Wrote {len(catalog)} anime records from {source} to {output_path}")


if __name__ == "__main__":
    main()
