"""Build the compact serving catalog used by the portfolio demo.

The full catalog is 119 MB, and almost all of it is entity data the demo never
touches: character lists, voice-actor roles, staff credits, and producer
relationships together account for ~90 MB. Those exist for the constraint-rich
Hybrid path, which resolves entities and performs exact catalog joins. The fast
ALS path does not read any of them.

So the deployment payload is dominated by fields the deployed page cannot use.
This script writes a catalog restricted to what the demo actually reads,
verifies that recommendations are unchanged, and reports the saving.

What this is not: a model change. No score, factor, ranking, or benchmark is
touched. The ALS artifact joins on `id`, which is preserved exactly, and the
catalog identity digest is computed over the ID set, which is also preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.anime_agent.als_serving import catalog_ids_digest  # noqa: E402
from backend.anime_agent.showcase import truncate_synopsis  # noqa: E402

# Every field the demo reads, and where. Anything not on this list is unread by
# streamlit_app.py, ShowcaseService, and the fast path, and is therefore dropped.
SERVING_FIELDS: dict[str, str] = {
    "id": "join key to the ALS artifact; MAL id, preserved exactly",
    "title": "search, display, series collapsing",
    "genres": "genre pills, shared-genre explanations, diversity rerank",
    "start_year": "result and search-row metadata",
    "type": "result and search-row metadata",
    "episodes": "result metadata",
    "score": "result metadata",
    "members": "search ranking and popularity ordering",
    "image_url": "poster",
    "synopsis": "result body, pre-trimmed to display length",
}

# `id` and `title` always ship. Everything else is omitted when empty, because
# every reader already goes through `.get(...)` with a default, so an absent key
# and an empty value render identically.
ALWAYS_PRESENT = ("id", "title")

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_TARGET = PROJECT_ROOT / "data" / "processed" / "anime_catalog_serving.json"


def compact_entry(item: Mapping[str, Any]) -> dict[str, Any]:
    """Project one catalog row onto the serving fields."""
    compact: dict[str, Any] = {"id": int(item["id"]), "title": str(item.get("title") or "")}
    for field in SERVING_FIELDS:
        if field in ALWAYS_PRESENT:
            continue
        value = truncate_synopsis(item.get(field)) if field == "synopsis" else item.get(field)
        if value in (None, "", [], {}):
            continue
        compact[field] = value
    return compact


def build_serving_catalog(catalog: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact a catalog, preserving order and every ID exactly."""
    return [compact_entry(item) for item in catalog]


def check_identity(full: Sequence[Mapping[str, Any]], compact: Sequence[Mapping[str, Any]]) -> list[str]:
    """Assert the properties serving depends on. Returns failures, empty if sound."""
    failures: list[str] = []
    full_ids = [int(item["id"]) for item in full]
    compact_ids = [int(item["id"]) for item in compact]
    if full_ids != compact_ids:
        failures.append("ID sequence changed")
    if catalog_ids_digest(sorted(set(full_ids))) != catalog_ids_digest(sorted(set(compact_ids))):
        failures.append("catalog identity digest changed")
    for original, small in zip(full, compact, strict=True):
        if str(original.get("title") or "") != str(small.get("title") or ""):
            failures.append(f"title changed for id {original['id']}")
            break
        if list(original.get("genres") or []) != list(small.get("genres") or []):
            failures.append(f"genres changed for id {original['id']}")
            break
    extra = {key for item in compact for key in item} - set(SERVING_FIELDS)
    if extra:
        failures.append(f"unexpected fields written: {sorted(extra)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--indent", type=int, default=0, help="0 writes the compact single-line form")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source catalog not found: {args.source}")

    started = time.perf_counter()
    full = json.loads(args.source.read_text(encoding="utf-8"))
    full_parse = time.perf_counter() - started

    compact = build_serving_catalog(full)
    failures = check_identity(full, compact)
    if failures:
        raise SystemExit("Refusing to write a catalog that changes serving behaviour:\n  " + "\n  ".join(failures))

    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(
        json.dumps(compact, ensure_ascii=False, separators=(",", ":"), indent=args.indent or None),
        encoding="utf-8",
    )

    started = time.perf_counter()
    reloaded = json.loads(args.target.read_text(encoding="utf-8"))
    compact_parse = time.perf_counter() - started

    source_mb = args.source.stat().st_size / 1e6
    target_mb = args.target.stat().st_size / 1e6
    print(f"source     {args.source}  {source_mb:8.2f} MB  {len(full):,} items  parse {full_parse:.2f}s")
    print(f"serving    {args.target}  {target_mb:8.2f} MB  {len(reloaded):,} items  parse {compact_parse:.2f}s")
    print(
        f"reduction  {100 * (1 - target_mb / source_mb):.1f}% smaller, {full_parse / compact_parse:.1f}x faster to parse"
    )
    print(f"fields     {', '.join(SERVING_FIELDS)}")
    print(f"digest     {catalog_ids_digest(sorted({int(item['id']) for item in reloaded}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
