from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from backend.anime_agent.collaborative import ARTIFACT_VERSION  # noqa: E402

DEFAULT_ARCHIVE = PROJECT_ROOT / "archive"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "collaborative_embeddings.npz"


def _user_projection(user_id: int, projections: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (1 << 64) - 1
    hashes: list[int] = []
    for projection in range(projections):
        value = (user_id * 0x9E3779B185EBCA87 + (projection + 1) * 0xC2B2AE3D27D4EB4F) & mask
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & mask
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        hashes.append(value)
    buckets: np.ndarray = np.asarray([value % width for value in hashes], dtype=np.int64)
    # The sign must use bits independent of the bucket bits. Reusing the low
    # bit would make every user in a bucket share a sign and create spurious
    # near-perfect similarities between popular titles.
    signs: np.ndarray = np.asarray(
        [1.0 if (value >> 32) & 1 == 0 else -1.0 for value in hashes],
        dtype=np.float32,
    )
    return buckets, signs


def build_collaborative_artifact(
    ratings_path: Path,
    catalog: list[dict[str, Any]],
    output_path: Path,
    *,
    projections: int = 3,
    width: int = 128,
    row_limit: int | None = None,
    progress_every: int = 5_000_000,
) -> dict[str, Any]:
    if projections < 1 or width < 8:
        raise ValueError("projections must be positive and width must be at least 8")
    catalog_ids: np.ndarray = np.asarray(
        sorted({int(item["id"]) for item in catalog}),
        dtype=np.int64,
    )
    if not len(catalog_ids):
        raise ValueError("Cannot train a collaborative model with an empty catalog")
    index_by_id = {int(anime_id): index for index, anime_id in enumerate(catalog_ids.tolist())}
    dimensions = projections * width
    vectors: np.ndarray = np.zeros((len(catalog_ids), dimensions), dtype=np.float32)
    counts: np.ndarray = np.zeros(len(catalog_ids), dtype=np.int64)
    sums: np.ndarray = np.zeros(len(catalog_ids), dtype=np.float64)
    offsets: np.ndarray = np.arange(projections, dtype=np.int64) * width

    rows_seen = 0
    ratings_used = 0
    users_seen = 0
    ignored_ratings = 0
    previous_user = -1
    current_user: int | None = None
    current_anime_ids: list[int] = []
    current_ratings: list[float] = []
    started = time.perf_counter()

    def flush_user() -> None:
        nonlocal ratings_used, users_seen, ignored_ratings
        if current_user is None or not current_ratings:
            return
        users_seen += 1
        all_ratings: np.ndarray = np.asarray(current_ratings, dtype=np.float32)
        kept_positions = [position for position, anime_id in enumerate(current_anime_ids) if anime_id in index_by_id]
        ignored_ratings += len(current_ratings) - len(kept_positions)
        if not kept_positions:
            return
        kept_anime_ids = [current_anime_ids[position] for position in kept_positions]
        if len(kept_anime_ids) != len(set(kept_anime_ids)):
            raise ValueError(f"Duplicate user-anime pair found for user {current_user}")
        indexes: np.ndarray = np.asarray(
            [index_by_id[anime_id] for anime_id in kept_anime_ids],
            dtype=np.int64,
        )
        kept_ratings = all_ratings[kept_positions]
        counts[indexes] += 1
        sums[indexes] += kept_ratings
        ratings_used += len(indexes)

        residuals = kept_ratings - float(np.mean(all_ratings))
        scale = max(float(np.std(all_ratings)), 1.0)
        residuals = residuals / scale
        if not np.any(np.abs(residuals) > 1e-7):
            return
        buckets, signs = _user_projection(current_user, projections, width)
        columns = offsets + buckets
        vectors[indexes[:, None], columns[None, :]] += residuals[:, None] * signs[None, :]

    with ratings_path.open("r", encoding="utf-8-sig", errors="strict", newline="") as file:
        header = file.readline().rstrip("\r\n").split(",")
        if header != ["user_id", "anime_id", "rating"]:
            raise ValueError(f"Unexpected rating_complete.csv header: {header}")
        for line in file:
            if row_limit is not None and rows_seen >= row_limit:
                break
            fields = line.rstrip("\r\n").split(",")
            if len(fields) != 3:
                raise ValueError(f"Malformed rating row at line {rows_seen + 2}")
            try:
                user_id, anime_id, rating = map(int, fields)
            except ValueError as exc:
                raise ValueError(f"Non-integer rating row at line {rows_seen + 2}") from exc
            if user_id < previous_user:
                raise ValueError("rating_complete.csv must be sorted by user_id")
            if rating < 1 or rating > 10:
                raise ValueError(f"Rating outside 1..10 at line {rows_seen + 2}")
            if current_user is not None and user_id != current_user:
                flush_user()
                current_anime_ids.clear()
                current_ratings.clear()
            current_user = user_id
            previous_user = user_id
            current_anime_ids.append(anime_id)
            current_ratings.append(float(rating))
            rows_seen += 1
            if progress_every and rows_seen % progress_every == 0:
                elapsed = max(time.perf_counter() - started, 0.001)
                print(
                    f"Processed {rows_seen:,} ratings ({rows_seen / elapsed:,.0f} rows/s, {users_seen:,} users)",
                    flush=True,
                )
    flush_user()

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 1e-8
    vectors[nonzero] /= norms[nonzero]
    rating_mean = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    ).astype(np.float32)
    global_mean = float(sums.sum() / max(counts.sum(), 1))
    prior_weight = 50.0
    bayesian_score = np.divide(
        sums + prior_weight * global_mean,
        counts + prior_weight,
        out=np.full_like(sums, global_mean),
        where=(counts + prior_weight) > 0,
    )
    bayesian_score = np.clip(bayesian_score / 10.0, 0.0, 1.0).astype(np.float32)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "method": "user-centred CountSketch item similarity",
        "ratings_path": str(ratings_path.resolve()),
        "rows_seen": rows_seen,
        "ratings_used": ratings_used,
        "ignored_ratings": ignored_ratings,
        "users_seen": users_seen,
        "catalog_items": len(catalog_ids),
        "items_with_ratings": int(np.count_nonzero(counts)),
        "projections": projections,
        "projection_width": width,
        "dimensions": dimensions,
        "global_rating_mean": round(global_mean, 6),
        "row_limit": row_limit,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        anime_ids=catalog_ids,
        vectors=vectors,
        rating_count=counts,
        rating_mean=rating_mean,
        bayesian_score=bayesian_score,
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
    )
    temporary_path.replace(output_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train compact collaborative item embeddings.")
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--projections", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--rating-limit", type=int, default=None)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    metadata = build_collaborative_artifact(
        args.archive_dir / "rating_complete.csv",
        catalog,
        args.output,
        projections=args.projections,
        width=args.width,
        row_limit=args.rating_limit,
    )
    print(json.dumps(metadata, indent=2))
    print(f"Wrote collaborative artifact to {args.output}")


if __name__ == "__main__":
    main()
