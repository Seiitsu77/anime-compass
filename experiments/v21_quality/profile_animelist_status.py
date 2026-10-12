"""What do MAL list status and progress add beyond completed-and-rated titles?

The production models train on `rating_complete.csv`: titles a user finished
*and* scored. `animelist.csv` is the superset it was cut from, and it also
carries the watching status and the number of episodes watched. This profiles
that extra signal. It reads raw interactions only; no model is trained or
scored, and no evaluation user is spent.

    python experiments/v21_quality/profile_animelist_status.py

Streams the 2 GB file in chunks; expect a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARCHIVE = PROJECT_ROOT / "archive" / "animelist.csv"
ALS_ARTIFACT = (
    PROJECT_ROOT / "data" / "evaluation" / "personalized" / "artifacts" / "holdout_seed42_pos8" / "als_train_only.npz"
)
OUTPUT = Path(__file__).resolve().parent / "results"
STATUS = {1: "currently_watching", 2: "completed", 3: "on_hold", 4: "dropped", 6: "plan_to_watch"}
RATING_COMPLETE_ROWS = 57_633_278  # data/evaluation/archive_quality.json
EPISODE_EDGES = (0, 1, 3, 6, 13, 26)
EPISODE_LABELS = ("0", "1-2", "3-5", "6-12", "13-25", "26+")
STATUS_SLOTS = 8
RATING_SLOTS = 11


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=8_000_000)
    args = parser.parse_args()
    started = time.perf_counter()

    with np.load(ALS_ARTIFACT, allow_pickle=False) as payload:
        catalog_ids = np.asarray(payload["anime_ids"], dtype=np.int64)
    in_catalog = np.zeros(int(catalog_ids.max()) + 1, dtype=bool)
    in_catalog[catalog_ids] = True

    cross = np.zeros((STATUS_SLOTS, RATING_SLOTS), dtype=np.int64)
    episodes = np.zeros((STATUS_SLOTS, len(EPISODE_LABELS)), dtype=np.int64)
    dropped_like = np.zeros((len(EPISODE_LABELS), 2), dtype=np.int64)  # [rated, rated >= 8]
    user_rows = np.zeros(400_000, dtype=np.int64)
    user_plan = np.zeros(400_000, dtype=np.int64)
    user_rated_completed = np.zeros(400_000, dtype=np.int64)
    user_other_engaged = np.zeros(400_000, dtype=np.int64)
    rows = 0
    catalog_rows = 0
    invalid = 0

    reader = pd.read_csv(
        args.archive,
        chunksize=args.chunk_rows,
        dtype={
            "user_id": "int32",
            "anime_id": "int32",
            "rating": "int16",
            "watching_status": "int16",
            "watched_episodes": "int32",
        },
    )
    for number, chunk in enumerate(reader, start=1):
        user = chunk["user_id"].to_numpy()
        anime = chunk["anime_id"].to_numpy()
        rating = chunk["rating"].to_numpy()
        status = chunk["watching_status"].to_numpy()
        watched = chunk["watched_episodes"].to_numpy()
        rows += len(chunk)

        valid = (status >= 0) & (status < STATUS_SLOTS) & (rating >= 0) & (rating < RATING_SLOTS)
        invalid += int((~valid).sum())
        user, anime, rating, status, watched = user[valid], anime[valid], rating[valid], status[valid], watched[valid]
        catalog_rows += int(in_catalog[np.clip(anime, 0, len(in_catalog) - 1)][anime < len(in_catalog)].sum())

        cross += np.bincount(status * RATING_SLOTS + rating, minlength=STATUS_SLOTS * RATING_SLOTS).reshape(
            STATUS_SLOTS, RATING_SLOTS
        )
        band = np.digitize(np.clip(watched, 0, None), EPISODE_EDGES) - 1
        episodes += np.bincount(
            status * len(EPISODE_LABELS) + band, minlength=STATUS_SLOTS * len(EPISODE_LABELS)
        ).reshape(STATUS_SLOTS, len(EPISODE_LABELS))
        dropped_rated = (status == 4) & (rating > 0)
        dropped_like[:, 0] += np.bincount(band[dropped_rated], minlength=len(EPISODE_LABELS))
        dropped_like[:, 1] += np.bincount(band[dropped_rated & (rating >= 8)], minlength=len(EPISODE_LABELS))

        top_user = int(user.max()) + 1
        if top_user > len(user_rows):
            grow = top_user - len(user_rows) + 50_000
            user_rows = np.concatenate([user_rows, np.zeros(grow, dtype=np.int64)])
            user_plan = np.concatenate([user_plan, np.zeros(grow, dtype=np.int64)])
            user_rated_completed = np.concatenate([user_rated_completed, np.zeros(grow, dtype=np.int64)])
            user_other_engaged = np.concatenate([user_other_engaged, np.zeros(grow, dtype=np.int64)])
        user_rows += np.bincount(user, minlength=len(user_rows))
        user_plan += np.bincount(user[status == 6], minlength=len(user_rows))
        rated_completed = (status == 2) & (rating > 0)
        user_rated_completed += np.bincount(user[rated_completed], minlength=len(user_rows))
        # Engagement the current training file cannot see: started or finished, but not finished-and-scored.
        other = ((status == 1) | (status == 2) | (status == 3)) & ~rated_completed
        user_other_engaged += np.bincount(user[other], minlength=len(user_rows))
        print(f"  chunk {number}: {rows:,} rows ({time.perf_counter() - started:.0f}s)")

    total = int(cross.sum())
    by_status: dict[str, Any] = {}
    for code, name in STATUS.items():
        counts = cross[code]
        rated = int(counts[1:].sum())
        n = int(counts.sum())
        by_status[name] = {
            "rows": n,
            "share_of_rows": n / total,
            "rated": rated,
            "unrated": int(counts[0]),
            "share_rated": rated / n if n else None,
            "mean_rating_when_rated": float((counts[1:] * np.arange(1, 11)).sum() / rated) if rated else None,
            "share_8_or_higher_when_rated": float(counts[8:].sum() / rated) if rated else None,
            "share_5_or_lower_when_rated": float(counts[1:6].sum() / rated) if rated else None,
            "episodes_watched": {
                label: int(value) for label, value in zip(EPISODE_LABELS, episodes[code].tolist(), strict=True)
            },
        }
    known = sum(int(cross[code].sum()) for code in STATUS)
    completed_rated = int(cross[2, 1:].sum())
    active = user_rows > 0
    has_profile = user_rated_completed > 0

    report: dict[str, Any] = {
        "source": {
            "file": args.archive.name,
            "bytes": args.archive.stat().st_size,
            "rows": rows,
            "invalid_rows": invalid,
        },
        "users": int(active.sum()),
        "rows_with_an_unlisted_status_code": total - known,
        "share_of_rows_on_catalog_titles": catalog_rows / total,
        "by_status": by_status,
        "relation_to_training_file": {
            "completed_and_rated_rows": completed_rated,
            "rating_complete_rows": RATING_COMPLETE_ROWS,
            "matches_rating_complete": completed_rated == RATING_COMPLETE_ROWS,
            "share_of_animelist_the_models_currently_use": completed_rated / total,
            "rows_the_models_never_see": total - completed_rated,
        },
        "additional_signal": {
            "completed_but_unrated": int(cross[2, 0]),
            "started_not_finished": int(cross[1].sum() + cross[3].sum()),
            "dropped": int(cross[4].sum()),
            "plan_to_watch": int(cross[6].sum()),
            "engaged_but_unseen_per_completed_rated_row": float(
                (cross[1].sum() + cross[3].sum() + cross[2, 0]) / completed_rated
            ),
        },
        "dropped_like_rate_by_progress": {
            label: {"rated": int(rated), "share_8_or_higher": float(liked / rated) if rated else None}
            for label, (rated, liked) in zip(EPISODE_LABELS, dropped_like.tolist(), strict=True)
        },
        "per_user": {
            "users_with_a_plan_to_watch_list": float((user_plan[active] > 0).mean()),
            "median_plan_to_watch_entries": float(np.median(user_plan[active])),
            "mean_plan_to_watch_entries": float(user_plan[active].mean()),
            "users_with_no_completed_rated_title": float((~has_profile[active]).mean()),
            "users_with_no_completed_rated_title_but_other_engagement": float(
                ((~has_profile) & (user_other_engaged > 0))[active].mean()
            ),
            "median_unseen_engagement_per_user": float(np.median(user_other_engaged[active])),
            "median_completed_rated_per_user": float(np.median(user_rated_completed[active])),
        },
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "animelist_status_profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'animelist_status_profile.json'} in {report['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
