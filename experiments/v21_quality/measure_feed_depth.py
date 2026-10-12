"""How deep does the paged feed go, and what does a page cost?

Drives the real serving code, in process, with the production artifacts:
`RecommendationService.page` over the production ALS factors and the production
LambdaMART. It answers three questions about pagination for realistic profiles
of 1 to 50 Likes:

* how many distinct ranked titles a profile gets, window by window;
* what a page costs when the sequence is cached, when it is not, and when a
  restarted instance has to rebuild several windows;
* how much one exclusion moves the order of the first batch.

This measures serving behaviour only. The production artifacts are trained on
every interaction, so nothing here is scored against held-out labels; quality
by depth is in `small_profile_eval.py`, on the evaluation artifacts. Profiles
are real users' Likes, used as inputs and never as targets.

Read-only: no artifact is written or modified.

    python experiments/v21_quality/measure_feed_depth.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.schemas import RecommendPageRequest  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.main import _load_als_index, _load_collaborative_index, _load_reranker  # noqa: E402
from app.repositories.session_repository import SQLiteSessionRepository  # noqa: E402
from app.services.recommendation_sequence import decode_cursor, encode_cursor  # noqa: E402
from app.services.recommendation_service import RecommendationService  # noqa: E402
from backend.anime_agent.data_pipeline import load_or_create_catalog  # noqa: E402
from backend.anime_agent.evaluation.split import SplitStore  # noqa: E402
from backend.anime_agent.fast_path import FastPathConfig  # noqa: E402
from backend.anime_agent.recommender import AnimeRecommender  # noqa: E402
from backend.anime_agent.retrieval import RetrievalConfig  # noqa: E402
from experiments.v21_quality.protocol import sample_profile  # noqa: E402
from experiments.v21_quality.stack import RESULTS, SPLIT, spent_user_ids  # noqa: E402

PROFILE_SIZES = (1, 3, 5, 10, 20, 50)
USERS = 150
BATCH = 18
PAGE = 36  # what the client requests: one batch on screen, one in reserve


def percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(data, 50)), "p95": float(np.percentile(data, 95)), "mean": float(data.mean())}


def build_service(settings: Settings, catalog: list[dict[str, Any]], recommender: AnimeRecommender, name: str):
    collaborative = _load_collaborative_index(settings, catalog)
    als = _load_als_index(settings, catalog, quality_source=collaborative)
    reranker = _load_reranker(settings, catalog, als)
    sessions = SQLiteSessionRepository(
        f"sqlite:///{(Path(tempfile.gettempdir()) / f'anime_compass_depth_{name}.db').as_posix()}", retention_days=1
    )
    return RecommendationService(
        recommender,
        sessions,
        als_index=als,
        fallback_index=collaborative,
        fast_path_config=FastPathConfig(
            reranker=reranker, retrieval=RetrievalConfig(als_top_n=settings.retrieval_als_top_n)
        ),
        sequence_windows=settings.recommendation_sequence_windows,
        starter_depth=settings.recommendation_starter_depth,
    )


async def read_sequence(
    service: RecommendationService, liked: list[int]
) -> tuple[list[int], dict[str, Any], list[dict[str, Any]]]:
    """Every page of one sequence, as the client reads it."""
    cursor = None
    shown: list[int] = []
    pages: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    for _guard in range(200):
        started = time.perf_counter()
        response = await service.page(RecommendPageRequest(liked_ids=liked, cursor=cursor, limit=PAGE))
        elapsed = (time.perf_counter() - started) * 1000
        sequence = response["sequence"]
        shown.extend(int(item["id"]) for item in response["results"])
        pages.append(
            {
                "ms": elapsed,
                "cache": sequence["cache"],
                "offset": sequence["offset"],
                "windows": sequence["windows_computed"],
            }
        )
        last = sequence
        cursor = sequence["next_cursor"]
        if not sequence["has_more"]:
            break
    return shown, last, pages


async def main() -> int:
    started = time.perf_counter()
    settings = Settings()
    catalog = load_or_create_catalog(PROJECT_ROOT)
    recommender = AnimeRecommender(catalog)
    service = build_service(settings, catalog, recommender, "a")
    restarted = build_service(settings, catalog, recommender, "b")  # a second instance with an empty cache
    assert service.fast_path_config.reranker is not None, "the production reranker did not load"
    info = service.als_index.model_info()
    known = service.als_index.index_by_id
    print(
        f"loaded production stack in {time.perf_counter() - started:.0f}s: {info['items']} items, role {info['artifact_role']}"
    )

    store = SplitStore(SPLIT)
    users = sorted(spent_user_ids()["reranker_confirm"])
    profiles: list[tuple[int, list[int]]] = []
    for user_id in users:
        user = store.get_user(user_id)
        liked = [anime_id for anime_id in (user.train_positive_ids if user else ()) if anime_id in known]
        if len(liked) >= max(PROFILE_SIZES):
            profiles.append((user_id, liked))
        if len(profiles) == USERS:
            break

    report: dict[str, Any] = {
        "protocol": {
            "users": len(profiles),
            "profiles": "K Likes drawn in a fixed hash order from real users' liked titles; inputs only, never targets",
            "stack": "production ALS factors and production LambdaMART, through RecommendationService.page, in process",
            "als_artifact_sha256": info["artifact_sha256"],
            "window_size": service.sequence_limits()["window_size"],
            "max_windows": service.sequence_limits()["max_windows"],
            "page_size_requested": PAGE,
            "note": "Latency is in-process on the development machine; the HTTP layer adds a few milliseconds.",
        },
        "by_profile_size": {},
    }

    for size in PROFILE_SIZES:
        totals: list[float] = []
        window_sizes: list[list[int]] = []
        miss_ms: list[float] = []
        hit_ms: list[float] = []
        first_ms: list[float] = []
        resume_ms: dict[int, list[float]] = {offset: [] for offset in (36, 216, 432, 648)}
        resume_ok = 0
        resume_total = 0
        repeats = 0
        overlap: list[float] = []
        same_order: list[float] = []
        for user_id, liked_all in profiles:
            liked = sample_profile(user_id, liked_all, size)
            service._sequences.clear()
            shown, last, pages = await read_sequence(service, liked)
            totals.append(len(shown))
            window_sizes.append(list(last["window_sizes"]))
            repeats += len(shown) - len(set(shown))
            first_ms.append(pages[0]["ms"])
            miss_ms.extend(page["ms"] for page in pages[1:] if page["cache"] == "miss")
            hit_ms.extend(page["ms"] for page in pages if page["cache"] == "hit")

            # A restarted instance resuming at several depths: it must rebuild every window before the offset.
            first = await service.page(RecommendPageRequest(liked_ids=liked, limit=PAGE))
            fingerprint = (
                decode_cursor(first["sequence"]["next_cursor"])[0] if first["sequence"]["next_cursor"] else None
            )
            for offset in resume_ms:
                if fingerprint is None or offset >= len(shown):
                    continue
                restarted._sequences.clear()
                tick = time.perf_counter()
                page = await restarted.page(
                    RecommendPageRequest(liked_ids=liked, limit=PAGE, cursor=encode_cursor(fingerprint, offset))
                )
                resume_ms[offset].append((time.perf_counter() - tick) * 1000)
                resume_total += 1
                resume_ok += [int(item["id"]) for item in page["results"]] == shown[offset : offset + PAGE]

            # How much does one exclusion move the first batch? (A reload after Already Watched.)
            excluded = shown[2]
            moved = await service.page(RecommendPageRequest(liked_ids=liked, excluded_ids=[excluded], limit=BATCH))
            after = [int(item["id"]) for item in moved["results"]]
            expected = [anime_id for anime_id in shown[: BATCH + 1] if anime_id != excluded][:BATCH]
            overlap.append(len(set(after) & set(expected)) / BATCH)
            same_order.append(float(after == expected))

        sizes = np.asarray([row + [0] * (4 - len(row)) for row in window_sizes], dtype=np.float64)
        report["by_profile_size"][str(size)] = {
            "distinct_titles": {
                "mean": float(np.mean(totals)),
                "p05": float(np.percentile(totals, 5)),
                "min": float(np.min(totals)),
                "max": float(np.max(totals)),
            },
            "batches_of_18": float(np.mean(totals) / BATCH),
            "titles_per_window": [float(value) for value in sizes.mean(axis=0)],
            "repeats": repeats,
            "latency_ms": {
                "first_page": percentiles(first_ms),
                "page_from_cache": percentiles(hit_ms),
                "page_that_ranks_a_new_window": percentiles(miss_ms) if miss_ms else None,
                "resume_on_a_fresh_instance": {
                    f"offset_{offset}": percentiles(values) for offset, values in resume_ms.items() if values
                },
            },
            "resume_matches_original": {"pages": resume_total, "identical": resume_ok},
            "after_one_exclusion": {
                "first_batch_overlap": float(np.mean(overlap)),
                "first_batch_identical_order": float(np.mean(same_order)),
            },
        }
        row = report["by_profile_size"][str(size)]
        print(
            f"K={size:>2} depth {row['distinct_titles']['mean']:.0f} (min {row['distinct_titles']['min']:.0f}) "
            f"windows {[round(v) for v in row['titles_per_window']]} first {row['latency_ms']['first_page']['p50']:.1f} ms "
            f"cached {row['latency_ms']['page_from_cache']['p50']:.1f} ms resume {resume_ok}/{resume_total} "
            f"overlap after exclusion {row['after_one_exclusion']['first_batch_overlap']:.3f}"
        )

    report["runtime_seconds"] = round(time.perf_counter() - started, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "feed_depth.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
