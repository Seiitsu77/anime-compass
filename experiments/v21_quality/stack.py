"""The frozen evaluation stack, loaded once and shared by the V2.1 scripts.

    train-only ALS  ->  top-N candidates  ->  the confirmed LambdaMART

Exactly what `scripts/evaluate_reranker.py` scores. Nothing here reads
`data/processed/`: the production artifacts are trained on the held-out
positives and are invalid for measuring.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.anime_agent.evaluation.reranking import RerankerFeatureSpace  # noqa: E402
from backend.anime_agent.evaluation.split import SplitStore  # noqa: E402
from backend.anime_agent.recommender import series_key  # noqa: E402
from experiments.v21_quality.protocol import apply_series_policy  # noqa: E402
from scripts.evaluate_reranker import FrozenALS, load_als  # noqa: E402

PERSONALIZED = PROJECT_ROOT / "data" / "evaluation" / "personalized"
SPLIT = PERSONALIZED / "splits" / "holdout_seed42_pos8.sqlite"
ARTIFACTS = PERSONALIZED / "artifacts" / "holdout_seed42_pos8"
CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
RERANKER_DIR = PERSONALIZED / "results" / "reranker"
INSPECTED = PERSONALIZED / "metadata" / "inspected_user_ids.json"
RESULTS = Path(__file__).resolve().parent / "results"
POPULATIONS = Path(__file__).resolve().parent / "populations"
WINDOW = 300  # RETRIEVAL_ALS_TOP_N in production


@dataclass
class Stack:
    anime_ids: np.ndarray
    als: FrozenALS
    space: RerankerFeatureSpace
    booster: Any
    store: SplitStore
    row_by_id: dict[int, int]
    keys: list[str]
    types: list[str]
    popularity: np.ndarray
    bucket: np.ndarray

    @property
    def items(self) -> int:
        return len(self.anime_ids)

    def rows(self, pairs: Any) -> list[int]:
        """Artifact rows for (anime_id, rating) pairs, dropping non-catalog IDs."""
        return [self.row_by_id[anime_id] for anime_id, _rating in pairs if anime_id in self.row_by_id]

    def als_scores(self, profile: list[int], excluded: Any = ()) -> np.ndarray:
        """Raw fold-in scores with the profile and `excluded` rows masked out."""
        scores = self.als.scores(list(profile))
        scores[list(profile)] = -np.inf
        blocked = list(excluded)
        if blocked:
            scores[blocked] = -np.inf
        return scores

    @staticmethod
    def top(scores: np.ndarray, depth: int) -> list[int]:
        take = min(depth, int(np.isfinite(scores).sum()))
        if take <= 0:
            return []
        top = np.argpartition(-scores, take - 1)[:take]
        return [int(row) for row in top[np.argsort(-scores[top], kind="stable")]]

    def rerank(self, profile: list[int], candidates: list[int], scores: np.ndarray) -> list[int]:
        """LambdaMART order of one candidate list, built from the profile alone."""
        if not candidates:
            return []
        features = self.space.build(list(profile), candidates, scores[candidates].astype(np.float32))
        learned = np.asarray(self.booster.predict(features), dtype=np.float32)
        return [candidates[index] for index in np.argsort(-learned, kind="stable")]

    def sequence(
        self,
        profile: list[int],
        *,
        excluded: Any = (),
        windows: int = 4,
        policy: str = "production",
        reranked: bool = True,
        timings: dict[str, list[float]] | None = None,
    ) -> tuple[list[int], list[int], list[list[int]]]:
        """The paged feed, as the service builds it: ALS windows, each reranked.

        Returns the filtered sequence, how many titles each window contributed,
        and each window's unfiltered order.
        """
        scores = self.als_scores(profile, excluded)
        sequence: list[int] = []
        sizes: list[int] = []
        raw: list[list[int]] = []
        ranked_so_far: list[int] = []
        for _window in range(windows):
            started = time.perf_counter()
            candidates = self.top(scores, WINDOW)
            retrieved = time.perf_counter()
            if not candidates:
                break
            ordered = self.rerank(profile, candidates, scores) if reranked else candidates
            if timings is not None:
                timings.setdefault("retrieval_ms", []).append((retrieved - started) * 1000)
                timings.setdefault("rerank_ms", []).append((time.perf_counter() - retrieved) * 1000)
            scores[candidates] = -np.inf  # the next window starts where this one ended
            raw.append(ordered)
            ranked_so_far.extend(ordered)
            # The policy is applied to everything ranked so far, so a franchise
            # shown in an earlier window stays counted in a later one.
            filtered = apply_series_policy(ranked_so_far, profile, self.keys, policy, types=self.types)
            sizes.append(len(filtered) - len(sequence))
            sequence = filtered
        return sequence, sizes, raw


def load_stack() -> Stack:
    import lightgbm as lgb

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    anime_ids, factors, meta = load_als(ARTIFACTS / "als_train_only.npz")
    als = FrozenALS(factors, float(meta.get("alpha", 5.0)), float(meta.get("regularization", 0.05)))
    space = RerankerFeatureSpace.from_artifacts(
        catalog,
        anime_ids,
        popularity_path=ARTIFACTS / "popularity_train_only.npz",
        quality_path=ARTIFACTS / "countsketch_train_only.npz",
        item_item_path=ARTIFACTS / "item_item_train_only.npz",
    )
    booster = lgb.Booster(model_file=str(RERANKER_DIR / "lambdamart.txt"))
    by_id = {int(item["id"]): item for item in catalog}
    keys = [series_key(str(by_id.get(int(a), {}).get("title") or a)) for a in anime_ids.tolist()]
    types = [str(by_id.get(int(a), {}).get("type") or "") for a in anime_ids.tolist()]
    with np.load(ARTIFACTS / "popularity_train_only.npz", allow_pickle=False) as payload:
        assert np.array_equal(payload["anime_ids"], anime_ids), "popularity artifact is not aligned with ALS"
        popularity = np.asarray(payload["positive_count"], dtype=np.float64)
    order = np.lexsort((anime_ids, -popularity))
    bucket = np.empty(len(anime_ids), dtype="<U4")
    bucket[order[: int(len(anime_ids) * 0.2)]] = "head"
    bucket[order[int(len(anime_ids) * 0.2) : int(len(anime_ids) * 0.5)]] = "mid"
    bucket[order[int(len(anime_ids) * 0.5) :]] = "tail"
    return Stack(
        anime_ids=anime_ids,
        als=als,
        space=space,
        booster=booster,
        store=SplitStore(SPLIT),
        row_by_id=space.index_by_id,
        keys=keys,
        types=types,
        popularity=popularity,
        bucket=bucket,
    )


def spent_user_ids() -> dict[str, set[int]]:
    """Every user some experiment has already scored, by where they were used."""
    inspected = {int(value) for value in json.loads(INSPECTED.read_text(encoding="utf-8"))}
    reranker = json.loads(gzip.open(RERANKER_DIR / "confirm_user_ids.json.gz").read())
    spent = {
        "inspected": inspected,
        "reranker_train": {int(value) for value in reranker["train"]},
        "reranker_val": {int(value) for value in reranker["val"]},
        "reranker_confirm": {int(value) for value in reranker["confirm"]},
    }
    for path in sorted(POPULATIONS.glob("*.json")) if POPULATIONS.exists() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, ids in record.get("cohorts", {}).items():
            spent[f"{path.stem}:{name}"] = {int(value) for value in ids}
    return spent


def reserved_user_ids() -> dict[str, set[int]]:
    """Users set aside as untouched confirmation sets. Never score or sample these.

    A new experiment draws its populations from users that are in neither
    `spent_user_ids()` nor here, and opens exactly one reserved set, once.
    """
    reserved: dict[str, set[int]] = {}
    for path in sorted(POPULATIONS.glob("*.json")) if POPULATIONS.exists() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, ids in record.get("reserved", {}).items():
            reserved[f"{path.stem}:{name}"] = {int(value) for value in ids}
    return reserved
