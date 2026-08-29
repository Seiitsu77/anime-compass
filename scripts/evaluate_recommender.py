from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from app.core.config import Settings  # noqa: E402
from app.embeddings.index import SemanticEmbeddingIndex  # noqa: E402
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider  # noqa: E402
from backend.anime_agent.collaborative import CollaborativeIndex  # noqa: E402
from backend.anime_agent.data_pipeline import load_or_create_catalog  # noqa: E402
from backend.anime_agent.recommender import (  # noqa: E402
    DEFAULT_CHANNEL_WEIGHTS,
    AnimeRecommender,
    entity_name_variants,
    series_key,
)

BENCHMARK_PATH = PROJECT_ROOT / "data" / "evaluation" / "benchmark.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "results.json"
DEFAULT_MARKDOWN_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "results.md"
TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reproducible content-recommender baselines.")
    parser.add_argument("--catalog", type=Path, help="Optional processed anime_catalog.json path.")
    parser.add_argument("--artifact", type=Path, help="Optional semantic embedding .npz path.")
    parser.add_argument(
        "--collaborative-artifact",
        type=Path,
        help="Optional collaborative embedding .npz path.",
    )
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def one_channel_weights(channel: str) -> dict[str, float]:
    return {name: float(name == channel) for name in DEFAULT_CHANNEL_WEIGHTS}


def load_catalog(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return load_or_create_catalog(PROJECT_ROOT)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Processed catalog must be a JSON array")
    return payload


def load_semantic_index(
    artifact_path: Path,
    catalog: list[dict[str, Any]],
) -> tuple[SemanticEmbeddingIndex, dict[str, Any]]:
    settings = Settings()
    provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model,
        model_revision=settings.embedding_model_revision,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        local_files_only=True,
    )
    index = SemanticEmbeddingIndex.load(
        artifact_path,
        provider,
        catalog,
        expected_dimension=settings.embedding_dimensions,
    )
    return index, index.model_info()


def popularity_baseline(
    recommender: AnimeRecommender,
    case: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    excluded_titles = [
        *case.get("reference_titles", []),
        *case.get("excluded_titles", []),
    ]
    ranked, _ = recommender.rank_catalog(
        include_genres=case.get("include_genres"),
        exclude_genres=case.get("exclude_genres"),
        formats=case.get("formats"),
        min_score=case.get("min_score"),
        min_year=case.get("min_year"),
        max_year=case.get("max_year"),
        max_episodes=case.get("max_episodes"),
        excluded_titles=excluded_titles,
        sort_by="popularity",
        sort_order="asc",
        limit=len(recommender.catalog),
    )
    blocked_series = {series_key(title) for title in excluded_titles}
    required_actors, resolution_errors = recommender.resolve_voice_actor_constraints(
        case.get("required_voice_actors", []),
        [],
    )
    if resolution_errors:
        return []
    allowed_actor_ids: set[int] | None = None
    for actor in required_actors:
        related_ids = {int(value) for value in actor.get("related_anime_ids", [])}
        allowed_actor_ids = related_ids if allowed_actor_ids is None else allowed_actor_ids.intersection(related_ids)

    results = []
    seen_series: set[str] = set()
    for item in ranked:
        anime_id = int(item["id"])
        item_series = series_key(item["title"])
        if item_series in blocked_series or item_series in seen_series:
            continue
        if allowed_actor_ids is not None and anime_id not in allowed_actor_ids:
            continue
        seen_series.add(item_series)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def recommendation_results(
    recommender: AnimeRecommender,
    case: dict[str, Any],
    top_k: int,
    *,
    weights: dict[str, float] | None,
    diversity_strength: float,
) -> list[dict[str, Any]]:
    return recommender.recommend(
        reference_titles=case.get("reference_titles"),
        include_genres=case.get("include_genres"),
        exclude_genres=case.get("exclude_genres"),
        formats=case.get("formats"),
        min_score=case.get("min_score"),
        min_year=case.get("min_year"),
        max_year=case.get("max_year"),
        max_episodes=case.get("max_episodes"),
        excluded_titles=case.get("excluded_titles"),
        required_voice_actors=case.get("required_voice_actors"),
        free_text_preferences=case.get("free_text_preferences", ""),
        one_per_series=True,
        diversity_strength=diversity_strength,
        weights=weights,
        limit=top_k,
    )


def hard_filter_satisfaction(results: list[dict[str, Any]], case: dict[str, Any]) -> float:
    if not results:
        return 0.0
    wanted = {value.casefold() for value in case.get("include_genres", [])}
    blocked = {value.casefold() for value in case.get("exclude_genres", [])}
    formats = {value.casefold() for value in case.get("formats", [])}
    excluded_series = {series_key(value) for value in case.get("excluded_titles", [])}
    checks = []
    for item in results:
        genres = {value.casefold() for value in item.get("genres", [])}
        checks.append(
            (not wanted or bool(wanted.intersection(genres)))
            and not blocked.intersection(genres)
            and (not formats or str(item.get("type") or "").casefold() in formats)
            and (case.get("min_score") is None or float(item.get("score") or -1) >= float(case["min_score"]))
            and (case.get("min_year") is None or int(item.get("start_year") or -1) >= int(case["min_year"]))
            and (case.get("max_year") is None or int(item.get("start_year") or 9999) <= int(case["max_year"]))
            and (
                case.get("max_episodes") is None
                or (item.get("episodes") is not None and int(item["episodes"]) <= int(case["max_episodes"]))
            )
            and series_key(item["title"]) not in excluded_series
        )
    return sum(checks) / len(checks)


def entity_constraint_satisfaction(
    recommender: AnimeRecommender,
    results: list[dict[str, Any]],
    case: dict[str, Any],
) -> float | None:
    required = case.get("required_voice_actors", [])
    if not required:
        return None
    if not results:
        return 0.0
    checks = []
    for item in results:
        roles = recommender.voice_actor_roles_by_anime.get(int(item["id"]), [])
        checks.append(
            all(
                any(
                    entity_name_variants(actor).intersection(entity_name_variants(role.get("voice_actor")))
                    for role in roles
                )
                for actor in required
            )
        )
    return sum(checks) / len(checks)


def hit_rate_at_k(results: list[dict[str, Any]], expected_titles: list[str]) -> float | None:
    if not expected_titles:
        return None
    expected = {series_key(title) for title in expected_titles}
    returned = {series_key(item["title"]) for item in results}
    return float(bool(expected.intersection(returned)))


def genre_recovery(results: list[dict[str, Any]], expected_genres: list[str]) -> float | None:
    if not expected_genres:
        return None
    expected = {genre.casefold() for genre in expected_genres}
    returned = {genre.casefold() for item in results for genre in item.get("genres", [])}
    return len(expected.intersection(returned)) / len(expected)


def intra_list_diversity(results: list[dict[str, Any]]) -> float:
    distances = []
    for index, left in enumerate(results):
        left_genres = {value.casefold() for value in left.get("genres", [])}
        for right in results[index + 1 :]:
            right_genres = {value.casefold() for value in right.get("genres", [])}
            union = left_genres.union(right_genres)
            similarity = len(left_genres.intersection(right_genres)) / len(union) if union else 0.0
            distances.append(1.0 - similarity)
    return statistics.mean(distances) if distances else 0.0


def mean_defined(values: Sequence[float | None]) -> float:
    defined = [value for value in values if value is not None]
    return statistics.mean(defined) if defined else 0.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def evaluate_variant(
    name: str,
    recommender: AnimeRecommender,
    benchmark: list[dict[str, Any]],
    top_k: int,
    *,
    weights: dict[str, float] | None = None,
    diversity_strength: float = 0.0,
) -> dict[str, Any]:
    hard_scores = []
    entity_scores: list[float | None] = []
    hit_scores: list[float | None] = []
    genre_scores: list[float | None] = []
    diversity_scores = []
    latencies = []
    returned_ids: set[int] = set()

    for case in benchmark:
        started = time.perf_counter()
        if name == "popularity":
            results = popularity_baseline(recommender, case, top_k)
        else:
            results = recommendation_results(
                recommender,
                case,
                top_k,
                weights=weights,
                diversity_strength=diversity_strength,
            )
        latencies.append((time.perf_counter() - started) * 1000)
        returned_ids.update(int(item["id"]) for item in results)
        hard_scores.append(hard_filter_satisfaction(results, case))
        entity_scores.append(entity_constraint_satisfaction(recommender, results, case))
        hit_scores.append(hit_rate_at_k(results, case.get("expected_related_titles", [])))
        genre_scores.append(genre_recovery(results, case.get("expected_genres", [])))
        diversity_scores.append(intra_list_diversity(results))

    return {
        "variant": name,
        "hard_filter_satisfaction": mean_defined(hard_scores),
        "entity_constraint_satisfaction": mean_defined(entity_scores),
        "hit_rate_at_k": mean_defined(hit_scores),
        "genre_recovery": mean_defined(genre_scores),
        "catalog_coverage": len(returned_ids) / len(recommender.catalog),
        "intra_list_diversity": statistics.mean(diversity_scores),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": percentile(latencies, 0.95),
        "unique_recommended_titles": len(returned_ids),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Recommender Proxy Evaluation",
        "",
        "These offline metrics measure catalog consistency and recovery against a small, manually curated benchmark. "
        "They do not measure real user satisfaction.",
        "",
        f"Catalog: {report['catalog_count']} titles. Benchmark: {report['benchmark_count']} cases. K={report['top_k']}.",
        "",
        "| model | hard filters | entity constraints | Hit Rate@K | genre recovery | coverage | diversity | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            "| {variant} | {hard_filter_satisfaction:.3f} | {entity_constraint_satisfaction:.3f} | "
            "{hit_rate_at_k:.3f} | {genre_recovery:.3f} | {catalog_coverage:.3f} | "
            "{intra_list_diversity:.3f} | {latency_p50_ms:.1f} | {latency_p95_ms:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- Hard filters: fraction of returned titles satisfying explicit genre, format, score, year, episode, and exclusion constraints.",
            "- Entity constraints: fraction of results with the required catalog relationship, measured only on entity cases.",
            "- Hit Rate@K: fraction of labeled similarity cases where at least one manually expected title family appears in the top K.",
            "- Genre recovery: fraction of expected genre labels present anywhere in each result list.",
            "- Coverage: unique titles returned across the benchmark divided by catalog size.",
            "- Diversity: mean pairwise genre Jaccard distance within each list.",
            "- Latency: local recommender execution only; model loading and API/LLM latency are excluded.",
            "",
            "The benchmark is intentionally a proxy. Its labels are small and subjective, so differences should be treated as engineering diagnostics rather than evidence of user preference quality.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    settings = Settings()
    semantic_index = None
    semantic_info: dict[str, Any] = {"available": False}
    artifact_path = args.artifact or settings.semantic_artifact_path
    if artifact_path.exists():
        try:
            semantic_index, semantic_info = load_semantic_index(artifact_path, catalog)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            semantic_info = {"available": False, "reason": type(exc).__name__}

    collaborative_path = args.collaborative_artifact or settings.collaborative_artifact_path
    collaborative_index = CollaborativeIndex.load(collaborative_path, catalog) if collaborative_path.exists() else None
    collaborative_info = collaborative_index.model_info() if collaborative_index is not None else {"available": False}
    base_recommender = AnimeRecommender(catalog)
    hybrid_recommender = AnimeRecommender(
        catalog,
        semantic_index=semantic_index,
        collaborative_index=collaborative_index,
    )

    variants = [
        ("popularity", base_recommender, None, 0.0),
        ("metadata_tfidf", base_recommender, one_channel_weights("metadata"), 0.0),
        ("synopsis_tfidf", base_recommender, one_channel_weights("synopsis"), 0.0),
        ("lsa", base_recommender, one_channel_weights("lsa"), 0.0),
    ]
    if semantic_index is not None:
        variants.append(
            (
                "pretrained_semantic",
                hybrid_recommender,
                one_channel_weights("semantic_embedding"),
                0.0,
            )
        )
    if collaborative_index is not None:
        variants.append(
            (
                "collaborative",
                hybrid_recommender,
                one_channel_weights("collaborative"),
                0.0,
            )
        )
    variants.append(("final_hybrid", hybrid_recommender, None, 0.12))
    rows = [
        evaluate_variant(name, recommender, benchmark, args.top_k, weights=weights, diversity_strength=diversity)
        for name, recommender, weights, diversity in variants
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_count": len(catalog),
        "benchmark_count": len(benchmark),
        "top_k": args.top_k,
        "semantic_model": semantic_info,
        "collaborative_model": collaborative_info,
        "results": rows,
        "limitations": [
            "Offline proxy metrics do not represent real user satisfaction.",
            "The benchmark is small and manually labeled.",
            "Latency excludes model startup, HTTP, and LLM generation.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
