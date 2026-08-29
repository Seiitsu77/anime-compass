from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from app.core.config import get_settings  # noqa: E402
from app.embeddings.index import SemanticEmbeddingIndex  # noqa: E402
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider  # noqa: E402
from backend.anime_agent.data_pipeline import load_or_create_catalog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or validate the cached pretrained anime embedding artifact.")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even when a valid artifact exists.")
    parser.add_argument(
        "--verify-only", action="store_true", help="Validate the existing artifact without rebuilding it."
    )
    parser.add_argument("--catalog", type=Path, help="Optional processed anime_catalog.json path.")
    parser.add_argument("--output", type=Path, help="Optional output .npz path.")
    parser.add_argument("--offline", action="store_true", help="Use only an already-cached model snapshot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model,
        model_revision=settings.embedding_model_revision,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        local_files_only=args.offline,
    )
    provider_dimension = int(str(provider.model_info().get("dimension") or 0))
    if provider_dimension != settings.embedding_dimensions:
        raise SystemExit(
            f"Model reports {provider_dimension} dimensions, but EMBEDDING_DIMENSIONS={settings.embedding_dimensions}."
        )
    if args.catalog:
        catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    else:
        catalog = load_or_create_catalog(PROJECT_ROOT)
    artifact_path = args.output or settings.semantic_artifact_path
    if artifact_path.exists() and not args.force:
        index = SemanticEmbeddingIndex.load(
            artifact_path,
            provider,
            catalog,
            expected_dimension=settings.embedding_dimensions,
        )
        print(
            f"Validated {len(index.anime_ids)} cached pretrained embeddings "
            f"({index.matrix.shape[1]} dimensions) at {artifact_path}"
        )
        return
    if args.verify_only:
        raise SystemExit(f"Semantic artifact does not exist: {artifact_path}")
    index = SemanticEmbeddingIndex.build(artifact_path, provider, catalog)
    print(f"Wrote {len(index.anime_ids)} pretrained embeddings ({index.matrix.shape[1]} dimensions) to {artifact_path}")


if __name__ == "__main__":
    main()
