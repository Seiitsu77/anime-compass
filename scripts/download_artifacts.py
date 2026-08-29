from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "artifacts.manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify Anime Compass runtime artifacts from a public Hugging Face Dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_DATASET_REPO", ""),
        help="Hugging Face Dataset repository in owner/name form (or set HF_DATASET_REPO).",
    )
    parser.add_argument(
        "--revision",
        default=os.getenv("HF_DATASET_REVISION", "main"),
        help="Dataset branch, tag, or commit SHA (default: main).",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        help="Download only this artifact filename; repeat to select more than one.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing artifacts even when they verify.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("artifacts"), dict):
        raise ValueError("Unsupported or malformed artifact manifest")
    return payload


def verifies(path: Path, metadata: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(metadata["size_bytes"])
        and sha256_file(path) == str(metadata["sha256"]).casefold()
    )


def artifact_url(repo_id: str, revision: str, filename: str) -> str:
    safe_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    safe_revision = urllib.parse.quote(revision, safe="")
    safe_filename = urllib.parse.quote(filename, safe="/")
    return f"https://huggingface.co/datasets/{safe_repo}/resolve/{safe_revision}/{safe_filename}?download=true"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "anime-compass-artifact-downloader/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_artifacts(
    repo_id: str,
    *,
    revision: str = "main",
    manifest_path: Path = DEFAULT_MANIFEST,
    selected_artifacts: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download missing runtime artifacts and reject files that do not match the manifest."""
    repo_id = repo_id.strip()
    if not repo_id or repo_id.count("/") != 1:
        raise ValueError("Provide a Hugging Face Dataset repository in OWNER/NAME form.")

    manifest = load_manifest(manifest_path)
    available: dict[str, dict[str, Any]] = manifest["artifacts"]
    selected = selected_artifacts or list(available)
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError("Unknown artifact(s): " + ", ".join(unknown))

    verified: list[Path] = []
    for filename in selected:
        metadata = available[filename]
        destination = PROJECT_ROOT / str(metadata["path"])
        if not force and verifies(destination, metadata):
            print(f"verified {destination.relative_to(PROJECT_ROOT)}")
            verified.append(destination)
            continue

        # Artifacts marked `required: false` enable an optional channel. The
        # application runs without them, so a deployment whose dataset repo does
        # not carry them yet degrades instead of failing to start.
        optional = metadata.get("required") is False

        url = artifact_url(repo_id, revision, filename)
        print(f"downloading {filename} from {repo_id}@{revision}")
        try:
            download(url, destination)
        except (OSError, urllib.error.URLError) as exc:
            if optional:
                print(f"skipped optional {filename}: {type(exc).__name__}")
                continue
            raise RuntimeError(f"Failed to download {filename}: {type(exc).__name__}") from exc
        if not verifies(destination, metadata):
            destination.unlink(missing_ok=True)
            if optional:
                print(f"skipped optional {filename}: checksum or size mismatch")
                continue
            raise RuntimeError(f"Checksum or size validation failed for {filename}")
        print(f"verified {destination.relative_to(PROJECT_ROOT)}")
        verified.append(destination)
    return verified


def main() -> int:
    args = parse_args()
    try:
        ensure_artifacts(
            args.repo_id,
            revision=args.revision,
            manifest_path=args.manifest,
            selected_artifacts=args.artifacts,
            force=args.force,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
