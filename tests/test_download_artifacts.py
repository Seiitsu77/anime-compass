"""Startup artifact download: required files gate boot, optional ones do not."""

from __future__ import annotations

import hashlib
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from scripts import download_artifacts


def write_manifest(tmp_path: Path, artifacts: dict[str, Any]) -> Path:
    path = tmp_path / "artifacts.manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )
    return path


def entry(relative: str, payload: bytes, *, required: bool | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    if required is not None:
        record["required"] = required
    return record


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: Any) -> Path:
    root = tmp_path / "project"
    (root / "data" / "processed").mkdir(parents=True)
    monkeypatch.setattr(download_artifacts, "PROJECT_ROOT", root)
    return root


def test_a_present_and_matching_artifact_is_not_downloaded(project_root: Path, tmp_path: Path, monkeypatch: Any):
    payload = b"already here"
    (project_root / "data" / "processed" / "catalog.json").write_bytes(payload)
    manifest = write_manifest(tmp_path, {"catalog.json": entry("data/processed/catalog.json", payload)})

    def fail_download(url: str, destination: Path) -> None:
        raise AssertionError("a verified artifact must not be re-downloaded")

    monkeypatch.setattr(download_artifacts, "download", fail_download)
    verified = download_artifacts.ensure_artifacts("owner/name", manifest_path=manifest)
    assert [path.name for path in verified] == ["catalog.json"]


def test_a_failed_required_download_raises(project_root: Path, tmp_path: Path, monkeypatch: Any):
    manifest = write_manifest(tmp_path, {"catalog.json": entry("data/processed/catalog.json", b"x")})

    def failing(url: str, destination: Path) -> None:
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(download_artifacts, "download", failing)
    with pytest.raises(RuntimeError, match="Failed to download catalog.json"):
        download_artifacts.ensure_artifacts("owner/name", manifest_path=manifest)


def test_a_failed_optional_download_is_skipped(project_root: Path, tmp_path: Path, monkeypatch: Any):
    """An optional channel's artifact must not stop the application from starting."""
    manifest = write_manifest(
        tmp_path,
        {"semantic.npz": entry("data/processed/semantic.npz", b"x", required=False)},
    )

    def failing(url: str, destination: Path) -> None:
        raise urllib.error.URLError("not in the dataset repo yet")

    monkeypatch.setattr(download_artifacts, "download", failing)
    assert download_artifacts.ensure_artifacts("owner/name", manifest_path=manifest) == []


def test_an_optional_artifact_failing_checksum_is_skipped_not_served(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: Any,
):
    manifest = write_manifest(
        tmp_path,
        {"semantic.npz": entry("data/processed/semantic.npz", b"expected", required=False)},
    )

    def corrupt(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"corrupted")

    monkeypatch.setattr(download_artifacts, "download", corrupt)
    assert download_artifacts.ensure_artifacts("owner/name", manifest_path=manifest) == []
    # A mismatched file is removed rather than left for the app to load.
    assert not (project_root / "data" / "processed" / "semantic.npz").exists()


def test_a_required_artifact_failing_checksum_still_raises(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: Any,
):
    manifest = write_manifest(tmp_path, {"catalog.json": entry("data/processed/catalog.json", b"expected")})

    def corrupt(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"corrupted")

    monkeypatch.setattr(download_artifacts, "download", corrupt)
    with pytest.raises(RuntimeError, match="Checksum or size validation failed"):
        download_artifacts.ensure_artifacts("owner/name", manifest_path=manifest)


def test_a_missing_optional_artifact_does_not_block_a_required_one(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: Any,
):
    payload = b"catalog bytes"
    manifest = write_manifest(
        tmp_path,
        {
            "catalog.json": entry("data/processed/catalog.json", payload),
            "semantic.npz": entry("data/processed/semantic.npz", b"y", required=False),
        },
    )

    def selective(url: str, destination: Path) -> None:
        if destination.name == "semantic.npz":
            raise urllib.error.URLError("absent")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    monkeypatch.setattr(download_artifacts, "download", selective)
    verified = download_artifacts.ensure_artifacts("owner/name", manifest_path=manifest)
    assert [path.name for path in verified] == ["catalog.json"]


def test_the_shipped_manifest_marks_semantic_embeddings_optional():
    """The catalog and collaborative index gate boot; the semantic channel does not."""
    manifest = json.loads(Path("data/artifacts.manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    assert artifacts["anime_catalog.json"]["required"] is True
    assert artifacts["collaborative_embeddings.npz"]["required"] is True
    assert artifacts["semantic_embeddings.npz"]["required"] is False


def test_an_unknown_artifact_name_is_rejected(project_root: Path, tmp_path: Path):
    manifest = write_manifest(tmp_path, {"catalog.json": entry("data/processed/catalog.json", b"x")})
    with pytest.raises(ValueError, match="Unknown artifact"):
        download_artifacts.ensure_artifacts(
            "owner/name",
            manifest_path=manifest,
            selected_artifacts=["nope.npz"],
        )


def test_a_malformed_repo_id_is_rejected(tmp_path: Path):
    manifest = write_manifest(tmp_path, {"catalog.json": entry("data/processed/catalog.json", b"x")})
    with pytest.raises(ValueError, match="OWNER/NAME"):
        download_artifacts.ensure_artifacts("not-a-repo", manifest_path=manifest)
