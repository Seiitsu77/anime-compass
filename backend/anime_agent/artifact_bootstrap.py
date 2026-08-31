"""Fetch and verify the deployment payload at startup.

A hosted deployment needs a 7.1 MB ALS artifact and a 6.6 MB compact serving
catalog. They can be bundled with a small demo repository or fetched from an
artifact host. The risk this module exists to prevent is subtle: a demo that
advertises the ALS benchmark while quietly falling back to the weaker
CountSketch model because a file is missing or corrupted.

So the contract is: fetch, verify, or say so. There is no path where an
unverified file is loaded and presented as the production model.

Only the standard library is used, matching the existing catalog downloader.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# The artifact is ~7.1 MB and the serving catalog ~6.6 MB. This bound stops a
# misconfigured URL from streaming something unbounded into a small hosted
# container.
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
ALLOWED_SCHEMES = frozenset({"https"})


@dataclass(frozen=True)
class BootstrapResult:
    """What happened, in enough detail for an honest health display."""

    path: Path
    present: bool
    downloaded: bool
    verified: bool
    detail: str

    @property
    def usable(self) -> bool:
        return self.present and self.verified


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    """Download to a temporary file, then move it into place atomically.

    A partial download must never be left where the loader would pick it up as
    a real artifact.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Artifact URL must use https, got {parsed.scheme or 'no scheme'!r}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "anime-compass-artifact-bootstrap"})
    handle = tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".part")
    temporary = Path(handle.name)
    try:
        with handle, urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - scheme checked above
            written = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_ARTIFACT_BYTES:
                    raise ValueError(f"Artifact exceeds {MAX_ARTIFACT_BYTES} bytes; refusing to continue")
                handle.write(chunk)
        shutil.move(str(temporary), str(destination))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def ensure_production_artifact(
    path: Path,
    *,
    url: str | None = None,
    expected_sha256: str | None = None,
) -> BootstrapResult:
    """Make the artifact present and verified, or report why not.

    Never raises for an ordinary missing-or-unreachable artifact: the caller
    renders a clear unavailable state instead. It does raise for a
    misconfigured URL, which is an operator error rather than a runtime one.
    """
    path = Path(path)

    if path.exists():
        if not expected_sha256:
            return BootstrapResult(path, True, False, True, "present (no checksum pinned)")
        actual = _sha256(path)
        if actual == expected_sha256:
            return BootstrapResult(path, True, False, True, "present and checksum verified")
        # A wrong local file is worse than none: remove it so a configured URL
        # can replace it rather than being shadowed forever.
        detail = f"local artifact checksum mismatch (got {actual[:12]}..., expected {expected_sha256[:12]}...)"
        if not url:
            return BootstrapResult(path, True, False, False, detail)
        path.unlink(missing_ok=True)

    if not url:
        return BootstrapResult(path, False, False, False, "artifact missing and no ALS_ARTIFACT_URL configured")

    try:
        _download(url, path)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        if isinstance(exc, ValueError) and "https" in str(exc):
            raise
        return BootstrapResult(path, path.exists(), False, False, f"download failed: {type(exc).__name__}")

    if expected_sha256:
        actual = _sha256(path)
        if actual != expected_sha256:
            path.unlink(missing_ok=True)
            return BootstrapResult(
                path,
                False,
                True,
                False,
                f"downloaded artifact failed verification (got {actual[:12]}...)",
            )
        return BootstrapResult(path, True, True, True, "downloaded and checksum verified")

    return BootstrapResult(path, True, True, True, "downloaded (no checksum pinned)")


def bootstrap_from_environment(
    default_path: Path,
    *,
    default_expected_sha256: str | None = None,
) -> BootstrapResult:
    """Read artifact configuration, retaining a repository-pinned checksum."""
    path = Path(os.environ.get("ALS_ARTIFACT_PATH") or default_path)
    return ensure_production_artifact(
        path,
        url=os.environ.get("ALS_ARTIFACT_URL") or None,
        expected_sha256=os.environ.get("ALS_EXPECTED_SHA256") or default_expected_sha256,
    )


def bootstrap_catalog_from_environment(
    default_path: Path,
    *,
    default_expected_sha256: str | None = None,
) -> BootstrapResult:
    """Same contract for the serving catalog.

    The catalog gets the identical fetch-verify-or-say-so treatment rather than
    a looser one. A silently wrong catalog is the failure the ALS loader's
    identity digest exists to catch, and catching it at download time gives a
    clearer message than catching it at model load.
    """
    path = Path(os.environ.get("SERVING_CATALOG_PATH") or default_path)
    return ensure_production_artifact(
        path,
        url=os.environ.get("SERVING_CATALOG_URL") or None,
        expected_sha256=os.environ.get("SERVING_CATALOG_EXPECTED_SHA256") or default_expected_sha256,
    )
