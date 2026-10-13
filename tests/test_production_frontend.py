"""FastAPI serving the compiled React bundle, and what happens without one.

The bundle is a build product: `npm ci && npm run build` in `frontend/` creates
`frontend/dist`. CI builds it before this module runs. In a checkout where it
has not been built the bundle tests skip with that instruction rather than
fail, because nothing about the Python code is wrong.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, Settings
from app.main import create_app
from tests.test_api import MockProvider

BUNDLE = PROJECT_ROOT / "frontend" / "dist"
LAUNCHER = PROJECT_ROOT / "scripts" / "container_start.sh"
DOCKERFILE = PROJECT_ROOT / "Dockerfile.cloudrun"


def _client(tmp_path, catalog, frontend_dir: Path) -> TestClient:
    settings = Settings(
        _env_file=None,
        frontend_dir=frontend_dir,
        database_url=f"sqlite:///{(tmp_path / 'static.db').as_posix()}",
        collaborative_enabled=False,
        als_enabled=False,
        reranker_enabled=False,
        rate_limit_requests=1000,
    )
    return TestClient(create_app(settings=settings, catalog=catalog, providers={"gemini": MockProvider("gemini")}))


@pytest.fixture
def production_client(tmp_path, catalog):
    if not (BUNDLE / "index.html").is_file():
        reason = "React bundle not built: run `npm ci && npm run build` in frontend/"
        # CI builds the bundle first and sets this, so a build that silently
        # produced nothing fails the job instead of skipping its way to green.
        if os.environ.get("REQUIRE_FRONTEND_BUNDLE"):
            pytest.fail(reason)
        pytest.skip(reason)
    with _client(tmp_path, catalog, BUNDLE) as client:
        yield client


def test_built_react_assets_are_served_without_shadowing_api(production_client):
    page = production_client.get("/")
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-cache"
    assert 'id="root"' in page.text
    assert "/src/main.jsx" not in page.text
    assert "@vite/client" not in page.text
    assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', page.text)
    assert any(asset.endswith(".js") for asset in assets)
    assert any(asset.endswith(".css") for asset in assets)
    for asset in assets:
        response = production_client.get(asset)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert "text/html" not in response.headers["content-type"]
        assert "__animeCompassDiagnostics" not in response.text
    assert production_client.get("/api/meta").json()["count"] > 0
    assert production_client.get("/api/not-a-real-endpoint").status_code == 404
    assert production_client.get("/assets/missing.js").status_code == 404


def test_existing_api_aliases_still_work_with_react_mount(production_client):
    search = production_client.get("/api/anime/search?q=&limit=2")
    assert search.status_code == 200 and len(search.json()["results"]) == 2
    recommendations = production_client.post("/api/recommend", json={"limit": 2})
    assert recommendations.status_code == 200
    assert recommendations.json()["results"] == recommendations.json()["recommendations"]
    agent = production_client.post("/api/agent", json={"message": "Show the 2 highest-rated TV anime"})
    assert agent.status_code == 200 and agent.json()["answer"]


def test_the_bundle_is_where_the_default_setting_looks():
    assert Settings(_env_file=None).frontend_dir == BUNDLE


def test_relative_frontend_path_is_resolved_against_project():
    settings = Settings(_env_file=None, frontend_dir="frontend/dist")
    assert settings.frontend_dir == BUNDLE
    assert isinstance(settings.frontend_dir, Path)


def test_without_a_bundle_the_api_still_serves(tmp_path, catalog):
    """A source checkout has no bundle; that must not stop backend work or its tests."""
    with _client(tmp_path, catalog, tmp_path / "no-bundle-here") as client:
        assert client.get("/api/meta").json()["count"] > 0
        assert client.post("/api/recommend", json={"limit": 2}).status_code == 200
        assert client.get("/").status_code == 404
        assert client.get("/api/not-a-real-endpoint").status_code == 404


def test_the_unbuilt_react_source_is_never_what_gets_served(tmp_path, catalog):
    """`frontend/index.html` is Vite's entry page. Served as-is it would load nothing."""
    source_page = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "/src/main.jsx" in source_page
    assert Settings(_env_file=None).frontend_dir != PROJECT_ROOT / "frontend"


def test_the_image_builds_serves_and_requires_the_same_bundle_path():
    """Three files must agree, or the container boots without a UI or not at all.

    The launcher once required the legacy `frontend/index.html` while FastAPI
    served a different directory, so deleting the legacy page would have
    stopped production from starting with nothing in the Python tests to say so.
    """
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    served = re.search(r"FRONTEND_DIR=(\S+)", dockerfile)
    assert served, "Dockerfile.cloudrun must set FRONTEND_DIR"
    bundle_dir = served.group(1).rstrip("\\").strip()
    assert bundle_dir == "frontend/dist"
    assert f"/ui/dist/ ./{bundle_dir}/" in dockerfile, "the Node stage's output must land where FastAPI serves it"
    assert "COPY frontend/package.json frontend/package-lock.json ./" in dockerfile
    assert f"{bundle_dir}/index.html" in launcher, "the launcher must require the page FastAPI will serve"
    assert Settings(_env_file=None, frontend_dir=bundle_dir).frontend_dir == BUNDLE
