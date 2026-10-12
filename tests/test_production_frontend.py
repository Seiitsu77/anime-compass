from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, Settings
from app.main import create_app
from tests.test_api import MockProvider


@pytest.fixture
def production_client(tmp_path, catalog):
    bundle = PROJECT_ROOT / "frontend-react" / "dist"
    assert (bundle / "index.html").is_file(), "Run npm ci and npm run build in frontend-react first"
    settings = Settings(
        _env_file=None,
        frontend_dir=bundle,
        database_url=f"sqlite:///{(tmp_path / 'static.db').as_posix()}",
        collaborative_enabled=False,
        als_enabled=False,
        reranker_enabled=False,
        rate_limit_requests=1000,
    )
    with TestClient(
        create_app(settings=settings, catalog=catalog, providers={"gemini": MockProvider("gemini")})
    ) as client:
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


def test_relative_frontend_path_is_resolved_against_project():
    settings = Settings(_env_file=None, frontend_dir="frontend-react/dist")
    assert settings.frontend_dir == PROJECT_ROOT / "frontend-react" / "dist"
    assert isinstance(settings.frontend_dir, Path)
