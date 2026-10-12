"""Check the built production image without credentials or production traffic."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def request(base: str, path: str, body: dict[str, Any] | None = None, *, method: str | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(req, timeout=40)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read().decode()
        content_type = response.headers.get("Content-Type", "")
        return response.status, json.loads(raw) if "application/json" in content_type else raw, content_type


def verify(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            status, ready, _ = request(base, "/api/ready")
            if status == 200:
                break
            if ready.get("error", {}).get("code") == "initialization_failed":
                raise RuntimeError("Production image initialization failed")
        except (urllib.error.URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Image did not become ready within the startup-probe budget")
        time.sleep(1)
    print("PASS readiness", ready.get("initialization", {}).get("elapsed_ms"), "ms", flush=True)
    status, html, _ = request(base, "/")
    assert status == 200 and 'id="root"' in html and "/src/main.jsx" not in html
    assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
    assert any(asset.endswith(".js") for asset in assets) and any(asset.endswith(".css") for asset in assets)
    for asset in assets:
        status, source, content_type = request(base, asset)
        assert status == 200 and "text/html" not in content_type
        assert "__animeCompassDiagnostics" not in source
    assert request(base, "/api/not-a-real-endpoint")[0] == 404
    print("PASS built React assets and API routing", flush=True)
    status, health, _ = request(base, "/api/health")
    assert status == 200 and health["catalog"]["count"] > 0
    assert health["ranking"]["retriever"] == "als" and health["ranking"]["reranker_active"]
    assert health["components"]["semantic_embeddings"]["status"] == "healthy"
    assert health["ranking"]["pagination"]["endpoint"] == "/api/recommend/page"
    print("PASS ALS, LambdaMART, semantic index and pagination capability", flush=True)
    liked = [1, 5, 21, 30, 199]
    snapshot = {"liked_ids": liked, "excluded_ids": [20], "limit": 18}
    status, first, _ = request(base, "/api/recommend/page", snapshot)
    assert status == 200 and first["results"] and first["sequence"]["has_more"]
    cursor = first["sequence"]["next_cursor"]
    status, second, _ = request(base, "/api/recommend/page", {**snapshot, "cursor": cursor})
    assert status == 200 and second["results"]
    first_ids = {item["id"] for item in first["results"]}
    second_ids = {item["id"] for item in second["results"]}
    assert len(first_ids) == len(first["results"]) and len(second_ids) == len(second["results"])
    assert not first_ids & second_ids and not (first_ids | second_ids) & {*liked, 20}
    status, resumed, _ = request(base, "/api/recommend/page", {**snapshot, "cursor": cursor})
    assert status == 200 and [item["id"] for item in resumed["results"]] == [item["id"] for item in second["results"]]
    assert request(base, "/api/recommend/page", {**snapshot, "liked_ids": [2, 4], "cursor": cursor})[0] == 409
    assert request(base, "/api/recommend/page", {**snapshot, "cursor": "not-a-cursor"})[0] == 422
    print("PASS cursor paging, exclusions, replay and stale/malformed cursors", flush=True)
    assert request(base, "/api/recommend", {**snapshot, "limit": 3})[0] == 200
    assert request(base, "/api/search", {"query": "", "top_k": 2})[0] == 200
    assert request(base, "/api/anime/search?limit=2")[0] == 200
    session = "production-check-" + uuid.uuid4().hex
    assert (
        request(
            base, "/api/session/preferences", {"session_id": session, "replace": True, "liked_titles": ["Cowboy Bebop"]}
        )[0]
        == 200
    )
    status, profile, _ = request(base, "/api/session/" + session)
    assert status == 200 and profile["profile"]["liked_titles"] == ["Cowboy Bebop"]
    assert request(base, "/api/session/" + session, method="DELETE")[0] == 200
    print("PASS existing recommendation, search and preference API contracts", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-hosted", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    server = None
    try:
        if args.self_hosted:
            # The test container has no credentials and never calls a text model.
            environment = {**os.environ, "GEMINI_API_KEY": ""}
            server = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8080"],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
            )
        verify(args.base_url.rstrip("/"), args.timeout)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
