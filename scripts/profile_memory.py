"""Where does Anime Compass's resident memory go?

Diagnostic only. Nothing here is imported by the application, and nothing here
changes recommendation behaviour.

Two rules shape the design:

**RSS is the authority.** `sys.getsizeof` misses nested objects, and
`tracemalloc` sees only the Python heap -- not NumPy buffers, not LightGBM, not
torch. Every headline number below is process resident set size.

**Each configuration gets a fresh interpreter.** Python and the platform
allocator keep freed pages, so loading a component, dropping it, and re-reading
RSS measures nothing. Comparative experiments therefore re-invoke this script as
a subprocess and report the child's RSS.

    python scripts/profile_memory.py stages          # staged startup, one process
    python scripts/profile_memory.py imports         # per-library import cost
    python scripts/profile_memory.py artifacts       # per-artifact load cost
    python scripts/profile_memory.py catalog         # catalog decomposition
    python scripts/profile_memory.py arrays          # live ndarray inventory
    python scripts/profile_memory.py requests        # steady-state request peaks
    python scripts/profile_memory.py ab              # component A/B attribution
    python scripts/profile_memory.py all --repeat 3
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.anime_agent.process_memory import (  # noqa: E402
    current_process_rss_bytes,
    peak_process_rss_bytes,
)

MIB = 1024 * 1024


def rss() -> int:
    return current_process_rss_bytes() or 0


def mib(value: float) -> str:
    return f"{value / MIB:8.1f}"


@dataclass
class Stage:
    name: str
    rss_bytes: int
    delta_bytes: int
    cumulative_bytes: int
    load_ms: float


@dataclass
class Recorder:
    """Records RSS after each labelled step of a single process."""

    baseline: int = field(default_factory=rss)
    previous: int = 0
    stages: list[Stage] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.previous = self.baseline
        self.stages.append(Stage("Python process baseline", self.baseline, 0, 0, 0.0))

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        gc.collect()
        started = time.perf_counter()
        yield
        gc.collect()
        elapsed = (time.perf_counter() - started) * 1000
        current = rss()
        self.stages.append(Stage(name, current, current - self.previous, current - self.baseline, elapsed))
        self.previous = current

    def table(self) -> str:
        header = f"  {'Stage':<38}{'RSS MiB':>10}{'d Stage':>10}{'d Base':>10}{'Load ms':>10}"
        lines = [header, "  " + "-" * (len(header) - 2)]
        for stage in self.stages:
            lines.append(
                f"  {stage.name:<38}{mib(stage.rss_bytes):>10}{mib(stage.delta_bytes):>10}"
                f"{mib(stage.cumulative_bytes):>10}{stage.load_ms:>10.0f}"
            )
        peak = peak_process_rss_bytes()
        if peak:
            lines.append(f"\n  peak RSS this process: {peak / MIB:.1f} MiB")
        return "\n".join(lines)

    def as_json(self) -> list[dict[str, Any]]:
        return [
            {
                "stage": s.name,
                "rss_bytes": s.rss_bytes,
                "delta_bytes": s.delta_bytes,
                "cumulative_bytes": s.cumulative_bytes,
                "load_ms": round(s.load_ms, 1),
            }
            for s in self.stages
        ]


# --------------------------------------------------------------- subprocess


def run_child(experiment: str, extra: list[str] | None = None) -> dict[str, Any]:
    """Run one experiment in a fresh interpreter and return its JSON result."""
    command = [sys.executable, str(Path(__file__).resolve()), "_child", experiment, *(extra or [])]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    for line in completed.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__") :])
    return {"error": completed.stderr.strip()[-400:] or "child produced no result"}


def emit(payload: dict[str, Any]) -> None:
    print("__RESULT__" + json.dumps(payload))


# ----------------------------------------------------------------- stages


def build_staged_app() -> Recorder:
    """Walk the real lifespan order from app/main.py, recording RSS per step."""
    recorder = Recorder()

    with recorder.step("Core imports (numpy, pydantic)"):
        import numpy  # noqa: F401
        import pydantic  # noqa: F401

    with recorder.step("FastAPI + SQLAlchemy imports"):
        import fastapi  # noqa: F401
        import sqlalchemy  # noqa: F401

    with recorder.step("app.main import (all backend modules)"):
        import app.main as main_module

    with recorder.step("Settings"):
        settings = main_module.get_settings()

    with recorder.step("Full catalog loaded"):
        catalog = main_module.load_or_create_catalog(main_module.PROJECT_ROOT)

    with recorder.step("Semantic index (+ provider)"):
        semantic = main_module._load_semantic_index(settings, catalog)

    with recorder.step("Collaborative (CountSketch)"):
        collaborative = main_module._load_collaborative_index(settings, catalog)

    with recorder.step("ALS production artifact"):
        als = main_module._load_als_index(settings, catalog, quality_source=collaborative)

    with recorder.step("Reranker (features + LambdaMART)"):
        reranker = main_module._load_reranker(settings, catalog, als)

    with recorder.step("AnimeRecommender (hybrid indexes)"):
        recommender = main_module.AnimeRecommender(catalog, semantic_index=semantic, collaborative_index=collaborative)

    with recorder.step("Session repository (SQLite)"):
        sessions = main_module.SQLiteSessionRepository(
            settings.database_url, retention_days=settings.session_retention_days
        )

    with recorder.step("LLM providers"):
        providers = main_module._build_providers(settings)

    with recorder.step("Agent orchestrator"):
        agent = main_module.AgentOrchestrator(recommender, sessions, providers, settings)

    with recorder.step("EntityResolver"):
        resolver = main_module.EntityResolver(catalog)

    # Keep every object alive so nothing is reclaimed before the final reading.
    globals()["_keepalive"] = (catalog, semantic, collaborative, als, reranker, recommender, sessions, agent, resolver)
    return recorder


# ---------------------------------------------------------------- imports


IMPORT_TARGETS = (
    "numpy",
    "pydantic",
    "fastapi",
    "sqlalchemy",
    "httpx",
    "scipy",
    "lightgbm",
    "torch",
    "sentence_transformers",
)


def measure_import(name: str) -> dict[str, Any]:
    baseline = rss()
    started = time.perf_counter()
    try:
        __import__(name)
    except ImportError as exc:
        return {"module": name, "available": False, "detail": str(exc)[:80]}
    gc.collect()
    return {
        "module": name,
        "available": True,
        "rss_delta_bytes": rss() - baseline,
        "load_ms": round((time.perf_counter() - started) * 1000, 1),
    }


# -------------------------------------------------------------- artifacts


def measure_artifact(kind: str) -> dict[str, Any]:
    """Load exactly one artifact in a fresh process and report its cost."""
    import numpy as np

    baseline = rss()
    started = time.perf_counter()
    detail: dict[str, Any] = {}

    if kind == "catalog_full":
        path = PROJECT_ROOT / "data/processed/anime_catalog.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        detail = {"records": len(payload)}
    elif kind == "catalog_serving":
        path = PROJECT_ROOT / "data/processed/anime_catalog_serving.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        detail = {"records": len(payload)}
    elif kind in {"als", "collaborative", "semantic", "reranker_features"}:
        names = {
            "als": "als_production_item_factors.npz",
            "collaborative": "collaborative_embeddings.npz",
            "semantic": "semantic_embeddings.npz",
            "reranker_features": "reranker_features.npz",
        }
        path = PROJECT_ROOT / "data/processed" / names[kind]
        with np.load(path, allow_pickle=False) as handle:
            payload = {key: handle[key] for key in handle.files}
        detail = {
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype), "nbytes": int(value.nbytes)}
                for key, value in payload.items()
                if getattr(value, "nbytes", 0) > 4096
            }
        }
    elif kind == "lambdamart":
        import lightgbm as lgb

        path = PROJECT_ROOT / "data/processed/reranker_lambdamart.txt"
        import_baseline = rss()
        payload = lgb.Booster(model_file=str(path))
        detail = {"trees": payload.num_trees(), "import_included_bytes": import_baseline - baseline}
    else:
        raise SystemExit(f"unknown artifact {kind}")

    gc.collect()
    globals()["_keepalive"] = payload
    return {
        "artifact": kind,
        "path": path.name,
        "disk_bytes": path.stat().st_size,
        "rss_delta_bytes": rss() - baseline,
        "load_ms": round((time.perf_counter() - started) * 1000, 1),
        **detail,
    }


# ---------------------------------------------------------------- catalog


def decompose_catalog() -> dict[str, Any]:
    """Break the catalog's cost into raw text, parsed objects, and indexes."""
    recorder = Recorder()
    path = PROJECT_ROOT / "data/processed/anime_catalog.json"

    with recorder.step("Raw JSON text in memory"):
        text = path.read_text(encoding="utf-8")

    with recorder.step("json.loads -> Python objects"):
        catalog = json.loads(text)

    with recorder.step("Release raw text"):
        del text
        gc.collect()

    with recorder.step("id -> record map"):
        by_id = {int(item["id"]): item for item in catalog}

    with recorder.step("AnimeRecommender indexes"):
        from backend.anime_agent.recommender import AnimeRecommender

        recommender = AnimeRecommender(catalog)

    with recorder.step("EntityResolver indexes"):
        from backend.anime_agent.entities import EntityResolver

        resolver = EntityResolver(catalog)

    field_counts: dict[str, int] = {}
    for item in catalog:
        for key in item:
            field_counts[key] = field_counts.get(key, 0) + 1

    globals()["_keepalive"] = (catalog, by_id, recommender, resolver)
    return {
        "stages": recorder.as_json(),
        "records": len(catalog),
        "distinct_fields": len(field_counts),
        "disk_bytes": path.stat().st_size,
    }


# ----------------------------------------------------------------- arrays


def inventory_arrays() -> dict[str, Any]:
    """Every large ndarray reachable from a fully built application."""
    import numpy as np

    import app.main as main_module

    settings = main_module.get_settings()
    catalog = main_module.load_or_create_catalog(main_module.PROJECT_ROOT)
    semantic = main_module._load_semantic_index(settings, catalog)
    collaborative = main_module._load_collaborative_index(settings, catalog)
    als = main_module._load_als_index(settings, catalog, quality_source=collaborative)
    reranker = main_module._load_reranker(settings, catalog, als)
    recommender = main_module.AnimeRecommender(catalog, semantic_index=semantic, collaborative_index=collaborative)

    roots = {
        "als_index": als,
        "collaborative_index": collaborative,
        "semantic_index": semantic,
        "reranker.feature_space": getattr(reranker, "feature_space", None),
        "recommender": recommender,
    }
    found: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def walk(label: str, obj: Any, depth: int = 0) -> None:
        if obj is None or depth > 2 or id(obj) in seen_ids:
            return
        seen_ids.add(id(obj))
        for name, value in list(vars(obj).items()) if hasattr(obj, "__dict__") else []:
            if isinstance(value, np.ndarray) and value.nbytes > 512 * 1024:
                found.append(
                    {
                        "name": f"{label}.{name}",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "nbytes": int(value.nbytes),
                        "owns_data": value.base is None,
                        "contiguous": bool(value.flags["C_CONTIGUOUS"]),
                        "data_ptr": int(value.__array_interface__["data"][0]),
                    }
                )
            elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
                walk(f"{label}.{name}", value, depth + 1)

    for label, obj in roots.items():
        walk(label, obj)

    by_pointer: dict[int, list[str]] = {}
    for entry in found:
        by_pointer.setdefault(entry["data_ptr"], []).append(entry["name"])
    shared = {ptr: names for ptr, names in by_pointer.items() if len(names) > 1}

    globals()["_keepalive"] = (catalog, semantic, collaborative, als, reranker, recommender)
    return {
        "arrays": sorted(found, key=lambda entry: -entry["nbytes"]),
        "total_ndarray_bytes": sum(entry["nbytes"] for entry in found),
        "shared_buffers": {str(k): v for k, v in shared.items()},
        "rss_bytes": rss(),
    }


# ---------------------------------------------------------------- requests


def profile_requests() -> dict[str, Any]:
    """Steady-state and per-request peaks against a fully built app."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    profiles = {
        "small (3 liked)": [9253, 1535, 5114],
        "medium (10 liked)": [9253, 1535, 5114, 13601, 11061, 16498, 23273, 4181, 199, 457],
    }
    results: dict[str, Any] = {}
    with TestClient(create_app()) as client:
        gc.collect()
        steady = rss()
        results["steady_state_bytes"] = steady

        for label, liked in profiles.items():
            client.post("/api/recommend", json={"liked_ids": liked, "top_k": 12})
            gc.collect()
            before = rss()
            for _ in range(20):
                client.post("/api/recommend", json={"liked_ids": liked, "top_k": 12})
            gc.collect()
            after = rss()
            results[label] = {"before_bytes": before, "after_bytes": after, "growth_bytes": after - before}

        heavy = list(range(1, 200))
        client.post("/api/recommend", json={"liked_ids": heavy, "top_k": 12})
        gc.collect()
        before = rss()
        for _ in range(10):
            client.post("/api/recommend", json={"liked_ids": heavy, "top_k": 12})
        gc.collect()
        results["heavy (199 liked)"] = {"before_bytes": before, "after_bytes": rss(), "growth_bytes": rss() - before}

        gc.collect()
        before = rss()
        for _ in range(10):
            client.post("/api/recommend", json={"include_genres": ["Fantasy"], "top_k": 12})
        gc.collect()
        results["constrained hybrid"] = {"before_bytes": before, "after_bytes": rss(), "growth_bytes": rss() - before}

        # Repeat the small profile again to separate warm-up from a true ratchet.
        gc.collect()
        before = rss()
        for _ in range(60):
            client.post("/api/recommend", json={"liked_ids": profiles["small (3 liked)"], "top_k": 12})
        gc.collect()
        results["repeat x60 (leak check)"] = {
            "before_bytes": before,
            "after_bytes": rss(),
            "growth_bytes": rss() - before,
        }
        results["peak_bytes"] = peak_process_rss_bytes() or 0
    return results


# --------------------------------------------------- A/B attribution only


def ab_configuration(name: str) -> dict[str, Any]:
    """MEMORY ATTRIBUTION EXPERIMENT ONLY -- never a production configuration."""
    import app.main as main_module

    settings = main_module.get_settings()
    catalog = main_module.load_or_create_catalog(main_module.PROJECT_ROOT)
    kept: list[Any] = [catalog]

    if name != "no_semantic":
        kept.append(main_module._load_semantic_index(settings, catalog))
    collaborative = None
    if name != "no_collaborative":
        collaborative = main_module._load_collaborative_index(settings, catalog)
        kept.append(collaborative)
    als = main_module._load_als_index(settings, catalog, quality_source=collaborative)
    kept.append(als)
    if name != "no_reranker":
        kept.append(main_module._load_reranker(settings, catalog, als))
    if name != "no_hybrid":
        kept.append(
            main_module.AnimeRecommender(
                catalog,
                semantic_index=kept[1] if name != "no_semantic" else None,
                collaborative_index=collaborative,
            )
        )
    gc.collect()
    globals()["_keepalive"] = kept
    return {"configuration": name, "rss_bytes": rss(), "peak_bytes": peak_process_rss_bytes() or 0}


CHILD_EXPERIMENTS: dict[str, Callable[..., Any]] = {
    "stages": lambda: {"stages": build_staged_app().as_json(), "peak_bytes": peak_process_rss_bytes() or 0},
    "catalog": decompose_catalog,
    "arrays": inventory_arrays,
    "requests": profile_requests,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["stages", "imports", "artifacts", "catalog", "arrays", "requests", "ab", "all", "_child"],
    )
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if args.command == "_child":
        experiment = args.rest[0]
        if experiment == "import":
            emit(measure_import(args.rest[1]))
        elif experiment == "artifact":
            emit(measure_artifact(args.rest[1]))
        elif experiment == "ab":
            emit(ab_configuration(args.rest[1]))
        else:
            emit(CHILD_EXPERIMENTS[experiment]())
        return 0

    print(f"python {sys.version.split()[0]} | {sys.platform} | {os.cpu_count()} cpus")
    commands = ["stages", "imports", "artifacts", "catalog", "arrays", "requests", "ab"]
    selected = commands if args.command == "all" else [args.command]

    for command in selected:
        print(f"\n{'=' * 78}\n{command.upper()}\n{'=' * 78}")
        if command == "imports":
            for name in IMPORT_TARGETS:
                result = run_child("import", [name])
                if result.get("available"):
                    print(f"  {name:24s} {mib(result['rss_delta_bytes'])} MiB   {result['load_ms']:8.0f} ms")
                else:
                    print(f"  {name:24s} {'not installed':>18s}")
        elif command == "artifacts":
            for kind in (
                "catalog_full",
                "catalog_serving",
                "als",
                "collaborative",
                "semantic",
                "reranker_features",
                "lambdamart",
            ):
                result = run_child("artifact", [kind])
                if "error" in result:
                    print(f"  {kind:20s} ERROR {result['error'][:90]}")
                    continue
                print(
                    f"  {kind:20s} disk {mib(result['disk_bytes'])} MiB -> RSS {mib(result['rss_delta_bytes'])} MiB"
                    f"   {result['load_ms']:8.0f} ms"
                )
        elif command == "ab":
            for name in ("full", "no_semantic", "no_reranker", "no_collaborative", "no_hybrid"):
                result = run_child("ab", [name])
                print(f"  {name:20s} RSS {mib(result.get('rss_bytes', 0))} MiB")
        elif command == "stages":
            for attempt in range(args.repeat):
                result = run_child("stages")
                if "error" in result:
                    print("  ERROR", result["error"][:300])
                    break
                if attempt == 0:
                    header = f"  {'Stage':<38}{'RSS MiB':>10}{'d Stage':>10}{'d Base':>10}{'Load ms':>10}"
                    print(header)
                    print("  " + "-" * (len(header) - 2))
                    for stage in result["stages"]:
                        print(
                            f"  {stage['stage']:<38}{mib(stage['rss_bytes']):>10}"
                            f"{mib(stage['delta_bytes']):>10}{mib(stage['cumulative_bytes']):>10}"
                            f"{stage['load_ms']:>10.0f}"
                        )
                    print(f"\n  peak: {result['peak_bytes'] / MIB:.1f} MiB")
                else:
                    final = result["stages"][-1]["rss_bytes"]
                    print(f"  repeat {attempt + 1}: final RSS {final / MIB:.1f} MiB")
        else:
            result = run_child(command)
            print(json.dumps(result, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
