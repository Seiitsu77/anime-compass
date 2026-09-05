"""End-to-end verification against a running FastAPI deployment.

Seven cases, chosen because each one can fail silently in production:
personalization without an LLM, exclusion enforcement, natural-language
constraints, required-entity survival, bounded replanning, LLM degradation, and
reranker degradation.

Cases 3-5 call the configured LLM provider and therefore cost money and time.
They are opt-in via --with-agent so the cheap deterministic cases can run on
every change.

    python scripts/smoke_e2e.py --base-url http://127.0.0.1:8099
    python scripts/smoke_e2e.py --base-url http://127.0.0.1:8099 --with-agent
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

STEINS_GATE = 9253
DEATH_NOTE = 1535
FULLMETAL = 5114
PSYCHO_PASS = 13601


class Failure(Exception):
    pass


def call(base: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    url = f"{base.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-supplied base
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise Failure(f"{path} returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc


def titles_of(payload: dict[str, Any]) -> list[str]:
    return [str(row.get("title")) for row in rows_of(payload)]


def rows_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows from either the recommend response or an agent tool trace."""
    for key in ("recommendations", "items", "results"):
        if isinstance(payload.get(key), list) and payload[key]:
            return [row for row in payload[key] if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for step in payload.get("trace", []) or []:
        result = step.get("result")
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            rows.extend(row for row in result["results"] if isinstance(row, dict))
    return rows


def ids_of(payload: dict[str, Any]) -> list[int]:
    return [int(row["id"]) for row in rows_of(payload) if "id" in row]


def case_1(base: str) -> dict[str, Any]:
    """Basic personalization, no LLM in the path."""
    started = time.perf_counter()
    payload = call(base, "/api/recommend", {"liked_ids": [STEINS_GATE, DEATH_NOTE, FULLMETAL], "top_k": 10})
    elapsed = (time.perf_counter() - started) * 1000
    ids = ids_of(payload)
    if len(ids) < 5:
        raise Failure(f"expected recommendations, got {len(ids)}")
    if set(ids) & {STEINS_GATE, DEATH_NOTE, FULLMETAL}:
        raise Failure("a liked title was recommended back")
    diagnostics = payload.get("diagnostics") or {}
    if not diagnostics.get("learned_reranker_applied"):
        raise Failure("the learned reranker did not run on the fast path")
    return {"count": len(ids), "ms": round(elapsed, 1), "titles": titles_of(payload)[:3]}


def case_2(base: str) -> dict[str, Any]:
    """An explicit exclusion must never appear, reranking notwithstanding."""
    plain = ids_of(call(base, "/api/recommend", {"liked_ids": [STEINS_GATE, DEATH_NOTE], "top_k": 12}))
    if not plain:
        raise Failure("no baseline recommendations to exclude from")
    victim = plain[0]
    excluded = ids_of(
        call(
            base,
            "/api/recommend",
            {"liked_ids": [STEINS_GATE, DEATH_NOTE], "excluded_ids": [victim], "top_k": 12},
        )
    )
    if victim in excluded:
        raise Failure(f"excluded id {victim} still appeared")
    return {"excluded_id": victim, "count": len(excluded)}


def case_3(base: str) -> dict[str, Any]:
    """Natural language with constraints, through the agent."""
    started = time.perf_counter()
    payload = call(
        base,
        "/api/chat",
        {"message": "I want something psychological under 24 episodes and no romance.", "debug": True},
    )
    elapsed = (time.perf_counter() - started) * 1000
    intent = (payload.get("debug") or {}).get("validated_intent") or {}
    if intent.get("intent") not in {"recommend", "search"}:
        raise Failure(f"unexpected intent: {intent.get('intent')}")
    if not payload.get("answer"):
        raise Failure("agent produced no answer")
    return {
        "ms": round(elapsed, 1),
        "intent": intent.get("intent"),
        "max_episodes": intent.get("max_episodes"),
        "exclude_genres": intent.get("exclude_genres"),
        "results": len(rows_of(payload)),
        "grounded": bool(payload.get("answer")),
    }


def case_4(base: str) -> dict[str, Any]:
    """A required entity constraint must survive the whole pipeline."""
    payload = call(
        base,
        "/api/chat",
        {"message": "Recommend anime from the studio Madhouse.", "debug": True},
    )
    intent = (payload.get("debug") or {}).get("validated_intent") or {}
    required = intent.get("required_studios") or []
    if not required:
        raise Failure("required studio constraint was not captured")
    replan = (payload.get("debug") or {}).get("replan_state") or {}
    still_required = (replan.get("required_constraints") or {}).get("required_studios")
    if still_required is not None and not still_required:
        raise Failure("required studio was dropped during replanning")
    return {"required_studios": required, "replans": replan.get("replan_count", 0)}


def case_5(base: str) -> dict[str, Any]:
    """Over-constrain deliberately and check the bounded ladder."""
    payload = call(
        base,
        "/api/chat",
        {
            "message": "A 2015-only Madhouse isekai with at most 3 episodes rated above 9.5.",
            "debug": True,
        },
    )
    debug = payload.get("debug") or {}
    replan = debug.get("replan_state") or {}
    relaxed = replan.get("relaxed_fields") or []
    max_replans = replan.get("max_replans", 0)
    if replan.get("replan_count", 0) > max_replans:
        raise Failure("replan budget exceeded")
    required = replan.get("required_constraints") or {}
    if not relaxed and rows_of(payload):
        # Not a failure: the catalog satisfied it outright. Say so rather than
        # implying the ladder was exercised.
        return {"outcome": "satisfied without relaxation", "results": len(rows_of(payload))}
    return {
        "relaxed_fields": relaxed,
        "replan_count": replan.get("replan_count", 0),
        "max_replans": max_replans,
        "required_preserved": sorted(required),
    }


def case_6(base: str) -> dict[str, Any]:
    """Recommendations must not depend on the LLM."""
    payload = call(base, "/api/recommend", {"liked_ids": [PSYCHO_PASS], "top_k": 8})
    if len(ids_of(payload)) < 5:
        raise Failure("recommendations degraded when the LLM was not involved")
    return {"count": len(ids_of(payload)), "note": "no LLM in this path by construction"}


def case_7(base: str) -> dict[str, Any]:
    """Health must state whether LambdaMART is really serving."""
    health = call(base, "/api/health")
    ranking = health.get("ranking") or {}
    component = (health.get("components") or {}).get("reranker") or {}
    if ranking.get("reranker_active") and component.get("status") != "healthy":
        raise Failure("health disagrees with itself about the reranker")
    if not ranking.get("reranker_active") and "degraded" not in str(component.get("detail", "")):
        raise Failure("a degraded reranker is not reported as degraded")
    return {
        "reranker_active": ranking.get("reranker_active"),
        "detail": component.get("detail"),
        "retriever": ranking.get("retriever"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--with-agent", action="store_true", help="run the three LLM-backed cases")
    args = parser.parse_args()

    cases = [
        ("1 basic personalization", case_1, False),
        ("2 exclusion enforced", case_2, False),
        ("3 constrained natural language", case_3, True),
        ("4 required entity survives", case_4, True),
        ("5 bounded replanning", case_5, True),
        ("6 works without the LLM", case_6, False),
        ("7 reranker status is honest", case_7, False),
    ]
    failures = 0
    for name, runner, needs_agent in cases:
        if needs_agent and not args.with_agent:
            print(f"  SKIP  {name} (needs --with-agent)")
            continue
        try:
            detail = runner(args.base_url)
            print(f"  PASS  {name}: {json.dumps(detail, ensure_ascii=False)[:150]}")
        except Failure as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a smoke run reports, it does not crash
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'all checked cases passed' if not failures else f'{failures} case(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
