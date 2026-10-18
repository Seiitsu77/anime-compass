"""Check that every title the agent returns satisfies the request's hard constraints.

Each case in data/evaluation/agent_constraints.json is a short conversation.
Every turn states what its results must be: in a franchise, of a format, from a
year range, by a studio or person, outside an excluded franchise. The runner
sends the conversation to the agent, looks every returned ID up in the catalog,
and checks it with an oracle written independently of the code under test --
franchise membership here is a whole-word match of the case's own names in a
title or alias, not a call into title_family.py.

A turn fails when any returned row violates a constraint, when a franchise
restriction is invented for a similarity request, or when fewer rows come back
than the catalog is known to hold. It never fails for returning fewer rows than
asked for: a short honest list is the correct answer to a narrow question.

The agent is reached either in-process, built the way the server builds it, or
over HTTP against a running deployment such as a zero-traffic candidate:

    python scripts/evaluate_agent_constraints.py --in-process --catalog <anime_catalog.json>
    python scripts/evaluate_agent_constraints.py --base-url https://candidate---service.run.app

Both call the configured language model when one is set up, which costs money
and depends on a live API. Neither runs in routine CI; the deterministic
versions of these cases live in tests/test_constraint_regressions.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "data" / "evaluation" / "agent_constraints.json"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
RESULT_TOOLS = frozenset({"recommend_anime", "rank_catalog", "search_anime"})

# Every key a turn's "expect" block may use. Unknown keys are rejected, so a
# misspelt constraint cannot pass by being silently ignored.
EXPECT_KEYS = frozenset(
    {
        "family_any",
        "family_prefix_any",
        "formats",
        "min_year",
        "max_year",
        "studio",
        "staff",
        "voice_actor",
        "character",
        "excluded_families",
        "max_episodes",
        "min_results",
        "max_results",
        "allow_empty",
        "no_results_expected",
        "no_family_constraint",
        "answer_contains",
    }
)

Chat = Callable[[str, list[dict[str, str]]], dict[str, Any]]


def words(text: Any) -> str:
    """Lower-case words joined by single spaces and padded, for whole-word tests."""
    return " " + " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold())) + " "


def person_keys(name: str) -> set[str]:
    """'Hanazawa, Kana' matches both 'hanazawa kana' and 'kana hanazawa'."""
    parts = [part.strip() for part in str(name or "").split(",") if part.strip()]
    keys = {words(name).strip()}
    if len(parts) == 2:
        keys.add(words(f"{parts[1]} {parts[0]}").strip())
    return keys


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    names = [case["name"] for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("case names must be unique")
    for case in cases:
        if not case.get("turns"):
            raise ValueError(f"{case['name']}: no turns")
        for turn in case["turns"]:
            unknown = set(turn.get("expect", {})) - EXPECT_KEYS
            if unknown:
                raise ValueError(f"{case['name']}: unknown expectation keys {sorted(unknown)}")
            if not str(turn.get("message") or "").strip():
                raise ValueError(f"{case['name']}: empty message")
    return cases


def violations(item: dict[str, Any] | None, expect: dict[str, Any]) -> list[str]:
    """Every hard constraint in `expect` that this catalog row breaks."""
    if item is None:
        return ["id not in catalog"]
    problems: list[str] = []
    names = [item.get("title") or "", *(item.get("aliases") or [])]
    blob = " | ".join(words(name) for name in names)
    family = expect.get("family_any")
    if family and not any(f" {words(name).strip()} " in blob for name in family):
        problems.append(f"not in family {family}")
    prefixes = expect.get("family_prefix_any")
    if prefixes and not any(words(name).startswith(words(prefix)) for name in names for prefix in prefixes):
        problems.append(f"no title starts with {prefixes}")
    if expect.get("formats") and item.get("type") not in expect["formats"]:
        problems.append(f"type {item.get('type')} not in {expect['formats']}")
    year = item.get("start_year")
    if expect.get("min_year") is not None and (year is None or year < expect["min_year"]):
        problems.append(f"start_year {year} < {expect['min_year']}")
    if expect.get("max_year") is not None and (year is None or year > expect["max_year"]):
        problems.append(f"start_year {year} > {expect['max_year']}")
    if expect.get("studio") and not any(
        words(expect["studio"]) == words(studio) for studio in item.get("studios") or []
    ):
        problems.append(f"studios {item.get('studios')} lack {expect['studio']}")
    if expect.get("staff"):
        staff = item.get("staff_relationships") or item.get("staff") or []
        keys: set[str] = set().union(set(), *(person_keys(p.get("name")) for p in staff if p.get("name")))
        if words(expect["staff"]).strip() not in keys:
            problems.append(f"staff lack {expect['staff']}")
    if expect.get("voice_actor"):
        cast = [p.get("name") for p in item.get("voice_actors") or [] if p.get("name")]
        cast += [r.get("voice_actor") for r in item.get("voice_actor_roles") or [] if r.get("voice_actor")]
        keys = set().union(set(), *(person_keys(str(name)) for name in cast))
        if words(expect["voice_actor"]).strip() not in keys:
            problems.append(f"voice actors lack {expect['voice_actor']}")
    if expect.get("character"):
        keys = set().union(set(), *(person_keys(name) for name in item.get("character_names") or [] if name))
        if words(expect["character"]).strip() not in keys:
            problems.append(f"characters lack {expect['character']}")
    for excluded in expect.get("excluded_families") or []:
        if f" {words(excluded).strip()} " in blob:
            problems.append(f"belongs to excluded family {excluded}")
    if expect.get("max_episodes") is not None:
        episodes = item.get("episodes")
        if episodes is None or episodes > expect["max_episodes"]:
            problems.append(f"episodes {episodes} > {expect['max_episodes']}")
    return problems


def returned_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in response.get("trace") or []:
        result = step.get("result") if isinstance(step, dict) else None
        if step.get("tool") in RESULT_TOOLS and isinstance(result, dict):
            rows.extend(row for row in result.get("results") or [] if isinstance(row, dict))
    return rows


def check_turn(
    response: dict[str, Any],
    expect: dict[str, Any],
    by_id: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Check one agent response; return the checked rows and any problems."""
    checked: list[dict[str, Any]] = []
    for row in returned_rows(response):
        try:
            anime_id: int | None = int(str(row.get("id")))
        except (TypeError, ValueError):
            anime_id = None
        item = by_id.get(anime_id) if anime_id is not None else None
        checked.append(
            {
                "id": anime_id,
                "title": item.get("title") if item else row.get("title"),
                "type": item.get("type") if item else None,
                "start_year": item.get("start_year") if item else None,
                "violations": violations(item, expect),
            }
        )
    problems: list[str] = []
    bad = [row for row in checked if row["violations"]]
    if bad:
        problems.append(f"{len(bad)} of {len(checked)} results violate a hard constraint")
    if expect.get("no_results_expected"):
        if checked:
            problems.append(f"expected no results, got {len(checked)}")
    elif not checked and not expect.get("allow_empty"):
        problems.append("no results")
    if expect.get("min_results") and len(checked) < expect["min_results"]:
        problems.append(f"{len(checked)} results; the catalog holds at least {expect['min_results']}")
    if expect.get("max_results") is not None and len(checked) > expect["max_results"]:
        problems.append(f"{len(checked)} results; at most {expect['max_results']} were asked for")
    intent = (response.get("debug") or {}).get("validated_intent") or response.get("validated_intent") or {}
    if expect.get("no_family_constraint") and intent.get("required_title_families"):
        problems.append(f"a reference was turned into a franchise restriction: {intent['required_title_families']}")
    answer = str(response.get("answer") or "").casefold()
    for text in expect.get("answer_contains") or []:
        if text.casefold() not in answer:
            problems.append(f"answer lacks {text!r}")
    return checked, problems


def http_chat(base_url: str, timeout: float) -> Chat:
    def chat(message: str, history: list[dict[str, str]]) -> dict[str, Any]:
        body = json.dumps({"message": message, "history": history, "debug": True}).encode()
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-supplied URL
                return dict(json.loads(response.read().decode()))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"/api/chat returned HTTP {exc.code}: {exc.read()[:200]!r}") from exc

    return chat


def http_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - operator-supplied URL
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError):
        return 0


def wait_until_ready(status: Callable[[], int], timeout: float) -> None:
    """Block until /api/ready answers 200; a warming server answers 503 to chat."""
    deadline = time.monotonic() + timeout
    while (code := status()) != 200:
        if time.monotonic() > deadline:
            raise SystemExit(f"/api/ready still answered {code} after {timeout:.0f}s")
        time.sleep(2)


def run(
    cases: list[dict[str, Any]],
    chat: Chat,
    by_id: dict[int, dict[str, Any]],
    *,
    require_provider: str | None,
    retries: int,
    delay: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        history: list[dict[str, str]] = []
        for index, turn in enumerate(case["turns"], start=1):
            expect = turn.get("expect", {})
            attempts = 0
            while True:
                attempts += 1
                started = time.perf_counter()
                try:
                    response, error = chat(turn["message"], list(history)), None
                except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                    response, error = {}, f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - started
                parser_mode = response.get("parser_mode") or (response.get("debug") or {}).get("parser_mode")
                live = require_provider is None or parser_mode == require_provider
                if (live and error is None) or attempts > retries:
                    break
                time.sleep(delay * attempts)
            checked, problems = check_turn(response, expect, by_id)
            if error:
                problems.insert(0, error)
            if require_provider and parser_mode != require_provider:
                problems.append(f"parsed by {parser_mode}, not {require_provider}")
            intent = (response.get("debug") or {}).get("validated_intent") or {}
            record = {
                "case": case["name"],
                "turn": index,
                "message": turn["message"],
                "parser_mode": parser_mode,
                "attempts": attempts,
                "elapsed_s": round(elapsed, 1),
                "intent": {
                    key: intent.get(key)
                    for key in (
                        "intent",
                        "required_title_families",
                        "excluded_title_families",
                        "formats",
                        "min_year",
                        "max_year",
                        "required_studios",
                        "required_staff",
                        "required_voice_actors",
                        "required_characters",
                        "reference_titles",
                    )
                },
                "relaxations": response.get("relaxations"),
                "results": checked,
                "problems": problems,
                "answer": response.get("answer", ""),
            }
            records.append(record)
            print(
                f"{'FAIL' if problems else 'ok  '} [{case['name']} t{index}] parser={parser_mode} {record['elapsed_s']}s"
            )
            print(
                f"     {len(checked)} results: "
                + "; ".join(
                    f"{row['title']} [{row['type']} {row['start_year']}]"
                    + (" !! " + ", ".join(row["violations"]) if row["violations"] else "")
                    for row in checked[:10]
                )
            )
            for problem in problems:
                print(f"     PROBLEM: {problem}")
            history.append({"role": "user", "content": turn["message"]})
            history.append({"role": "assistant", "content": str(response.get("answer") or "")[:3900]})
            time.sleep(delay)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--base-url", help="a running deployment, e.g. a zero-traffic candidate URL")
    target.add_argument("--in-process", action="store_true", help="build the app in this process")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="catalog the oracle checks IDs against")
    parser.add_argument("--only", action="append", default=[], help="run only cases whose name contains this text")
    parser.add_argument("--require-provider", help="fail any turn not parsed by this provider, e.g. gemini")
    parser.add_argument("--retries", type=int, default=2, help="retries for a turn the provider did not answer")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ready-timeout", type=float, default=900.0, help="seconds to wait for /api/ready")
    parser.add_argument("--out", type=Path, help="write the full per-turn record as JSON")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.only:
        cases = [case for case in cases if any(text.casefold() in case["name"].casefold() for text in args.only)]
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    by_id = {int(item["id"]): item for item in catalog}

    if args.base_url:
        wait_until_ready(lambda: http_status(f"{args.base_url.rstrip('/')}/api/ready"), args.ready_timeout)
        records = run(
            cases,
            http_chat(args.base_url, args.timeout),
            by_id,
            require_provider=args.require_provider,
            retries=args.retries,
            delay=args.delay,
        )
    else:
        sys.path.insert(0, str(PROJECT_ROOT))
        from fastapi.testclient import TestClient

        from app.core.config import Settings
        from app.main import create_app

        database = Path(tempfile.mkdtemp(prefix="agent_constraints_")) / "sessions.db"
        settings = Settings(database_url=f"sqlite:///{database.as_posix()}", rate_limit_requests=10_000)
        with TestClient(create_app(settings=settings, catalog=catalog)) as client:
            wait_until_ready(lambda: client.get("/api/ready").status_code, args.ready_timeout)

            def chat(message: str, history: list[dict[str, str]]) -> dict[str, Any]:
                response = client.post("/api/chat", json={"message": message, "history": history, "debug": True})
                if response.status_code != 200:
                    raise RuntimeError(f"/api/chat returned HTTP {response.status_code}: {response.text[:200]}")
                return dict(response.json())

            records = run(
                cases,
                chat,
                by_id,
                require_provider=args.require_provider,
                retries=args.retries,
                delay=args.delay,
            )

    if args.out:
        args.out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = sum(1 for record in records if record["problems"])
    print(f"\n{len(records) - failed}/{len(records)} turns satisfied every hard constraint")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
