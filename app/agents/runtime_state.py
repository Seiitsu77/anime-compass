"""Per-request state, kept out of the static system policy.

The agent's prompt has three sources of truth and they change on different
clocks:

* **Policy** is stable across every request and lives in `prompting.py`.
* **Runtime state** — this module — is request-specific and rebuilt each turn.
* **Tool results** are verified evidence and are the only support for a claim.

Keeping them apart matters for more than tidiness. Anything written into the
static policy is, in effect, asserted for all time; a catalog genre list, a
user's liked titles, or a routing default baked into prose goes stale silently
and cannot be tested. Everything here is a validated Pydantic model instead, so
it can be asserted against, diffed between turns, and logged.

Model configuration deliberately does not appear here. Which retrieval sources
run, whether segment-aware routing is on, and whether a reranker is enabled are
decided by `FastPathConfig` and `Settings`; the agent is told the *outcome* so
it can describe what happened, never the knobs so it can second-guess them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas import AgentIntent

# Fields that pin an answer to a verified catalog relationship. They are frozen
# for the life of a request: `replan.PROTECTED_FIELDS` keeps the relaxation
# ladder away from them, and `ReplanState` refuses to carry a changed set.
REQUIRED_CONSTRAINT_FIELDS: tuple[str, ...] = (
    "required_studios",
    "required_staff",
    "required_voice_actors",
    "required_characters",
)

# Fields the relaxation ladder is allowed to drop, in the order it drops them.
OPTIONAL_CONSTRAINT_FIELDS: tuple[str, ...] = (
    "min_score",
    "max_episodes",
    "min_year",
    "max_year",
    "formats",
    "preferred_studios",
    "preferred_staff",
    "preferred_voice_actors",
    "preferred_characters",
    "exclude_genres",
    "include_genres",
)


def _present(value: Any) -> bool:
    return value not in (None, "", [], (), {})


def constraint_snapshot(intent: AgentIntent, fields: tuple[str, ...]) -> dict[str, Any]:
    """The subset of `fields` this intent actually carries."""
    return {name: getattr(intent, name) for name in fields if _present(getattr(intent, name, None))}


class ToolResultSummary(BaseModel):
    """What a tool run produced, without the payload itself.

    The agent is told how many rows came back and which titles they were, never
    the raw result dictionaries. Raw payloads in a prompt invite the model to
    quote internal fields, and the response contract forbids that.
    """

    tool_names: tuple[str, ...] = ()
    result_count: int = 0
    result_titles: tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReplanState(BaseModel):
    """Explicit bounded-replanning state.

    This exists because the state was previously implicit. A relaxed request was
    re-derived from the original user message, which reinstated the very
    constraint that had just been dropped. Carrying the state forward instead of
    recomputing it is the fix, so `relaxed_fields` is authoritative and the raw
    message is never re-parsed mid-replan.
    """

    replan_count: int = Field(default=0, ge=0)
    max_replans: int = Field(default=0, ge=0)
    required_constraints: dict[str, Any] = Field(default_factory=dict)
    optional_constraints: dict[str, Any] = Field(default_factory=dict)
    relaxed_fields: tuple[str, ...] = ()
    previous_tool_result_summary: ToolResultSummary | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def initial(cls, intent: AgentIntent, *, max_replans: int) -> ReplanState:
        return cls(
            replan_count=0,
            max_replans=max_replans,
            required_constraints=constraint_snapshot(intent, REQUIRED_CONSTRAINT_FIELDS),
            optional_constraints=constraint_snapshot(intent, OPTIONAL_CONSTRAINT_FIELDS),
            relaxed_fields=(),
            previous_tool_result_summary=None,
        )

    @property
    def exhausted(self) -> bool:
        return self.replan_count >= self.max_replans

    def after_relaxing(
        self,
        field: str,
        relaxed_intent: AgentIntent,
        *,
        summary: ToolResultSummary | None = None,
    ) -> ReplanState:
        """Advance the state by one relaxation.

        Raises if the relaxation touched a required constraint. That can only
        happen through a programming error — the ladder excludes those fields —
        but the check is here rather than only in the ladder so the invariant
        holds however the state is advanced.
        """
        still_required = constraint_snapshot(relaxed_intent, REQUIRED_CONSTRAINT_FIELDS)
        if still_required != self.required_constraints:
            raise ValueError(
                f"Relaxing {field!r} changed required constraints from "
                f"{sorted(self.required_constraints)} to {sorted(still_required)}; "
                "required entity constraints must survive replanning unchanged"
            )
        return self.model_copy(
            update={
                "replan_count": self.replan_count + 1,
                "optional_constraints": constraint_snapshot(relaxed_intent, OPTIONAL_CONSTRAINT_FIELDS),
                "relaxed_fields": (*self.relaxed_fields, field),
                "previous_tool_result_summary": summary or self.previous_tool_result_summary,
            }
        )


class RuntimeContext(BaseModel):
    """Request-specific facts handed to the model as data, not as policy.

    `selected_route`, `candidate_count`, and `reranker_enabled` describe what the
    deterministic pipeline decided. They are reported so the agent can talk about
    the result accurately; they are not settings the agent may change.
    """

    # Catalog vocabulary. Dynamic because the catalog is, and because a stale
    # genre list in a static prompt is how models start inventing genres.
    catalog_genres: tuple[str, ...] = ()
    catalog_formats: tuple[str, ...] = ()

    # Session facts.
    liked_titles: tuple[str, ...] = ()
    disliked_titles: tuple[str, ...] = ()
    known_titles: tuple[str, ...] = ()
    excluded_titles: tuple[str, ...] = ()
    excluded_genres: tuple[str, ...] = ()
    reference_titles: tuple[str, ...] = ()
    user_activity_count: int = 0

    # Constraints, split by whether they may ever be relaxed.
    required_constraints: dict[str, Any] = Field(default_factory=dict)
    preferred_constraints: dict[str, Any] = Field(default_factory=dict)

    # Decisions the deterministic application already made.
    selected_route: str | None = None
    candidate_count: int | None = None
    reranker_enabled: bool | None = None

    replan: ReplanState = Field(default_factory=ReplanState)

    model_config = ConfigDict(extra="forbid", frozen=True)

    def as_prompt_payload(self) -> dict[str, Any]:
        """The JSON block rendered into a prompt, with empty fields dropped.

        Empty values are omitted rather than sent as `[]`: an empty list reads
        to a model as a fact about the user ("they like nothing") instead of an
        absence of information.
        """
        payload = self.model_dump(mode="json", exclude_none=True)
        replan = payload.pop("replan", {})
        trimmed = {key: value for key, value in payload.items() if _present(value)}
        # Replanning state is always shown once a relaxation has happened, so
        # the model describes what was actually run rather than what was asked.
        if replan.get("relaxed_fields"):
            trimmed["replan"] = {
                key: value
                for key, value in replan.items()
                if key in {"replan_count", "max_replans", "relaxed_fields", "previous_tool_result_summary"}
                and _present(value)
            }
        return trimmed


def build_runtime_context(
    *,
    catalog_genres: list[str] | tuple[str, ...] = (),
    catalog_formats: list[str] | tuple[str, ...] = (),
    session: dict[str, Any] | None = None,
    intent: AgentIntent | None = None,
    selected_route: str | None = None,
    candidate_count: int | None = None,
    reranker_enabled: bool | None = None,
    replan: ReplanState | None = None,
) -> RuntimeContext:
    """Collect this request's facts into one validated object."""
    profile = session or {}

    def titles(key: str) -> tuple[str, ...]:
        value = profile.get(key) or []
        return tuple(str(item) for item in value if str(item).strip())

    liked = titles("liked_titles")
    watched = titles("watched_titles")
    return RuntimeContext(
        catalog_genres=tuple(catalog_genres),
        catalog_formats=tuple(catalog_formats),
        liked_titles=liked,
        disliked_titles=titles("disliked_titles"),
        known_titles=watched,
        excluded_titles=titles("excluded_titles"),
        excluded_genres=titles("excluded_genres"),
        reference_titles=tuple(intent.reference_titles) if intent else (),
        user_activity_count=len(set(liked) | set(watched)),
        required_constraints=constraint_snapshot(intent, REQUIRED_CONSTRAINT_FIELDS) if intent else {},
        preferred_constraints=constraint_snapshot(intent, OPTIONAL_CONSTRAINT_FIELDS) if intent else {},
        selected_route=selected_route,
        candidate_count=candidate_count,
        reranker_enabled=reranker_enabled,
        replan=replan or ReplanState(),
    )


class ToolObservation(BaseModel):
    """Verified tool output: the only thing an explanation may rest on."""

    mode: str = ""
    tool_calls: tuple[str, ...] = ()
    results: list[dict[str, Any]] = Field(default_factory=list)
    result_titles: tuple[str, ...] = ()
    relaxations: list[dict[str, Any]] = Field(default_factory=list)
    deterministic_fallback: str = ""

    model_config = ConfigDict(extra="forbid")

    def summary(self) -> ToolResultSummary:
        return ToolResultSummary(
            tool_names=self.tool_calls,
            result_count=len(self.results),
            result_titles=self.result_titles,
        )


def build_tool_observation(
    response: dict[str, Any],
    *,
    tool_calls: tuple[str, ...] = (),
) -> ToolObservation:
    """Extract verified evidence from a tool trace.

    Result order is preserved exactly as the ranking tools produced it. The
    response policy makes that order authoritative, so reordering here would
    quietly break the guarantee the policy states.
    """
    results: list[dict[str, Any]] = []
    titles: list[str] = []
    for step in response.get("trace", []) or []:
        payload = step.get("result")
        if not isinstance(payload, dict):
            continue
        rows = payload.get("results")
        if isinstance(rows, list):
            results.extend(row for row in rows if isinstance(row, dict))
        for title in payload.get("result_titles", []) or []:
            if str(title) not in titles:
                titles.append(str(title))
    if not titles:
        titles = [str(row.get("title")) for row in results if row.get("title")]
    return ToolObservation(
        mode=str(response.get("mode") or ""),
        tool_calls=tool_calls,
        results=results,
        result_titles=tuple(titles),
        relaxations=list(response.get("relaxations", []) or []),
        deterministic_fallback=str(response.get("answer") or ""),
    )


def tool_observation_from_verified(verified: dict[str, Any]) -> ToolObservation:
    """Adapt the orchestrator's verified-data payload into an observation.

    The orchestrator assembles this dictionary; providers consume it. Converting
    it here keeps both providers free of payload-shape knowledge, which is what
    let the two of them drift apart in the first place.
    """
    return build_tool_observation(
        {
            "mode": verified.get("mode"),
            "trace": verified.get("verified_tool_trace", []),
            "relaxations": verified.get("relaxations", []),
            "answer": verified.get("deterministic_fallback", ""),
        },
        tool_calls=tuple(
            str(call.get("tool"))
            for call in verified.get("validated_tool_calls", []) or []
            if isinstance(call, dict) and call.get("tool")
        ),
    )
