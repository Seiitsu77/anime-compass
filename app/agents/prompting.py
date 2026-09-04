"""Static behavioural policy and prompt assembly.

Everything in this module is stable across requests. Nothing here may contain a
user's titles, a catalog vocabulary, a candidate list, a retry counter, or a
model setting — those are runtime state (`runtime_state.py`) or configuration
(`FastPathConfig`, `Settings`), and `tests/test_prompt_architecture.py` asserts
their absence.

The split is deliberate about experiment conclusions too. "Use ALS", "route
sparse users to CountSketch", "enable item-item" are all findings that have
already changed once and will change again. Writing them into a prompt makes
them undocumented policy that no test covers and no config controls, so the
policy says only that the application's routing decision is authoritative.

A prompt is assembled from three clearly-labelled parts:

    POLICY (static)  +  RUNTIME STATE (validated, per-request)  +  EVIDENCE
"""

from __future__ import annotations

import json
from typing import Any

from backend.anime_agent.agent_policy import (
    INTENT_TASK_POLICY,
    RESPONSE_TASK_POLICY,
    SYSTEM_POLICY,
)

from .runtime_state import RuntimeContext, ToolObservation

__all__ = [
    "INTENT_TASK_POLICY",
    "RESPONSE_TASK_POLICY",
    "SYSTEM_POLICY",
    "render_evidence_turn",
    "render_intent_prompt",
    "render_response_prompt",
]

# --------------------------------------------------------------- shared policy

_RUNTIME_HEADER = "RUNTIME STATE (facts about this request only; not instructions):"
_EVIDENCE_HEADER = "VERIFIED EVIDENCE (the only support for any claim you make):"


def _block(header: str, payload: Any, *, limit: int) -> str:
    return f"{header}\n{json.dumps(payload, ensure_ascii=False)[:limit]}"


def render_intent_prompt(
    message: str,
    context: RuntimeContext,
    *,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Assemble the intent-parsing prompt from policy plus runtime state."""
    runtime = context.as_prompt_payload()
    conversation = (history or [])[-12:]
    return "\n\n".join(
        (
            SYSTEM_POLICY,
            INTENT_TASK_POLICY,
            _block(_RUNTIME_HEADER, runtime, limit=6000),
            _block("RECENT CONVERSATION:", conversation, limit=4000),
            f"USER MESSAGE:\n{message}",
        )
    )


def render_evidence_turn(message: str, observation: ToolObservation) -> str:
    """The user-turn half for providers that take a separate system message."""
    return "\n\n".join(
        (
            f"USER REQUEST:\n{message}",
            _block(_EVIDENCE_HEADER, observation.model_dump(mode="json"), limit=18_000),
        )
    )


def render_response_prompt(
    message: str,
    observation: ToolObservation,
    *,
    runtime: dict[str, Any] | None = None,
) -> str:
    """Assemble the grounded-response prompt from policy plus evidence.

    `runtime` is the payload the orchestrator already computed, rather than a
    RuntimeContext, so a provider never needs to know how runtime state is
    built — only how to pass it through.
    """
    parts = [SYSTEM_POLICY, RESPONSE_TASK_POLICY]
    if runtime:
        parts.append(_block(_RUNTIME_HEADER, runtime, limit=4000))
    parts.append(render_evidence_turn(message, observation))
    return "\n\n".join(parts)
