"""Portfolio demo for the anime recommendation system.

Streamlit Community Cloud looks for this file at the repository root.

The demo talks to the recommendation core directly rather than over HTTP. Both
would run in the same process on Community Cloud, so a FastAPI hop would add a
deployment dependency and a failure mode without adding anything. The FastAPI
app is untouched and still works; see docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from backend.anime_agent.artifact_bootstrap import bootstrap_from_environment
from backend.anime_agent.showcase import ShowcaseService, load_showcase_service

PROJECT_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_ARTIFACT = PROJECT_ROOT / "data" / "processed" / "als_production_item_factors.npz"

st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STYLE = """
<style>
  .block-container { max-width: 1180px; padding-top: 2.2rem; }
  .hero-title { font-size: 2.4rem; font-weight: 700; margin-bottom: .3rem; line-height: 1.15; }
  .hero-sub { font-size: 1.05rem; opacity: .78; margin-bottom: 1.6rem; }
  .metric-value { font-size: 2.0rem; font-weight: 700; line-height: 1.1; }
  .metric-label { font-size: .84rem; opacity: .72; margin-top: .15rem; }
  .rec-title { font-weight: 650; font-size: 1.02rem; margin-bottom: .15rem; }
  .rec-meta { font-size: .82rem; opacity: .70; margin-bottom: .4rem; }
  .rec-why { font-size: .86rem; opacity: .92; }
  .pill { display:inline-block; padding:.12rem .5rem; margin:.1rem .22rem .1rem 0;
          border-radius:999px; font-size:.72rem; background:rgba(128,128,128,.16); }
  .poster-fallback { display:flex; align-items:center; justify-content:center;
          height:190px; border-radius:8px; background:rgba(128,128,128,.12);
          font-size:.75rem; opacity:.6; text-align:center; padding:.5rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading catalog and model…")
def get_service() -> ShowcaseService:
    """Load catalog and model once per container.

    cache_resource rather than cache_data: the loaded index holds NumPy arrays
    and a cached Gram matrix that must not be copied per session.
    """
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    bootstrap = bootstrap_from_environment(DEFAULT_ARTIFACT)
    service = load_showcase_service(
        catalog,
        bootstrap.path,
        expected_sha256=os.environ.get("ALS_EXPECTED_SHA256") or None,
        expected_catalog_ids_sha256=os.environ.get("ALS_EXPECTED_CATALOG_IDS_SHA256") or None,
        require_production=os.environ.get("ALS_EXPECTED_ROLE", "production") == "production",
    )
    service.health.error = service.health.error or (None if bootstrap.usable else bootstrap.detail)
    return service


def init_state() -> None:
    st.session_state.setdefault("liked", [])
    st.session_state.setdefault("disliked", [])
    st.session_state.setdefault("results", None)


def hero() -> None:
    st.markdown('<div class="hero-title">Anime Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">A production-style personalized recommender, '
        "evaluated on 57M+ historical ratings.</div>",
        unsafe_allow_html=True,
    )
    left, middle, right = st.columns(3)
    for column, value, label in (
        (left, "+42.6%", "NDCG@10 vs the previous production architecture"),
        (middle, "~2 ms", "Fast-path recommendation latency (p50)"),
        (right, "30.9M", "Positive interactions in the production model"),
    ):
        with column:
            st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(
        "Pick a few titles you like. The system builds a preference profile with a tuned "
        "implicit-feedback ALS model and ranks the full ~18,000-title catalog against it. "
        "Requests with explicit constraints route to a richer pipeline instead."
    )


def model_unavailable(service: ShowcaseService) -> None:
    st.error(
        "**The production ALS model is not available in this deployment**, so recommendations "
        "are disabled. The demo will not fall back to a weaker model and present it as the "
        "benchmarked one."
    )
    with st.expander("Details for whoever is deploying this"):
        st.write(service.health.error or "No further detail was recorded.")
        st.markdown(
            "Set `ALS_ARTIFACT_URL` to an https location serving "
            "`als_production_item_factors.npz`, plus `ALS_EXPECTED_SHA256` to pin it. "
            "See the Deployment section of the README."
        )


def picker(service: ShowcaseService) -> None:
    st.subheader("1. Tell it what you like")

    profiles = service.example_profiles()
    if profiles:
        st.caption("Not sure where to start? Load an example profile:")
        columns = st.columns(len(profiles))
        for column, (name, (description, ids)) in zip(columns, profiles.items(), strict=False):
            with column:
                if st.button(name, use_container_width=True, help=description):
                    st.session_state.liked = list(ids)
                    st.session_state.results = None
                    st.rerun()

    query = st.text_input("Search the catalog", placeholder="e.g. Steins;Gate, Cowboy Bebop, Monster")
    if query:
        matches = service.search(query, limit=8)
        if not matches:
            st.caption("No titles matched that search.")
        for item in matches:
            anime_id = int(item["id"])
            row, button = st.columns([5, 1])
            year = item.get("start_year") or "—"
            row.markdown(
                f"**{item['title']}**  \n<span class='rec-meta'>{year} · {item.get('type') or '—'}</span>",
                unsafe_allow_html=True,
            )
            if button.button("Like", key=f"like-{anime_id}", use_container_width=True):
                if anime_id not in st.session_state.liked:
                    st.session_state.liked.append(anime_id)
                    st.session_state.results = None
                    st.rerun()

    if st.session_state.liked:
        st.write("")
        st.caption("Your profile")
        for anime_id in list(st.session_state.liked):
            row, button = st.columns([5, 1])
            cold = " · not in the trained model" if service.is_cold_start(anime_id) else ""
            row.markdown(f"✅ {service.title_of(anime_id)}<span class='rec-meta'>{cold}</span>", unsafe_allow_html=True)
            if button.button("Remove", key=f"drop-{anime_id}", use_container_width=True):
                st.session_state.liked.remove(anime_id)
                st.session_state.results = None
                st.rerun()
    else:
        st.info("Add at least one title, or load an example profile above.")


def controls(service: ShowcaseService) -> tuple[int, str]:
    with st.expander("Optional: exclusions and a natural-language request"):
        exclude_query = st.text_input("Exclude a title", placeholder="Search a title to exclude")
        if exclude_query:
            for item in service.search(exclude_query, limit=5):
                anime_id = int(item["id"])
                row, button = st.columns([5, 1])
                row.write(item["title"])
                if button.button("Exclude", key=f"ex-{anime_id}", use_container_width=True):
                    if anime_id not in st.session_state.disliked:
                        st.session_state.disliked.append(anime_id)
                        st.session_state.results = None
                        st.rerun()
        if st.session_state.disliked:
            st.caption("Excluded: " + ", ".join(service.title_of(i) for i in st.session_state.disliked))
            if st.button("Clear exclusions"):
                st.session_state.disliked = []
                st.rerun()

        llm_ready = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("OLLAMA_BASE_URL"))
        free_text = st.text_input(
            "Anything else you're looking for?",
            placeholder="e.g. something dark and psychological, under 24 episodes",
            disabled=not llm_ready,
        )
        if not llm_ready:
            st.caption(
                "Natural-language constraint parsing needs an LLM provider, which this deployment "
                "has not configured. Profile-based recommendations below work without it."
            )
            free_text = ""
        count = st.slider("How many recommendations", 4, 24, 12, step=4)
    return count, free_text


def poster(item: Any) -> None:
    if item.image_url:
        st.image(item.image_url, use_container_width=True)
    else:
        st.markdown('<div class="poster-fallback">No poster<br/>in the CC0 dataset</div>', unsafe_allow_html=True)


def render_results(service: ShowcaseService) -> None:
    result = st.session_state.results
    if result is None:
        return
    if not result.items:
        st.warning("No recommendations came back for that profile. Try adding another title.")
        return

    st.subheader("2. Your recommendations")
    st.caption(
        f"Ranked against all {service.health.catalog_items:,} catalog titles "
        f"in {result.latency_ms:.0f} ms from a {result.candidate_pool_size}-candidate pool."
    )
    for start in range(0, len(result.items), 3):
        for column, item in zip(st.columns(3), result.items[start : start + 3], strict=False):
            with column, st.container(border=True):
                poster(item)
                st.markdown(f'<div class="rec-title">{item.title}</div>', unsafe_allow_html=True)
                bits = [str(item.year or "—"), item.media_type or "—"]
                if item.episodes:
                    bits.append(f"{item.episodes} eps")
                if item.score:
                    bits.append(f"★ {item.score:.2f}")
                st.markdown(f'<div class="rec-meta">{" · ".join(bits)}</div>', unsafe_allow_html=True)
                if item.genres:
                    st.markdown(
                        "".join(f'<span class="pill">{g}</span>' for g in item.genres),
                        unsafe_allow_html=True,
                    )
                st.markdown(f'<div class="rec-why">{item.explanation}</div>', unsafe_allow_html=True)


def how_it_works() -> None:
    with st.expander("How it works"):
        st.markdown(
            """
```text
                 User preferences
                        |
                        v
                  Request router
                    /         \\
           simple request    constrained request
                 |                    |
                 v                    v
          Production ALS          Rich Hybrid
            fast path           constraint path
                 \\                    /
                  v                  v
                    Recommendations
                          |
                          v
                     Explanations
```
"""
        )
        st.markdown(
            "- **ALS is the default** personalized retrieval model, trained on 30.9M positive interactions.\n"
            "- **The rich Hybrid runs only when a request carries explicit constraints** — a studio, a "
            "voice actor, a year window — that need exact catalog joins.\n"
            "- **The architecture was chosen by controlled offline experiments**, not by preference.\n"
            "- **The production model trains on all available history**; a separate artifact, which "
            "withholds held-out interactions, is used for evaluation so the metrics stay leakage-free.\n"
            "- **Explanations are grounded** in learned item-factor similarity and real genre overlap, "
            "never generated prose."
        )


def model_results() -> None:
    with st.expander("Model results"):
        st.markdown("**Primary comparison** — same 800 held-out users, same full-catalog protocol:")
        st.table(
            {
                "Architecture": ["Old Hybrid + CountSketch", "Fast production ALS"],
                "NDCG@10": ["0.1815", "**0.2588**"],
                "Recall@10": ["0.1682", "**0.2480**"],
                "p50 latency": ["924.6 ms", "**2.0 ms**"],
                "Decision": ["Replaced as default", "**Shipped**"],
            }
        )
        st.markdown(
            "**+42.6% relative NDCG@10** (paired 95% CI `[+0.0641, +0.0897]`) at roughly **465× lower latency**."
        )
        st.markdown("---")
        st.markdown("**Collaborative model progression** — 800-user confirmation samples, threshold 8:")
        st.table(
            {
                "Model": [
                    "Random",
                    "Popularity",
                    "CountSketch (previous)",
                    "Exact item-item",
                    "LightFM-ID",
                    "Tuned ALS",
                    "Oracle (ceiling)",
                ],
                "NDCG@10": ["0.0008", "0.1087", "0.1516", "0.1650", "0.1833", "**0.2624**", "1.0000"],
                "Recall@10": ["0.0009", "0.1000", "0.1430", "0.1472", "0.1589", "**0.2475**", "0.8650"],
                "Decision": [
                    "Floor reference",
                    "Baseline",
                    "Replaced",
                    "Retained as retrieval source",
                    "Rejected",
                    "**Promoted**",
                    "Not deployable",
                ],
            }
        )
        st.caption(
            "Exact item-item and LightFM-ID were measured on an earlier 1,000-user sample; the other "
            "rows come from the 800-user confirmation samples. Populations differ, so treat "
            "cross-row gaps as indicative rather than paired."
        )


def evaluation_protocol() -> None:
    with st.expander("Evaluation protocol"):
        st.markdown(
            "Every headline number ranks each user's held-out favourites against the **full "
            "~18,000-item catalog**. That is substantially harder than the common protocol of "
            "one positive against 99 sampled negatives, where the same model scores NDCG@10 "
            "**0.826**. That figure appears here only as a comparability diagnostic and is never "
            "used to select a model."
        )
        st.markdown(
            "- Held-out positives per user: ~10.6, so Recall@10 has a hard ceiling of **0.865**.\n"
            "- Confirmation samples are drawn from users no earlier experiment scored, and "
            "disjointness is asserted at sample time.\n"
            "- Decision rules are frozen in writing before the confirmation set is opened, once."
        )


def negative_results() -> None:
    with st.expander("What I tested — and rejected"):
        st.markdown(
            """
| Experiment | Result | Decision |
|---|---|---|
| **Pretrained synopsis embeddings** | Reduced NDCG@10 by 8.9% and Recall@10 by 11.9% on held-out users | Retired from the default path |
| **Learned linear fusion** | Generalised worse than the hand-set blend on untouched users (0.1814 vs 0.1996) | Rejected |
| **Segment-aware routing** | Routing sparse users to CountSketch dropped their NDCG@10 from 0.2003 to 0.1660 | Disabled by default |
| **Item-item supplementation** | Restored tail reach but cost 9% NDCG@10 and 4× retrieval latency | Kept optional |
| **Rich Hybrid as default** | No measurable ranking gain once ALS was strong, for 125× the latency | Constraint-rich requests only |
"""
        )
        st.caption(
            "Each of these was a real hypothesis with a predeclared decision rule. Reporting them "
            "is the point: the architecture is what survived, not what was hoped for."
        )


def health_panel(service: ShowcaseService) -> None:
    with st.expander("Deployment health"):
        health = service.health
        st.markdown(
            f"- **Model:** {health.headline}\n"
            f"- **Artifact verified:** {'yes' if health.als_available else 'no'}\n"
            f"- **Catalog aligned:** {'yes' if health.als_covered_items else 'no'}\n"
            f"- **Catalog items:** {health.catalog_items:,}\n"
            f"- **Covered by the trained model:** {health.als_covered_items:,}\n"
            f"- **Cold-start items (searchable, never given a model score):** {health.cold_start_items:,}\n"
            f"- **Model load time:** {health.load_seconds:.2f} s\n"
            f"- **Fast path ready:** {'yes' if health.serving_production_als else 'no'}"
        )
        if health.artifact_sha256:
            st.caption(f"Artifact SHA-256 {health.artifact_sha256[:16]}…")
        if health.error:
            st.warning(health.error)


def main() -> None:
    init_state()
    service = get_service()
    hero()
    st.divider()

    if not service.health.serving_production_als:
        model_unavailable(service)
    else:
        picker(service)
        count, free_text = controls(service)
        st.write("")
        if st.button(
            "Recommend",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.liked,
        ):
            with st.spinner("Ranking the full catalog…"):
                st.session_state.results = service.recommend(
                    st.session_state.liked,
                    disliked_ids=st.session_state.disliked,
                    limit=count,
                    free_text=free_text,
                )
        render_results(service)

    st.divider()
    st.subheader("Under the hood")
    how_it_works()
    model_results()
    evaluation_protocol()
    negative_results()
    health_panel(service)
    st.caption(
        "Portfolio project. Data is the CC0 Anime Recommendation Database 2020; there are no real "
        "users and no production traffic behind these numbers."
    )


if __name__ == "__main__":
    main()
