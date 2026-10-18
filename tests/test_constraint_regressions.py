"""Every returned title must satisfy the request's hard constraints.

The production failure: "Hey man, recommend me some Gundam movies. I want the
year after 2020." returned movies from 2021 onward that were not Gundam. Gemini
had put "Gundam" in a free-text field that the recommender ignores, so the only
hard constraints left were the format and the year.

These tests cover that prompt and its variations, and the neighbouring
constraint paths an audit of the same pipeline found weak: franchise exclusion,
follow-ups, franchise names containing a genre word or an ordinary word, a
voice actor whose name contains another credited name, a director treated as a
voice actor, and year windows. Each test checks returned IDs against the
fixture catalog rather than trusting the response's own description of them.

Everything here is deterministic. The live-model versions of these cases are in
data/evaluation/agent_constraints.json, run by scripts/evaluate_agent_constraints.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agents.replan import PROTECTED_FIELDS, candidate_relaxations
from app.agents.schemas import AgentIntent, ProviderHealth
from app.core.config import Settings
from app.main import create_app
from backend.anime_agent.agent import AnimeAgent
from backend.anime_agent.recommender import AnimeRecommender
from backend.anime_agent.title_family import (
    EligibilityConstraints,
    extract_excluded_title_families,
    extract_required_title_families,
    is_eligible,
)
from scripts.evaluate_agent_constraints import DEFAULT_CASES, load_cases, parser_provider, violations

PRODUCTION_PROMPT = "Hey man, recommend me some Gundam movies. I want the year after 2020."

MECHA = ["Action", "Sci-Fi", "Mecha"]

GUNDAM = {1, 2, 3, 4, 5, 6, 7}
GUNDAM_MOVIES_AFTER_2020 = {1, 2, 3}
GUNDAM_MOVIES = {1, 2, 3, 6, 7}
GUNDAM_TV = {4, 5}
NARUTO_MOVIES_AFTER_2010 = {21, 22, 23}
DETECTIVE_CONAN = {50, 51, 52}
SHINKAI_STAFF = {33, 72, 73}
HANAZAWA = {70, 74}


def entry(
    anime_id: int,
    title: str,
    *,
    media_type: str,
    year: int | None,
    genres: list[str],
    score: float = 7.5,
    episodes: int = 1,
    studios: list[str] | None = None,
    aliases: list[str] | None = None,
    synopsis: str | None = None,
    staff: list[dict[str, Any]] | None = None,
    roles: list[dict[str, Any]] | None = None,
    characters: list[str] | None = None,
) -> dict[str, Any]:
    character_names = characters or [f"Lead {anime_id}"]
    character_rows = [{"id": anime_id * 10 + i, "name": name, "role": "Main"} for i, name in enumerate(character_names)]
    staff_rows = staff or [{"id": 900, "name": "Test Director", "role": "Director"}]
    return {
        "id": anime_id,
        "title": title,
        "aliases": aliases or [title],
        "score": score,
        "rank": anime_id,
        "popularity": anime_id * 10,
        "members": 100_000 - anime_id,
        "synopsis": synopsis or f"{title} is a {' and '.join(genres).lower()} story.",
        "start_year": year,
        "type": media_type,
        "episodes": episodes,
        "image_url": "",
        "genres": genres,
        "genre_groups": {},
        "metadata_tokens": [f"genre_{genre}" for genre in genres],
        "studios": studios if studios is not None else ["Sunrise"],
        "producers": [],
        "characters": character_rows,
        "character_names": character_names,
        "character_relationships": character_rows,
        "staff": staff_rows,
        "staff_relationships": staff_rows,
        "creators": staff_rows,
        "voice_actors": [{"id": role["voice_actor_id"], "name": role["voice_actor"]} for role in roles or []],
        "voice_actor_roles": roles or [],
    }


def shinkai(role: str = "Director") -> list[dict[str, Any]]:
    return [{"id": 1117, "name": "Shinkai, Makoto", "role": role}]


def voice(actor_id: int, name: str, character: str) -> dict[str, Any]:
    return {
        "voice_actor_id": actor_id,
        "voice_actor": name,
        "character_id": actor_id * 7,
        "character": character,
        "language": "Japanese",
    }


# "Fate" and "ninja" are ordinary words in these synopses, as they are in the
# real catalog, so neither reads as a franchise name on its own.
FATE_SYNOPSIS = "A story about fate, choice and consequence."
NINJA_SYNOPSIS = "A young ninja trains to protect his village."


@pytest.fixture
def catalog() -> list[dict[str, Any]]:
    return [
        # Gundam. Ranked below the non-Gundam movies on purpose.
        entry(
            1,
            "Gundam: G no Reconguista Movie IV - Gekitou ni Sakebu Ai",
            media_type="Movie",
            year=2022,
            genres=MECHA,
            score=6.8,
        ),
        entry(2, "Kidou Senshi Gundam SEED Freedom", media_type="Movie", year=2024, genres=MECHA, score=7.5),
        entry(3, "Mobile Suit Gundam: Hathaway's Flash", media_type="Movie", year=2021, genres=MECHA, score=7.3),
        entry(
            4,
            "Kidou Senshi Gundam: Suisei no Majo",
            media_type="TV",
            year=2022,
            genres=MECHA,
            episodes=12,
            aliases=["Kidou Senshi Gundam: Suisei no Majo", "Mobile Suit Gundam: The Witch from Mercury"],
        ),
        entry(5, "SD Gundam World Heroes", media_type="TV", year=2021, genres=MECHA, score=6.0, episodes=24),
        entry(6, "Mobile Suit Gundam Wing: Endless Waltz Movie", media_type="Movie", year=1998, genres=MECHA),
        # A Gundam movie without the Mecha tag: a genre inferred from the
        # franchise name must not remove it.
        entry(7, "Turn A Gundam I Movie: Earth Light", media_type="Movie", year=2002, genres=["Sci-Fi", "Drama"]),
        # Not Gundam: the same format and years, and higher scores.
        entry(
            8,
            "Gintama: The Final",
            media_type="Movie",
            year=2021,
            genres=["Action", "Comedy"],
            score=9.0,
            synopsis=FATE_SYNOPSIS,
        ),
        entry(
            9,
            "Evangelion: 3.0+1.0 Thrice Upon a Time",
            media_type="Movie",
            year=2021,
            genres=MECHA,
            score=8.9,
            studios=["Khara"],
        ),
        entry(
            10,
            "Code Geass: Hangyaku no Lelouch",
            media_type="TV",
            year=2006,
            genres=MECHA,
            score=8.7,
            episodes=25,
            synopsis=FATE_SYNOPSIS,
            characters=["Lamperouge, Lelouch"],
        ),
        entry(
            11,
            "Tengen Toppa Gurren Lagann",
            media_type="TV",
            year=2007,
            genres=MECHA,
            score=8.6,
            episodes=27,
            synopsis=FATE_SYNOPSIS,
        ),
        entry(
            12,
            "Kyoukai Senki",
            media_type="TV",
            year=2021,
            genres=MECHA,
            score=6.5,
            episodes=13,
            synopsis=FATE_SYNOPSIS,
        ),
        # Naruto, and a ninja film that is not Naruto.
        entry(
            20,
            "Naruto",
            media_type="TV",
            year=2002,
            genres=["Action"],
            episodes=220,
            studios=["Pierrot"],
            synopsis=NINJA_SYNOPSIS,
        ),
        entry(
            21,
            "Naruto: Shippuuden Movie 6 - Road to Ninja",
            media_type="Movie",
            year=2012,
            genres=["Action"],
            studios=["Pierrot"],
        ),
        entry(
            22,
            "The Last: Naruto the Movie",
            media_type="Movie",
            year=2014,
            genres=["Action"],
            studios=["Pierrot"],
            synopsis=NINJA_SYNOPSIS,
        ),
        entry(
            23,
            "Boruto: Naruto the Movie",
            media_type="Movie",
            year=2015,
            genres=["Action"],
            studios=["Pierrot"],
            synopsis=NINJA_SYNOPSIS,
        ),
        entry(
            24,
            "Naruto Movie 1: Dai Katsugeki!! Yuki Hime Shinobu Houjou Dattebayo!",
            media_type="Movie",
            year=2004,
            genres=["Action"],
            studios=["Pierrot"],
            synopsis=NINJA_SYNOPSIS,
        ),
        entry(
            25,
            "Ninja Scroll",
            media_type="Movie",
            year=1993,
            genres=["Action"],
            score=8.9,
            studios=["Madhouse"],
            synopsis=FATE_SYNOPSIS,
        ),
        # Studio Ghibli. The newest film's year is unknown to this catalog.
        entry(
            30, "Gake no Ue no Ponyo", media_type="Movie", year=2008, genres=["Adventure"], studios=["Studio Ghibli"]
        ),
        entry(31, "Kaze Tachinu", media_type="Movie", year=2013, genres=["Drama"], studios=["Studio Ghibli"]),
        entry(
            32,
            "Kimitachi wa Dou Ikiru ka",
            media_type="Movie",
            year=None,
            genres=["Adventure"],
            score=0.0,
            studios=["Studio Ghibli"],
        ),
        entry(
            33,
            "Suzume no Tojimari",
            media_type="Movie",
            year=2022,
            genres=["Adventure"],
            score=8.6,
            studios=["CoMix Wave Films"],
            staff=shinkai(),
        ),
        # Fate, plus a title that merely contains the word.
        entry(
            40,
            "Fate/stay night Movie: Heaven's Feel - III. Spring Song",
            media_type="Movie",
            year=2020,
            genres=["Action", "Fantasy"],
            studios=["ufotable"],
        ),
        entry(
            41,
            "Fate/Grand Order: Shinsei Entaku Ryouiki Camelot 1",
            media_type="Movie",
            year=2020,
            genres=["Action", "Fantasy"],
            studios=["Signal.MD"],
        ),
        entry(
            42, "Fate/Zero", media_type="TV", year=2011, genres=["Action", "Fantasy"], episodes=13, studios=["ufotable"]
        ),
        entry(
            43, "Bayonetta: Bloody Fate", media_type="Movie", year=2013, genres=["Action", "Fantasy"], studios=["Gonzo"]
        ),
        # Detective Conan: the English name is on only some entries, and
        # "Detective" is also a genre. Mirai Shounen Conan is a different show.
        entry(
            50,
            "Meitantei Conan",
            media_type="TV",
            year=1996,
            genres=["Mystery", "Detective"],
            episodes=1000,
            studios=["TMS Entertainment"],
            aliases=["Meitantei Conan", "Detective Conan", "Case Closed"],
        ),
        entry(
            51,
            "Meitantei Conan Movie 01: Tokei Jikake no Matenrou",
            media_type="Movie",
            year=1997,
            genres=["Mystery", "Adventure"],
            studios=["TMS Entertainment"],
        ),
        entry(
            52,
            "Meitantei Conan Movie 25: Halloween no Hanayome",
            media_type="Movie",
            year=2022,
            genres=["Mystery", "Adventure"],
            studios=["TMS Entertainment"],
            aliases=["Meitantei Conan Movie 25: Halloween no Hanayome", "Detective Conan: The Bride of Halloween"],
        ),
        entry(
            53,
            "Mirai Shounen Conan",
            media_type="TV",
            year=1978,
            genres=["Adventure"],
            episodes=26,
            studios=["Nippon Animation"],
            aliases=["Mirai Shounen Conan", "Future Boy Conan"],
        ),
        # Demon Slayer: one entry carries the English name.
        entry(
            60,
            "Kimetsu no Yaiba",
            media_type="TV",
            year=2019,
            genres=["Action"],
            episodes=26,
            studios=["ufotable"],
            aliases=["Kimetsu no Yaiba", "Demon Slayer: Kimetsu no Yaiba"],
        ),
        entry(
            61,
            "Kimetsu no Yaiba Movie: Mugen Ressha-hen",
            media_type="Movie",
            year=2020,
            genres=["Action"],
            studios=["ufotable"],
        ),
        entry(
            62,
            "Kimetsu no Yaiba: Yuukaku-hen",
            media_type="TV",
            year=2021,
            genres=["Action"],
            episodes=11,
            studios=["ufotable"],
        ),
        # People. "Kana" is a different, separately credited voice actor.
        entry(
            70,
            "Steins;Gate",
            media_type="TV",
            year=2011,
            genres=["Sci-Fi"],
            episodes=24,
            studios=["White Fox"],
            roles=[voice(471, "Hanazawa, Kana", "Shiina, Mayuri")],
        ),
        entry(
            71,
            "Kanon",
            media_type="TV",
            year=2006,
            genres=["Drama"],
            episodes=24,
            studios=["Kyoto Animation"],
            roles=[voice(999, "Kana", "Tsukimiya, Ayu")],
        ),
        entry(
            72,
            "Hoshi wo Ou Kodomo",
            media_type="Movie",
            year=2011,
            genres=["Adventure"],
            studios=["CoMix Wave Films"],
            staff=shinkai(),
        ),
        entry(
            73,
            "Kanojo to Kanojo no Neko",
            media_type="ONA",
            year=1999,
            genres=["Drama"],
            studios=["CoMix Wave Films"],
            staff=shinkai(),
            roles=[voice(1117, "Shinkai, Makoto", "Chobi")],
        ),
        entry(
            74,
            "Psycho-Pass",
            media_type="TV",
            year=2012,
            genres=["Sci-Fi"],
            episodes=22,
            studios=["Production I.G"],
            roles=[voice(471, "Hanazawa, Kana", "Tsunemori, Akane")],
        ),
        entry(80, "Golden Boy", media_type="OVA", year=1995, genres=["Comedy"], episodes=6, studios=["APPP"]),
    ]


@pytest.fixture
def by_id(catalog: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["id"]): item for item in catalog}


@pytest.fixture
def agent(catalog: list[dict[str, Any]]) -> AnimeAgent:
    return AnimeAgent(AnimeRecommender(catalog))


def returned_ids(response: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for step in response.get("trace") or []:
        if step.get("tool") not in {"recommend_anime", "rank_catalog", "search_anime"}:
            continue
        for row in (step.get("result") or {}).get("results") or []:
            if isinstance(row, dict) and row.get("id") is not None:
                ids.append(int(row["id"]))
    return ids


def respond(agent: AnimeAgent, message: str, earlier: list[str] | None = None) -> dict[str, Any]:
    """The deterministic path, end to end: what users get when Gemini is down."""
    history: list[dict[str, str]] = []
    for text in earlier or []:
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": "Here are some titles."}]
    intent = agent._rule_based_intent(message, history)
    return agent._respond_from_intent(intent, message, history, None, {})


def assert_satisfies(
    by_id: dict[int, dict[str, Any]],
    ids: list[int],
    *,
    within: set[int] | None = None,
    outside: set[int] | None = None,
    formats: set[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
) -> None:
    """Check each returned id against the catalog, not against the response."""
    for anime_id in ids:
        item = by_id[anime_id]
        label = f"{item['title']} ({anime_id})"
        if within is not None:
            assert anime_id in within, f"{label} is outside the required set"
        if outside is not None:
            assert anime_id not in outside, f"{label} belongs to an excluded set"
        if formats is not None:
            assert item["type"] in formats, f"{label} is a {item['type']}"
        year = item["start_year"]
        if min_year is not None:
            assert year is not None and year >= min_year, f"{label} started in {year}"
        if max_year is not None:
            assert year is not None and year <= max_year, f"{label} started in {year}"


# ------------------------------------------------- the prompt and its variations


def test_the_production_prompt_returns_only_gundam_movies_after_2020(
    agent: AnimeAgent, by_id: dict[int, dict[str, Any]]
) -> None:
    ids = returned_ids(respond(agent, PRODUCTION_PROMPT))

    assert set(ids) == GUNDAM_MOVIES_AFTER_2020
    assert_satisfies(by_id, ids, within=GUNDAM, formats={"Movie"}, min_year=2021)


@pytest.mark.parametrize(
    ("message", "expected", "formats", "min_year"),
    [
        ("Can you recommend Gundam movies released after 2020?", GUNDAM_MOVIES_AFTER_2020, {"Movie"}, 2021),
        ("Give me five Gundam films from 2021 onward.", GUNDAM_MOVIES_AFTER_2020, {"Movie"}, 2021),
        ("Recommend some Gundam TV series.", GUNDAM_TV, {"TV"}, None),
        ("Only suggest Gundam movies.", GUNDAM_MOVIES, {"Movie"}, None),
        ("Recommend Naruto movies released after 2010.", NARUTO_MOVIES_AFTER_2010, {"Movie"}, 2011),
    ],
)
def test_franchise_variations_return_exactly_the_eligible_entries(
    agent: AnimeAgent,
    by_id: dict[int, dict[str, Any]],
    message: str,
    expected: set[int],
    formats: set[str],
    min_year: int | None,
) -> None:
    ids = returned_ids(respond(agent, message))

    assert set(ids) == expected
    assert_satisfies(by_id, ids, within=expected, formats=formats, min_year=min_year)


def test_five_films_is_a_ceiling_not_a_quota(agent: AnimeAgent) -> None:
    """Three Gundam films from 2021 on exist, so five asked for returns three."""
    ids = returned_ids(respond(agent, "Give me five Gundam films from 2021 onward."))

    assert len(ids) == 3


def test_similar_to_gundam_is_a_reference_not_a_restriction(agent: AnimeAgent) -> None:
    message = "Recommend 10 anime similar to Gundam."
    assert extract_required_title_families(message, agent.recommender) == []

    ids = returned_ids(respond(agent, message))

    assert ids
    assert set(ids) - GUNDAM, "a similarity request may reach beyond the franchise"


def test_movies_like_gundam_keeps_format_and_year_hard(agent: AnimeAgent, by_id: dict[int, dict[str, Any]]) -> None:
    message = "Recommend 10 movies like Gundam after 2020."
    assert extract_required_title_families(message, agent.recommender) == []

    ids = returned_ids(respond(agent, message))

    assert ids
    assert_satisfies(by_id, ids, formats={"Movie"}, min_year=2021)
    assert set(ids) - GUNDAM


def test_studio_ghibli_is_a_studio_and_an_empty_answer_says_why(agent: AnimeAgent) -> None:
    """The newest Ghibli film has no year in this catalog, so nothing qualifies."""
    message = "Recommend Studio Ghibli movies after 2020."
    assert extract_required_title_families(message, agent.recommender) == []

    response = respond(agent, message)

    assert returned_ids(response) == []
    answer = response["answer"]
    assert "2021" in answer
    assert "Movie" in answer
    assert "Try adding a genre, title, or format" not in answer


def test_an_unsatisfiable_franchise_request_is_reported_not_padded(agent: AnimeAgent) -> None:
    response = respond(agent, "Recommend Gundam movies released after 2035.")

    assert returned_ids(response) == []
    assert "no verified match" in response["answer"].casefold()


# ------------------------------------------------- names: aliases and ambiguity


def test_a_genre_word_inside_a_franchise_name_is_not_a_genre_filter(
    agent: AnimeAgent, by_id: dict[int, dict[str, Any]]
) -> None:
    """ "Detective Conan" once became genre=Detective, and the movies lack that tag."""
    message = "Recommend Detective Conan movies."
    intent = agent._rule_based_intent(message, [])
    agent._normalize_explicit_request(intent, message)
    assert intent.required_title_families == ["detective conan"]
    assert "Detective" not in intent.include_genres

    ids = returned_ids(respond(agent, message))

    assert set(ids) == {51, 52}
    assert_satisfies(by_id, ids, within=DETECTIVE_CONAN, formats={"Movie"})


def test_detective_conan_does_not_include_every_title_containing_conan(agent: AnimeAgent) -> None:
    members = agent.recommender.title_family_members("detective conan")

    assert members == DETECTIVE_CONAN
    assert 53 not in members


def test_an_english_name_reaches_entries_that_only_carry_the_romaji_name(agent: AnimeAgent) -> None:
    assert set(returned_ids(respond(agent, "Recommend Demon Slayer movies."))) == {61}


def test_an_ordinary_word_franchise_uses_strict_membership(agent: AnimeAgent) -> None:
    """ "Fate movies" means the Fate/ titles, not "Bayonetta: Bloody Fate"."""
    assert not agent.recommender.title_family_reads_as_name("fate")
    assert extract_required_title_families("Recommend Fate movies.", agent.recommender) == ["fate"]

    assert set(returned_ids(respond(agent, "Recommend Fate movies."))) == {40, 41}


@pytest.mark.parametrize("message", ["Recommend Ninja movies.", "Recommend some fate-themed stories."])
def test_an_ordinary_word_without_franchise_evidence_is_not_a_franchise(agent: AnimeAgent, message: str) -> None:
    assert extract_required_title_families(message, agent.recommender) == []


# --------------------------------------------------------------- follow-ups


def test_a_follow_up_keeps_the_franchise_without_the_model(agent: AnimeAgent, by_id: dict[int, dict[str, Any]]) -> None:
    ids = returned_ids(respond(agent, "What about TV series instead?", ["Recommend Gundam movies after 2020."]))

    assert ids
    assert_satisfies(by_id, ids, within=GUNDAM_TV, formats={"TV"})


def test_a_follow_up_naming_another_franchise_switches_to_it(agent: AnimeAgent) -> None:
    """The year carries over from the earlier turn; the franchise does not.

    No Naruto movie in this catalog starts after 2020, so the honest answer is
    an empty one that names Naruto -- never the Gundam movies from before.
    """
    response = respond(agent, "What about Naruto movies instead?", ["Recommend Gundam movies after 2020."])

    assert response["_debug"]["franchise_constraints"]["required_title_families"] == ["naruto"]
    assert set(returned_ids(response)) <= {21, 22, 23, 24}
    assert "Naruto" in response["answer"]


def test_a_similarity_follow_up_does_not_inherit_the_franchise(agent: AnimeAgent) -> None:
    message = "What about 10 anime similar to Gundam instead?"
    ids = returned_ids(respond(agent, message, ["Recommend Gundam movies after 2020."]))

    assert set(ids) - GUNDAM


# ------------------------------------------------------- franchise exclusion


@pytest.mark.parametrize(
    "message",
    [
        "Recommend mecha anime but not Gundam.",
        "Recommend mecha anime, anything but Gundam.",
        "Recommend mecha anime except Gundam.",
        "Recommend mecha anime without any Gundam.",
    ],
)
def test_an_excluded_franchise_is_removed_before_ranking(
    agent: AnimeAgent, by_id: dict[int, dict[str, Any]], message: str
) -> None:
    assert extract_excluded_title_families(message, agent.recommender) == ["gundam"]

    response = respond(agent, message)
    ids = returned_ids(response)

    assert ids
    assert_satisfies(by_id, ids, outside=GUNDAM)
    recommend = next(step for step in response["trace"] if step["tool"] == "recommend_anime")
    assert recommend["arguments"]["excluded_title_families"] == ["gundam"]


def test_an_exclusion_survives_a_follow_up(agent: AnimeAgent, by_id: dict[int, dict[str, Any]]) -> None:
    ids = returned_ids(
        respond(agent, "Show me more mecha, movies this time.", ["Recommend mecha anime but not Gundam."])
    )

    assert ids
    assert_satisfies(by_id, ids, outside=GUNDAM, formats={"Movie"})


def test_asking_for_a_franchise_outright_overrides_an_earlier_exclusion(agent: AnimeAgent) -> None:
    ids = returned_ids(respond(agent, "Actually, recommend Gundam movies.", ["Recommend mecha anime but not Gundam."]))

    assert set(ids) == GUNDAM_MOVIES


def test_a_negated_genre_is_not_a_franchise_exclusion(agent: AnimeAgent) -> None:
    assert extract_excluded_title_families("Recommend Gundam movies, not romance.", agent.recommender) == []


def test_the_final_gate_removes_an_excluded_franchise_row(agent: AnimeAgent) -> None:
    message = "Recommend mecha anime but not Gundam."
    response = respond(agent, message)
    step = next(step for step in response["trace"] if step["tool"] == "recommend_anime")
    step["result"]["results"].append({"id": 2, "title": "Kidou Senshi Gundam SEED Freedom"})

    intent = agent._rule_based_intent(message, [])
    agent._normalize_explicit_request(intent, message)
    dropped = agent.enforce_result_eligibility(intent, response)

    assert dropped == ["Kidou Senshi Gundam SEED Freedom"]
    assert 2 not in returned_ids(response)


# ------------------------------------------------ the catalog, not the row


def test_a_row_claiming_the_wrong_format_and_year_is_checked_against_the_catalog(agent: AnimeAgent) -> None:
    """A result row is a claim. Suisei no Majo is a 2022 TV series whatever the row says."""
    response = respond(agent, PRODUCTION_PROMPT)
    step = next(step for step in response["trace"] if step["tool"] == "recommend_anime")
    step["result"]["results"].append(
        {"id": 4, "title": "Kidou Senshi Gundam: Suisei no Majo", "type": "Movie", "start_year": 2023}
    )

    intent = agent._rule_based_intent(PRODUCTION_PROMPT, [])
    agent._normalize_explicit_request(intent, PRODUCTION_PROMPT)

    assert agent.enforce_result_eligibility(intent, response) == ["Kidou Senshi Gundam: Suisei no Majo"]


@pytest.mark.parametrize(
    "constraints",
    [EligibilityConstraints(min_year=2021), EligibilityConstraints(max_year=2030)],
)
def test_an_unknown_year_satisfies_no_year_constraint(agent: AnimeAgent, constraints: EligibilityConstraints) -> None:
    assert not is_eligible(agent.recommender, constraints, {"id": 32})


# ------------------------------------------------------------------- people


def test_a_longer_name_wins_over_a_credited_name_inside_it(agent: AnimeAgent) -> None:
    """ "Kana Hanazawa" also contains "Kana", another voice actor's credited name."""
    assert agent._extract_required_voice_actors("Recommend anime with Kana Hanazawa.") == ["Hanazawa, Kana"]


def test_a_voice_actor_request_returns_only_verified_credits(
    agent: AnimeAgent, by_id: dict[int, dict[str, Any]]
) -> None:
    ids = returned_ids(respond(agent, "Recommend anime with Kana Hanazawa."))

    assert set(ids) == HANAZAWA


def test_a_director_is_not_made_a_required_voice_actor(agent: AnimeAgent) -> None:
    """Shinkai has one voice credit; "directed by" asks for his work as a director."""
    message = "Recommend anime directed by Makoto Shinkai."
    assert agent._extract_required_voice_actors(message) == []

    ids = returned_ids(respond(agent, message))

    assert ids
    assert set(ids) <= SHINKAI_STAFF
    assert len(ids) > 1


def test_voiced_by_still_requires_the_voice_credit(agent: AnimeAgent) -> None:
    assert "Shinkai, Makoto" in agent._extract_required_voice_actors("Recommend anime voiced by Makoto Shinkai.")


# -------------------------------------------------------------------- years


@pytest.mark.parametrize(
    ("message", "min_year", "max_year"),
    [
        (PRODUCTION_PROMPT, 2021, None),
        ("Give me five Gundam films from 2021 onward.", 2021, None),
        ("Recommend TV anime released between 2015 and 2017.", 2015, 2017),
        ("Recommend TV anime from 2010-2012.", 2010, 2012),
        ("Anime from 2015 to 2017, please", 2015, 2017),
        ("Recommend anime released in 2019.", 2019, 2019),
        ("Recommend anime released in 2019 or later.", 2019, None),
        ("Recommend some OVAs from before 2000.", None, 1999),
        ("Recommend 5 anime", None, None),
    ],
)
def test_year_constraints_are_read_from_the_message(
    agent: AnimeAgent, message: str, min_year: int | None, max_year: int | None
) -> None:
    assert agent._requested_min_year(message) == min_year
    assert agent._requested_max_year(message) == max_year


@pytest.mark.parametrize(
    ("message", "formats"),
    [
        ("Recommend some OVAs from before 2000.", ["OVA"]),
        ("Give me five Gundam films from 2021 onward.", ["Movie"]),
        ("Recommend some Gundam TV series.", ["TV"]),
    ],
)
def test_formats_are_read_in_plural_and_everyday_forms(agent: AnimeAgent, message: str, formats: list[str]) -> None:
    assert agent._matched_formats(message) == formats


def test_explicit_format_and_year_are_protected_from_replanning(agent: AnimeAgent) -> None:
    protected = agent.explicit_constraint_fields("Mecha movies after 2030")
    intent = AgentIntent(intent="recommend", formats=["Movie"], min_year=2031, min_score=9.0)

    assert protected == frozenset({"formats", "min_year"})
    assert candidate_relaxations(intent, protected) == ["min_score"]


# ------------------------------------------------------------ through the API


class ScriptedProvider:
    """Returns a fixed intent, as Gemini would, and echoes the verified answer."""

    def __init__(self, intent: AgentIntent):
        self.name = "gemini"
        self.model = "scripted"
        self.intent = intent
        self.verified_data: dict[str, Any] | None = None

    async def parse_intent(self, message: str, context: Any) -> AgentIntent:
        return self.intent

    async def generate_tool_response(self, user_message: str, verified_tool_data: dict[str, Any]) -> str:
        self.verified_data = verified_tool_data
        return str(verified_tool_data["deterministic_fallback"])

    async def generate_explanation(self, user_message: str, verified_recommendation: dict[str, Any]) -> str:
        return await self.generate_tool_response(user_message, verified_recommendation)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, model=self.model, available=True)

    async def close(self) -> None:
        return None


def chat(
    tmp_path: Path,
    catalog: list[dict[str, Any]],
    intent: AgentIntent,
    message: str,
    history: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Any], ScriptedProvider]:
    provider = ScriptedProvider(intent)
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'constraints.db').as_posix()}",
        llm_provider="gemini",
        rate_limit_requests=1000,
    )
    app = create_app(settings=settings, catalog=catalog, providers={"gemini": provider})
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": message, "history": history or [], "debug": True})
    assert response.status_code == 200
    return response.json(), provider


def handed_to_model(provider: ScriptedProvider) -> list[int]:
    assert provider.verified_data is not None
    return [
        int(row["id"])
        for step in provider.verified_data["verified_tool_trace"]
        for row in (step.get("result") or {}).get("results") or []
        if row.get("id") is not None
    ]


def test_the_production_intent_cannot_surface_non_gundam_movies(
    tmp_path: Path, catalog: list[dict[str, Any]], by_id: dict[int, dict[str, Any]]
) -> None:
    """The intent Gemini actually produced: the franchise only in catalog_query."""
    intent = AgentIntent(intent="recommend", catalog_query="Gundam", formats=["Movie"], min_year=2021, top_k=10)
    body, provider = chat(tmp_path, catalog, intent, PRODUCTION_PROMPT)

    assert set(returned_ids(body)) == GUNDAM_MOVIES_AFTER_2020
    assert set(handed_to_model(provider)) == GUNDAM_MOVIES_AFTER_2020
    assert body["debug"]["validated_intent"]["required_title_families"] == ["gundam"]


def test_a_franchise_the_model_names_is_accepted_when_the_user_typed_it(
    tmp_path: Path, catalog: list[dict[str, Any]], by_id: dict[int, dict[str, Any]]
) -> None:
    """No pattern reads "anything Gundam from 2022?"; the model's proposal does."""
    message = "anything Gundam from 2022?"
    assert extract_required_title_families(message, AnimeRecommender(catalog)) == []
    intent = AgentIntent(intent="recommend", required_franchises=["Gundam"], min_year=2022, top_k=10)

    body, _provider = chat(tmp_path, catalog, intent, message)
    ids = returned_ids(body)

    assert set(ids) == {1, 2, 4}
    assert_satisfies(by_id, ids, within=GUNDAM, min_year=2022)


@pytest.mark.parametrize(
    ("message", "proposed"),
    [
        ("Recommend some mecha anime.", "Gundam"),
        ("Recommend 10 anime similar to Gundam.", "Gundam"),
        ("Recommend Studio Ghibli movies.", "Studio Ghibli"),
    ],
)
def test_a_franchise_proposal_is_refused_when_unsupported(
    tmp_path: Path, catalog: list[dict[str, Any]], message: str, proposed: str
) -> None:
    intent = AgentIntent(intent="recommend", required_franchises=[proposed], top_k=10)

    body, _provider = chat(tmp_path, catalog, intent, message)

    assert body["debug"]["validated_intent"]["required_title_families"] == []


def test_a_typed_franchise_the_catalog_lacks_matches_nothing(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    intent = AgentIntent(intent="recommend", required_franchises=["Zorblax"], formats=["Movie"], top_k=10)

    body, _provider = chat(tmp_path, catalog, intent, "Recommend Zorblax movies.")

    assert returned_ids(body) == []
    assert "no verified match" in body["answer"].casefold()


def test_a_follow_up_keeps_the_franchise_when_the_model_drops_it(
    tmp_path: Path, catalog: list[dict[str, Any]], by_id: dict[int, dict[str, Any]]
) -> None:
    history = [
        {"role": "user", "content": "Recommend Gundam movies after 2020."},
        {"role": "assistant", "content": "Here are three Gundam movies."},
    ]
    intent = AgentIntent(intent="recommend", formats=["TV"], top_k=10)

    body, _provider = chat(tmp_path, catalog, intent, "What about TV series instead?", history)
    ids = returned_ids(body)

    assert ids
    assert_satisfies(by_id, ids, within=GUNDAM_TV, formats={"TV"})


def test_a_genre_the_model_infers_from_a_franchise_does_not_filter_it(
    tmp_path: Path, catalog: list[dict[str, Any]]
) -> None:
    """Turn A Gundam is not tagged Mecha; "Gundam movies" did not ask for Mecha."""
    intent = AgentIntent(
        intent="recommend", catalog_query="Gundam", include_genres=["Mecha"], formats=["Movie"], top_k=10
    )

    body, _provider = chat(tmp_path, catalog, intent, "Only suggest Gundam movies.")

    assert set(returned_ids(body)) == GUNDAM_MOVIES


def test_typed_format_and_year_are_never_relaxed(tmp_path: Path, catalog: list[dict[str, Any]]) -> None:
    intent = AgentIntent(intent="recommend", include_genres=["Mecha"], formats=["Movie"], min_year=2031, top_k=5)

    body, _provider = chat(tmp_path, catalog, intent, "Mecha movies after 2030")

    relaxed = {step["field"] for step in body.get("relaxations") or []}
    assert not relaxed & {"formats", "min_year"}
    assert returned_ids(body) == []


def test_an_exclusion_holds_through_the_api(
    tmp_path: Path, catalog: list[dict[str, Any]], by_id: dict[int, dict[str, Any]]
) -> None:
    intent = AgentIntent(intent="recommend", include_genres=["Mecha"], top_k=10)

    body, provider = chat(tmp_path, catalog, intent, "Recommend mecha anime but not Gundam.")

    assert returned_ids(body)
    assert_satisfies(by_id, returned_ids(body), outside=GUNDAM)
    assert_satisfies(by_id, handed_to_model(provider), outside=GUNDAM)


# ---------------------------------------------------- schema and protection


def test_the_model_proposes_franchises_but_cannot_set_the_verified_fields() -> None:
    properties = AgentIntent.provider_json_schema()["properties"]

    assert "required_franchises" in properties
    assert "required_title_families" not in properties
    assert "excluded_title_families" not in properties


def test_franchise_fields_are_protected_from_the_relaxation_ladder() -> None:
    assert {"required_title_families", "excluded_title_families"} <= PROTECTED_FIELDS


# ------------------------------------------------ the permanent case suite


def test_the_permanent_suite_is_well_formed_and_covers_the_reported_prompts() -> None:
    cases = load_cases(DEFAULT_CASES)
    messages = {turn["message"] for case in cases for turn in case["turns"]}

    for required in (
        PRODUCTION_PROMPT,
        "Can you recommend Gundam movies released after 2020?",
        "Give me five Gundam films from 2021 onward.",
        "Recommend some Gundam TV series.",
        "Recommend anime similar to Gundam.",
        "Only suggest Gundam movies.",
        "Recommend Naruto movies released after 2010.",
        "Recommend Studio Ghibli movies after 2020.",
    ):
        assert required in messages


def test_the_suite_oracle_flags_each_kind_of_violation(by_id: dict[int, dict[str, Any]]) -> None:
    gintama, seed_freedom, turn_a = by_id[8], by_id[2], by_id[7]
    expect = {"family_any": ["gundam"], "formats": ["Movie"], "min_year": 2021}

    assert violations(seed_freedom, expect) == []
    assert violations(gintama, expect) == ["not in family ['gundam']"]
    assert violations(turn_a, expect) == ["start_year 2002 < 2021"]
    assert violations(seed_freedom, {"excluded_families": ["gundam"]}) == ["belongs to excluded family gundam"]
    assert violations(by_id[43], {"family_prefix_any": ["fate"]}) == ["no title starts with ['fate']"]
    assert violations(by_id[40], {"family_prefix_any": ["fate"]}) == []
    assert violations(None, expect) == ["id not in catalog"]


def test_the_suite_reads_the_parsing_provider_from_the_debug_trace() -> None:
    """parser_mode says "structured_llm" for a Gemini parse; the provider is in the trace."""
    gemini = {
        "parser_mode": "structured_llm",
        "debug": {"provider_attempts": [{"phase": "intent", "provider": "gemini", "outcome": "success"}]},
    }
    fallback = {
        "parser_mode": "rule_fallback",
        "debug": {"provider_attempts": [{"phase": "intent", "provider": "gemini", "outcome": "failed"}]},
    }

    assert parser_provider(gemini) == "gemini"
    assert parser_provider(fallback) == "rule_fallback"
    assert parser_provider({}) == "rule_fallback"
