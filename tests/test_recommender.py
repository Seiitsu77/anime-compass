from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from anime_agent.agent import AnimeAgent  # noqa: E402
from anime_agent.data_pipeline import (  # noqa: E402
    build_catalog,
    clean_label,
    metadata_token,
    normalize_character_role,
    selected_staff_role,
    write_catalog,
)
from anime_agent.entities import EntityResolver  # noqa: E402
from anime_agent.recommender import AnimeRecommender, creator_roles, series_key  # noqa: E402
from anime_agent.server import SessionStore, merge_profiles  # noqa: E402

CATALOG = [
    {
        "id": 1,
        "title": "Space Quest",
        "score": 8.8,
        "rank": 10,
        "popularity": 20,
        "members": 100000,
        "synopsis": "A crew travels across space with comedy and action.",
        "start_year": 2019,
        "type": "TV",
        "episodes": 24,
        "image_url": "",
        "genres": ["Action", "Sci-Fi", "Comedy"],
        "studios": ["North Studio"],
        "producers": [],
    },
    {
        "id": 2,
        "title": "Quiet Kitchen",
        "score": 8.2,
        "rank": 50,
        "popularity": 80,
        "members": 50000,
        "synopsis": "A warm slice of life story about cooking.",
        "start_year": 2020,
        "type": "TV",
        "episodes": 12,
        "image_url": "",
        "genres": ["Slice of Life", "Gourmet"],
        "studios": ["Home Studio"],
        "producers": [],
    },
    {
        "id": 3,
        "title": "Space Quest Movie",
        "score": 8.4,
        "rank": 40,
        "popularity": 60,
        "members": 70000,
        "synopsis": "The space crew returns for a focused movie.",
        "start_year": 2022,
        "type": "Movie",
        "episodes": 1,
        "image_url": "",
        "genres": ["Action", "Sci-Fi"],
        "studios": ["North Studio"],
        "producers": [],
    },
    {
        "id": 4,
        "title": "Stage Hearts",
        "score": 7.6,
        "rank": 120,
        "popularity": 200,
        "members": 40000,
        "synopsis": "Two classmates build a close friendship and slowly face their romantic feelings for each other.",
        "start_year": 2021,
        "type": "TV",
        "episodes": 12,
        "image_url": "",
        "genres": ["Drama", "Romance", "School"],
        "studios": ["Heart Studio"],
        "producers": [],
    },
    {
        "id": 5,
        "title": "Stage Battles",
        "score": 9.1,
        "rank": 5,
        "popularity": 5,
        "members": 120000,
        "synopsis": "Elite students compete in a school tournament with intense battles and tactical rivalries.",
        "start_year": 2021,
        "type": "TV",
        "episodes": 12,
        "image_url": "",
        "genres": ["Drama", "School"],
        "studios": ["Heart Studio"],
        "producers": [],
    },
    {
        "id": 6,
        "title": "Future Mystery",
        "score": 8.3,
        "rank": 20,
        "popularity": 30,
        "members": 90000,
        "synopsis": "A compact mystery about hidden memories and a careful investigation.",
        "start_year": 2021,
        "type": "TV",
        "episodes": 10,
        "image_url": "",
        "genres": ["Mystery"],
        "studios": ["Puzzle Studio"],
        "producers": [],
    },
    {
        "id": 7,
        "title": "Long Mystery",
        "score": 8.5,
        "rank": 18,
        "popularity": 25,
        "members": 95000,
        "synopsis": "A long-form mystery with many suspects and a slow investigation.",
        "start_year": 2022,
        "type": "TV",
        "episodes": 24,
        "image_url": "",
        "genres": ["Mystery"],
        "studios": ["Puzzle Studio"],
        "producers": [],
    },
    {
        "id": 8,
        "title": "Old Mystery",
        "score": 9.0,
        "rank": 3,
        "popularity": 10,
        "members": 150000,
        "synopsis": "A classic short mystery about a locked-room case.",
        "start_year": 2018,
        "type": "TV",
        "episodes": 10,
        "image_url": "",
        "genres": ["Mystery"],
        "studios": ["Puzzle Studio"],
        "producers": [],
    },
    {
        "id": 9,
        "title": "Death Note",
        "score": 9.0,
        "rank": 2,
        "popularity": 2,
        "members": 200000,
        "synopsis": "A supernatural mystery about a notebook that can kill people.",
        "start_year": 2006,
        "type": "TV",
        "episodes": 37,
        "image_url": "",
        "genres": ["Mystery", "Supernatural", "Suspense"],
        "studios": ["Madhouse"],
        "producers": [],
    },
    {
        "id": 10,
        "title": "Death Note: Rewrite",
        "score": 7.7,
        "rank": 200,
        "popularity": 100,
        "members": 80000,
        "synopsis": "A recap special for the supernatural notebook case.",
        "start_year": 2007,
        "type": "TV Special",
        "episodes": 2,
        "image_url": "",
        "genres": ["Mystery", "Supernatural", "Suspense"],
        "studios": ["Madhouse"],
        "producers": [],
    },
    {
        "id": 11,
        "title": "Spirit Garden",
        "score": 7.8,
        "rank": 180,
        "popularity": 160,
        "members": 60000,
        "synopsis": "A gentle supernatural story about spirits helping a small town.",
        "start_year": 2020,
        "type": "TV",
        "episodes": 12,
        "image_url": "",
        "genres": ["Supernatural", "Slice of Life"],
        "studios": ["Spirit Studio"],
        "producers": [],
    },
]


VOICE_ACTOR_TITLES = (
    "Crimson Harbor",
    "Silver Lantern",
    "Autumn Signal",
    "Glass Horizon",
    "Midnight Recipe",
    "Blue Archive",
    "Hidden Melody",
    "Winter Circuit",
)


def voice_actor_catalog() -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    for index, title in enumerate(VOICE_ACTOR_TITLES, start=1):
        anime_id = 100 + index
        character_id = 500 + index
        catalog.append(
            {
                "id": anime_id,
                "title": title,
                "score": 8.9 - index * 0.1,
                "rank": 20 + index,
                "popularity": 40 + index,
                "members": 100000 - index * 1000,
                "synopsis": f"A distinct catalog story for {title}.",
                "start_year": 2020 + (index % 4),
                "type": "TV",
                "episodes": 12,
                "image_url": "",
                "genres": ["Drama", "Fantasy" if index % 2 else "Comedy"],
                "studios": [f"Studio {index}"],
                "producers": [],
                "characters": [
                    {
                        "id": character_id,
                        "name": f"Character {index}",
                        "role": "Main",
                        "voice_actors": [
                            {
                                "id": 642,
                                "name": "Matsuoka, Yoshitsugu",
                                "language": "Japanese",
                            }
                        ],
                    }
                ],
                "staff": [],
                "voice_actors": [{"id": 642, "name": "Matsuoka, Yoshitsugu", "language": "Japanese"}],
                "voice_actor_roles": [
                    {
                        "voice_actor_id": 642,
                        "voice_actor": "Matsuoka, Yoshitsugu",
                        "character_id": character_id,
                        "character": f"Character {index}",
                        "language": "Japanese",
                    }
                ],
            }
        )
    catalog.append(
        {
            **CATALOG[0],
            "id": 999,
            "title": "Unrelated Perfect Match",
            "score": 10.0,
            "voice_actors": [{"id": 900, "name": "Different, Actor", "language": "Japanese"}],
            "voice_actor_roles": [
                {
                    "voice_actor_id": 900,
                    "voice_actor": "Different, Actor",
                    "character_id": 9990,
                    "character": "Someone Else",
                    "language": "Japanese",
                }
            ],
        }
    )
    return catalog


class FakeOllamaClient:
    model = "fake"
    base_url = "http://fake"

    def is_available(self) -> bool:
        return True

    def chat(self, messages: list[dict[str, str]]) -> str:
        return "This fake response intentionally ignores the catalog."


class StructuredOllamaClient:
    model = "fake-qwen"
    base_url = "http://fake"

    def __init__(self, *intents: dict[str, object]):
        self.intents = list(intents)

    def is_available(self) -> bool:
        return True

    def chat(self, messages: list[dict[str, str]]) -> str:
        system = messages[0].get("content", "") if messages else ""
        if "intent parser and tool planner" in system:
            if not self.intents:
                return "{}"
            return json.dumps(self.intents.pop(0))
        return "This controlled response intentionally fails final-answer validation."


class OfflineOllamaClient:
    model = "offline"
    base_url = "http://offline"

    def is_available(self) -> bool:
        return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("Offline client must not be called")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DataPipelineTests(unittest.TestCase):
    def test_label_and_metadata_token_normalization(self) -> None:
        self.assertEqual(clean_label("Action Action"), "Action")
        self.assertEqual(metadata_token("theme", "Isekai Isekai"), "theme_isekai")
        self.assertEqual(metadata_token("demographic", "Shounen Shounen"), "demographic_shounen")
        self.assertIsNone(normalize_character_role("Director"))
        self.assertEqual(selected_staff_role("Producer, Director, Script"), "Director")
        self.assertEqual(creator_roles("Producer, Director, Script"), ["director", "script"])

    def test_build_catalog_dedupes_and_joins_entities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir)
            write_csv(
                raw / "anime.csv",
                [
                    {
                        "anime_id": 10,
                        "title": "Dream Stage",
                        "score": "8.1",
                        "rank": "12",
                        "popularity": "30",
                        "members": "5000",
                        "synopsis": "Students chase a creative dream.",
                        "start_date": "2021-04-01",
                        "end_date": "",
                        "type": "TV",
                        "episodes": "12",
                        "image_url": "",
                    },
                    {
                        "anime_id": 10,
                        "title": "Duplicate Dream Stage",
                        "score": "7.0",
                        "rank": "90",
                        "popularity": "80",
                        "members": "10",
                        "synopsis": "Duplicate row.",
                        "start_date": "2021-04-01",
                        "end_date": "",
                        "type": "TV",
                        "episodes": "12",
                        "image_url": "",
                    },
                ],
                [
                    "anime_id",
                    "title",
                    "score",
                    "rank",
                    "popularity",
                    "members",
                    "synopsis",
                    "start_date",
                    "end_date",
                    "type",
                    "episodes",
                    "image_url",
                ],
            )
            write_csv(
                raw / "anime_genres.csv",
                [
                    {"anime_id": 10, "genre": "Action Action"},
                    {"anime_id": 10, "genre": "Theme::Isekai Isekai"},
                    {"anime_id": 10, "genre": "Demographic::Shounen Shounen"},
                ],
                ["anime_id", "genre"],
            )
            write_csv(
                raw / "entities.csv",
                [
                    {"entity_id": 100, "entity_type": "studio", "name": "Clean Studio", "image_url": ""},
                    {"entity_id": 101, "entity_type": "producer", "name": "Clean Producer", "image_url": ""},
                    {"entity_id": 102, "entity_type": "producer", "name": "None found", "image_url": ""},
                    {"entity_id": 103, "entity_type": "studio", "name": "add some", "image_url": ""},
                    {"entity_id": 200, "entity_type": "person", "name": "Example Director", "image_url": ""},
                    {"entity_id": 300, "entity_type": "character", "name": "Main Hero", "image_url": ""},
                    {"entity_id": 400, "entity_type": "person", "name": "Voice Person", "image_url": ""},
                ],
                ["entity_id", "entity_type", "name", "image_url"],
            )
            write_csv(
                raw / "anime_companies.csv",
                [
                    {"anime_id": 10, "company_id": 100, "role": "Studio"},
                    {"anime_id": 10, "company_id": 101, "role": "Producer"},
                    {"anime_id": 10, "company_id": 102, "role": "Producer"},
                    {"anime_id": 10, "company_id": 103, "role": "Studio"},
                ],
                ["anime_id", "company_id", "role"],
            )
            write_csv(
                raw / "anime_staff.csv",
                [{"anime_id": 10, "person_id": 200, "role": "Director"}],
                ["anime_id", "person_id", "role"],
            )
            write_csv(
                raw / "anime_characters.csv",
                [
                    {"anime_id": 10, "character_id": 300, "role": "Main"},
                    {"anime_id": 10, "character_id": 200, "role": "Director"},
                ],
                ["anime_id", "character_id", "role"],
            )
            write_csv(
                raw / "anime_voice_actors.csv",
                [{"character_id": 300, "person_id": 400, "language": "Japanese"}],
                ["character_id", "person_id", "language"],
            )

            catalog = build_catalog(raw)

        self.assertEqual(len(catalog), 1)
        item = catalog[0]
        self.assertEqual(item["title"], "Dream Stage")
        self.assertEqual(item["start_year"], 2021)
        self.assertIn("theme_isekai", item["metadata_tokens"])
        self.assertIn("demographic_shounen", item["metadata_tokens"])
        self.assertEqual(item["studios"], ["Clean Studio"])
        self.assertEqual(item["studio_relationships"], [{"id": 100, "name": "Clean Studio", "role": "Studio"}])
        self.assertEqual(item["producers"], ["Clean Producer"])
        self.assertEqual(
            item["producer_relationships"],
            [{"id": 101, "name": "Clean Producer", "role": "Producer"}],
        )
        self.assertEqual(item["creators"][0]["name"], "Example Director")
        self.assertEqual([character["name"] for character in item["characters"]], ["Main Hero"])
        self.assertEqual(item["voice_actors"][0]["name"], "Voice Person")
        self.assertEqual(
            item["voice_actor_roles"],
            [
                {
                    "voice_actor_id": 400,
                    "voice_actor": "Voice Person",
                    "character_id": 300,
                    "character": "Main Hero",
                    "language": "Japanese",
                }
            ],
        )

    def test_processed_catalog_uses_compact_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "anime_catalog.json"
            write_catalog([CATALOG[0]], output)
            content = output.read_text(encoding="utf-8")

        self.assertNotIn("\n", content)
        self.assertEqual(json.loads(content)[0]["title"], "Space Quest")


class SessionTests(unittest.TestCase):
    def test_session_store_replace_and_merge_profiles(self) -> None:
        store = SessionStore()
        store.update("abc", {"liked_titles": ["Space Quest"], "excluded_genres": ["Mecha"]})
        replaced = store.update("abc", {"replace": True, "liked_titles": ["Quiet Kitchen"]})

        self.assertEqual(replaced["liked_titles"], ["Quiet Kitchen"])
        self.assertEqual(replaced["excluded_genres"], [])

        merged = merge_profiles(replaced, {"seen_titles": ["Space Quest"], "liked_titles": ["Quiet Kitchen"]})
        self.assertEqual(merged["liked_titles"], ["Quiet Kitchen"])
        self.assertEqual(merged["seen_titles"], ["Space Quest"])


class AnimeRecommenderTests(unittest.TestCase):
    def test_series_key_groups_colon_subtitles(self) -> None:
        self.assertEqual(
            series_key("Sasaki to Miyano: Koi ni Kizuku Mae no Chotto Shita Hanashi."),
            series_key("Sasaki to Miyano"),
        )

    def test_search_finds_title_matches(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        results = recommender.search("space quest", limit=2)
        self.assertEqual(results[0]["id"], 1)

    def test_search_respects_filters(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        results = recommender.search("space", media_type="Movie", limit=10)
        self.assertEqual([item["id"] for item in results], [3])

    def test_recommend_excludes_liked_titles(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        results = recommender.recommend(liked_ids=[1], limit=3)
        ids = [item["id"] for item in results]
        self.assertNotIn(1, ids)
        self.assertEqual(ids[0], 3)

    def test_recommend_excludes_rejected_titles(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        results = recommender.recommend(
            query="relationship between main characters",
            excluded_titles=["Stage Hearts"],
            limit=5,
        )
        ids = [item["id"] for item in results]
        self.assertNotIn(4, ids)

    def test_excluded_title_blocks_same_series_entries(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        results = recommender.recommend(
            include_genres=["Supernatural"],
            excluded_titles=["Death Note"],
            limit=5,
        )
        titles = [item["title"] for item in results]

        self.assertNotIn("Death Note", titles)
        self.assertNotIn("Death Note: Rewrite", titles)
        self.assertIn("Spirit Garden", titles)

    def test_filters_by_type_and_genre(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        results = recommender.recommend(genres=["Action"], media_type="Movie", limit=3)
        self.assertEqual([item["id"] for item in results], [3])

    def test_recommend_uses_story_similarity_from_synopsis(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        results = recommender.recommend(query="relationship between main characters", limit=3)
        self.assertEqual(results[0]["id"], 4)
        self.assertTrue(any("story themes" in reason for reason in results[0]["reasons"]))

    def test_recommend_exposes_hybrid_score_breakdown(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        results = recommender.recommend(query="relationship between main characters", limit=3)
        scores = results[0]["hybrid_scores"]

        self.assertIn("dense", scores)
        self.assertIn("synopsis", scores)
        self.assertIn("metadata", scores)
        self.assertIn("lsa", scores)
        self.assertIn("creator", scores)
        self.assertIn("quality", scores)
        self.assertIn("session", scores)
        self.assertIn("final", scores)
        self.assertGreaterEqual(scores["quality"], 0)

    def test_recommend_builds_session_history_once_without_changing_scores(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        liked_ids = [1, 2, 4]
        profile = {
            "preferred_genres": ["Mystery"],
            "temporary_ratings": {"Future Mystery": 0.75},
        }
        history = recommender._session_history(liked_ids)
        for item in CATALOG:
            original = recommender._session_score(
                item,
                liked_ids,
                profile,
                {"mystery"},
                set(),
            )
            precomputed = recommender._session_score(
                item,
                liked_ids,
                profile,
                {"mystery"},
                set(),
                history=history,
            )
            self.assertEqual(precomputed, original)

        calls = 0
        original_builder = recommender._session_history

        def tracked_builder(values: list[int]):
            nonlocal calls
            calls += 1
            return original_builder(values)

        recommender._session_history = tracked_builder  # type: ignore[method-assign]
        recommender.recommend(liked_ids=liked_ids, session_profile=profile, limit=5)
        self.assertEqual(calls, 1)

    def test_ranking_only_path_preserves_filters_scores_and_reranked_ids(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        arguments = {
            "liked_ids": [1, 2, 4],
            "excluded_ids": [6],
            "session_profile": {"preferred_genres": ["Drama"]},
            "diversity_strength": 0.18,
            "exclude_related_series": False,
            "limit": 6,
        }
        full = recommender.recommend(**arguments)
        ranking_only = recommender.recommend(**arguments, include_explanations=False)

        self.assertEqual([item["id"] for item in ranking_only], [item["id"] for item in full])
        self.assertTrue(all(set(item) == {"id"} for item in ranking_only))

    def test_recommend_uses_embedding_and_svd_channels_for_liked_title(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        results = recommender.recommend(liked_titles=["Space Quest"], limit=3)
        movie = next(item for item in results if item["title"] == "Space Quest Movie")

        self.assertGreater(movie["hybrid_scores"]["dense"], 0)
        self.assertGreater(movie["hybrid_scores"]["lsa"], 0)

    def test_agent_excludes_titles_rejected_from_previous_answer(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())
        history = [
            {
                "role": "user",
                "content": "Recommend anime about relationships between main characters.",
            },
            {
                "role": "assistant",
                "content": (
                    "- Stage Hearts (TV, score 7.60): Close friendship and romance.\n"
                    "- Stage Battles (TV, score 9.10): School drama."
                ),
            },
        ]

        response = agent.respond(
            "I don't like these recommendations. Give me different ones.",
            history=history,
        )

        trace = response["trace"][0]
        result_titles = [item["title"] for item in trace["result"]["results"]]
        self.assertEqual(trace["arguments"]["excluded_titles"], ["Stage Hearts", "Stage Battles"])
        self.assertNotIn("Stage Hearts", result_titles)
        self.assertNotIn("Stage Battles", result_titles)
        self.assertIn("avoid", response["answer"].casefold())

    def test_agent_respects_requested_recommendation_count(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())

        response = agent.respond("Give me 4 anime like Space Quest.")

        trace = response["trace"][0]
        answer_lines = [line for line in response["answer"].splitlines() if line.startswith("- ")]
        self.assertEqual(trace["arguments"]["top_k"], 4)
        self.assertEqual(len(answer_lines), 4)

    def test_agent_routes_more_options_followup_through_catalog(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())
        history = [
            {"role": "user", "content": "I like watching Space Quest. Give me 3 other anime."},
            {
                "role": "assistant",
                "content": "- Space Quest Movie (Movie, score 8.40): Same crew and genre.",
            },
        ]

        response = agent.respond("Give me 3 more options.", history=history)

        self.assertEqual(response["trace"][0]["tool"], "recommend_anime")
        self.assertEqual(response["trace"][0]["arguments"]["top_k"], 3)
        self.assertIn("Space Quest", response["trace"][0]["arguments"]["query"])

    def test_agent_parses_min_score_followup(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())
        history = [
            {
                "role": "user",
                "content": "Give me 5 anime. I like watching Drama and romance.",
            },
            {
                "role": "assistant",
                "content": "- Stage Hearts (TV, score 7.60): School romance.",
            },
        ]

        response = agent.respond("I want the score is at least 8 or above.", history=history)

        trace = response["trace"][0]
        self.assertEqual(trace["arguments"]["min_score"], 8.0)
        self.assertTrue(all(item["score"] >= 8.0 for item in trace["result"]["results"]))

    def test_agent_excludes_same_series_feedback(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())
        history = [
            {"role": "user", "content": "Give me 5 school drama anime."},
            {
                "role": "assistant",
                "content": (
                    "- Stage Hearts (TV, score 7.60): School romance.\n"
                    "- Stage Battles (TV, score 9.10): School rivalry."
                ),
            },
        ]

        response = agent.respond("They are from the same series. Give me 3 more.", history=history)

        excluded = response["trace"][0]["arguments"]["excluded_titles"]
        result_titles = [item["title"] for item in response["trace"][0]["result"]["results"]]
        self.assertIn("Stage Hearts", excluded)
        self.assertIn("Stage Battles", excluded)
        self.assertNotIn("Stage Hearts", result_titles)
        self.assertNotIn("Stage Battles", result_titles)

    def test_agent_handles_romance_typo_without_search_seed(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())

        response = agent.respond("recommend 4 rommance anime")

        trace = response["trace"][0]
        self.assertEqual(trace["arguments"]["reference_titles"], [])
        self.assertIn("Romance", trace["arguments"]["include_genres"])
        self.assertTrue(all("Romance" in item["genres"] for item in trace["result"]["results"]))

    def test_agent_hard_excludes_negative_title_series_inside_request(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())

        response = agent.respond("Recommend 5 Supernatural anime to me. But I don't want Death Note.")

        trace = response["trace"][0]
        titles = [item["title"] for item in trace["result"]["results"]]
        self.assertIn("Death Note", trace["arguments"]["excluded_titles"])
        self.assertNotIn("Death Note", titles)
        self.assertNotIn("Death Note: Rewrite", titles)

    def test_agent_uses_all_watched_examples_and_hides_model_jargon(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        store = SessionStore()
        agent = AnimeAgent(
            recommender,
            client=FakeOllamaClient(),
            get_session_profile=store.get,
            update_session_preferences=store.update,
        )

        response = agent.respond(
            "I really enjoy watching Space Quest and Stage Hearts. Recommend 3 anime.",
            session_id="watched-examples",
        )

        recommendation_trace = response["trace"][0]
        reference_titles = recommendation_trace["arguments"]["reference_titles"]
        result_titles = [item["title"] for item in recommendation_trace["result"]["results"]]
        profile = store.get("watched-examples")

        self.assertEqual(set(reference_titles), {"Space Quest", "Stage Hearts"})
        self.assertNotIn("Space Quest", result_titles)
        self.assertNotIn("Stage Hearts", result_titles)
        self.assertEqual(set(profile["liked_titles"]), {"Space Quest", "Stage Hearts"})
        self.assertEqual(set(profile["seen_titles"]), {"Space Quest", "Stage Hearts"})
        self.assertNotRegex(
            response["answer"].casefold(),
            r"dense text|embedding|tf-idf|latent semantic|metadata similarity|session profile",
        )
        self.assertIn("Connects to your examples", response["answer"])

    def test_agent_introduction_is_grounded_in_catalog_details(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())

        response = agent.respond("Tell me about Space Quest.")

        self.assertEqual(response["mode"], "catalog_introduction")
        self.assertEqual(response["trace"][0]["tool"], "get_anime_details")
        self.assertEqual(response["trace"][0]["arguments"]["anime_id"], 1)
        self.assertIn("Space Quest", response["answer"])
        self.assertIn("crew travels across space", response["answer"])
        self.assertIn("North Studio", response["answer"])

    def test_agent_updates_session_preferences_without_recommendation_request(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        store = SessionStore()
        agent = AnimeAgent(
            recommender,
            client=FakeOllamaClient(),
            get_session_profile=store.get,
            update_session_preferences=store.update,
        )

        response = agent.respond("I do not like mystery.", session_id="abc")

        self.assertEqual(response["mode"], "session_update")
        self.assertEqual(response["trace"][0]["tool"], "update_session_preferences")
        self.assertIn("Mystery", store.get("abc")["excluded_genres"])

    def test_agent_parses_min_year_filter(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())

        response = agent.respond("Give me 3 mystery anime from 2020 or later.")

        trace = response["trace"][0]
        self.assertEqual(trace["arguments"]["min_year"], 2020)
        self.assertTrue(all(item["start_year"] >= 2020 for item in trace["result"]["results"]))

    def test_agent_parses_max_episode_filter(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())

        response = agent.respond("I only watch short series, 12 episodes or fewer. Recommend 4 mystery anime.")

        trace = response["trace"][0]
        self.assertEqual(trace["arguments"]["max_episodes"], 12)
        self.assertTrue(all(item["episodes"] <= 12 for item in trace["result"]["results"]))

    def test_agent_enforces_one_per_series(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())

        response = agent.respond("Recommend 3 anime like Space Quest. Only one title from each series.")

        trace = response["trace"][0]
        result_titles = [item["title"] for item in trace["result"]["results"]]
        self.assertTrue(trace["arguments"]["one_per_series"])
        self.assertNotIn("Space Quest Movie", result_titles)

    def test_one_per_series_preference_does_not_change_requested_limit(self) -> None:
        recommender = AnimeRecommender(CATALOG)
        agent = AnimeAgent(recommender, client=FakeOllamaClient())
        history = [
            {"role": "user", "content": "recommend 7 romance anime"},
            {"role": "assistant", "content": "- Stage Hearts (TV, score 7.60): Romance."},
            {
                "role": "user",
                "content": "Only recommend one anime from each Serie.",
            },
            {"role": "assistant", "content": "- Quiet Kitchen (TV, score 8.20): Slice of life."},
        ]

        response = agent.respond("Give me a new batch, but only titles from 2020 or later", history=history)

        trace = response["trace"][0]
        self.assertEqual(trace["arguments"]["top_k"], 7)
        self.assertEqual(trace["arguments"]["reference_titles"], [])
        self.assertTrue(trace["arguments"]["one_per_series"])
        self.assertEqual(trace["arguments"]["min_year"], 2020)


class StructuredIntentAndFlowTests(unittest.TestCase):
    def test_controlled_qwen_paraphrases_validate_to_equivalent_intents(self) -> None:
        groups = [
            (
                ("under 24 episodes", "no more than 24 episodes", "keep it below 24 episodes"),
                {"intent": "recommend", "max_episodes": 24},
                lambda value: value.max_episodes == 24,
            ),
            (
                ("no mecha", "avoid robot shows", "anything except mecha"),
                {"intent": "recommend", "exclude_genres": ["Mecha"]},
                lambda value: value.exclude_genres == ["Mecha"],
            ),
            (
                ("something like Death Note", "similar to Death Note", "another show with the feel of Death Note"),
                {"intent": "recommend", "reference_titles": ["Death Note"]},
                lambda value: value.reference_titles == ["Death Note"],
            ),
            (
                ("supernatural anime", "anime involving ghosts and spirits", "occult-themed anime"),
                {
                    "intent": "recommend",
                    "include_genres": ["Supernatural"],
                    "free_text_preferences": "ghosts spirits occult",
                },
                lambda value: value.include_genres == ["Supernatural"] and bool(value.free_text_preferences),
            ),
            (
                ("I dislike this", "this was not for me", "do not show me things like this"),
                {"intent": "update_preferences"},
                lambda value: value.intent == "update_preferences",
            ),
        ]
        for messages, payload, assertion in groups:
            for message in messages:
                with self.subTest(message=message):
                    catalog = CATALOG
                    if "Mecha" in payload.get("exclude_genres", []):
                        catalog = [{**CATALOG[0], "genres": CATALOG[0]["genres"] + ["Mecha"]}] + CATALOG[1:]
                    agent = AnimeAgent(AnimeRecommender(catalog), client=StructuredOllamaClient(payload))
                    intent, mode, error = agent._parse_intent(message, [], {})
                    self.assertEqual(mode, "qwen_structured")
                    self.assertIsNone(error)
                    self.assertTrue(assertion(intent))

    def test_qwen_structured_parser_is_primary_and_debuggable(self) -> None:
        payload = {
            "intent": "recommend",
            "reference_titles": ["Death Note"],
            "free_text_preferences": "similar psychological atmosphere",
            "top_k": 3,
        }
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=StructuredOllamaClient(payload))

        response = agent.respond(
            "I enjoyed Death Note and want another show with a similar atmosphere.",
            debug=True,
        )

        self.assertEqual(response["debug"]["parser_mode"], "qwen_structured")
        self.assertEqual(response["debug"]["validated_intent"]["intent"], "recommend")
        self.assertIn("Death Note", response["trace"][0]["arguments"]["reference_titles"])
        self.assertNotIn("validated_intent", response["answer"])

    def test_malformed_qwen_output_uses_rule_fallback(self) -> None:
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=FakeOllamaClient())
        response = agent.respond("Recommend 3 supernatural anime.", debug=True)

        self.assertEqual(response["debug"]["parser_mode"], "rule_fallback")
        self.assertIn("Supernatural", response["debug"]["validated_intent"]["include_genres"])

    def test_structured_tool_failure_retries_with_rule_fallback(self) -> None:
        payload = {"intent": "recommend", "include_genres": ["Supernatural"], "top_k": 2}
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=StructuredOllamaClient(payload))
        original = agent._respond_from_intent
        calls = 0

        def flaky(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("simulated tool validation failure")
            return original(*args, **kwargs)

        agent._respond_from_intent = flaky
        response = agent.respond("Recommend two supernatural anime.", debug=True)

        self.assertEqual(response["debug"]["parser_mode"], "rule_fallback")
        self.assertIn("tool orchestration failed", response["debug"]["parser_error"].casefold())
        self.assertEqual(response["trace"][0]["tool"], "recommend_anime")

    def test_offline_ollama_uses_rule_fallback_without_crashing(self) -> None:
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=OfflineOllamaClient())
        response = agent.respond("Show me anime involving ghosts and spirits.", debug=True)

        self.assertEqual(response["debug"]["parser_mode"], "rule_fallback")
        self.assertEqual(response["trace"][0]["tool"], "recommend_anime")

    def test_rule_fallback_short_paraphrases_preserve_explicit_vs_inferred(self) -> None:
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=OfflineOllamaClient())
        explicit = (
            "Recommend anime under 24 episodes",
            "Recommend anime with no more than 24 episodes",
            "Recommend anime; keep it below 24 episodes",
        )
        vague = (
            "Recommend something short",
            "Recommend something not too long",
            "Recommend a show I can finish over a weekend",
        )

        for message in explicit:
            with self.subTest(message=message):
                intent = agent._rule_based_intent(message, [])
                self.assertIsNotNone(intent.max_episodes)
                self.assertFalse(intent.inferred_constraints)
        for message in vague:
            with self.subTest(message=message):
                intent = agent._rule_based_intent(message, [])
                self.assertIsNone(intent.max_episodes)
                self.assertEqual(intent.inferred_constraints[0].field, "max_episodes")

    def test_rule_fallback_excluded_genre_paraphrases(self) -> None:
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=OfflineOllamaClient())
        messages = (
            "Recommend anime with no mecha",
            "Recommend anime but avoid robot shows",
            "Recommend anime; I am not interested in giant robots",
            "Recommend anything except mecha",
        )
        for message in messages:
            with self.subTest(message=message):
                intent = agent._rule_based_intent(message, [])
                self.assertIn("Mecha", intent.exclude_genres)

    def test_rule_fallback_similar_title_paraphrases(self) -> None:
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=OfflineOllamaClient())
        messages = (
            "Recommend something like Death Note",
            "Recommend something similar to Death Note",
            "I enjoyed Death Note and want the same kind of atmosphere",
            "Give me another show with the feel of Death Note",
        )
        for message in messages:
            with self.subTest(message=message):
                intent = agent._rule_based_intent(message, [])
                self.assertEqual(intent.intent, "recommend")
                self.assertIn("Death Note", intent.reference_titles)

    def test_rule_fallback_supernatural_paraphrases(self) -> None:
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=OfflineOllamaClient())
        messages = (
            "Recommend supernatural anime",
            "Show me anime involving ghosts and spirits",
            "Recommend something about curses or the paranormal",
            "Find occult-themed anime",
        )
        for message in messages:
            with self.subTest(message=message):
                intent = agent._rule_based_intent(message, [])
                self.assertIn("Supernatural", intent.include_genres)

    def test_rule_fallback_negative_feedback_paraphrases(self) -> None:
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=OfflineOllamaClient())
        messages = (
            "I dislike this",
            "This was not for me",
            "Do not show me things like this",
            "I am not interested in this kind of anime",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(agent._rule_based_intent(message, []).intent, "update_preferences")

    def test_entity_resolver_handles_reordered_names_and_fuzzy_titles(self) -> None:
        catalog = [
            {
                **CATALOG[8],
                "characters": [{"id": 100, "name": "Yagami, Light", "role": "Main"}],
                "staff": [],
                "voice_actors": [],
            },
            {**CATALOG[0], "title": "Attack on Titan", "id": 90},
        ]
        resolver = EntityResolver(catalog)

        character = resolver.resolve("Light Yagami", "character")
        title = resolver.resolve("Atack on Titan", "anime")

        self.assertEqual(character["matched_name"], "Yagami, Light")
        self.assertEqual(character["related_anime"][0]["title"], "Death Note")
        self.assertEqual(title["matched_name"], "Attack on Titan")
        self.assertEqual(title["resolution_method"], "fuzzy_title")
        self.assertGreater(title["confidence"], 0.8)

    def test_voice_actor_resolver_supports_equivalent_name_forms(self) -> None:
        resolver = EntityResolver(voice_actor_catalog())
        name_forms = (
            "Matsuoka, Yoshitsugu",
            "Yoshitsugu Matsuoka",
            "matsuoka, yoshitsugu",
            "  MATSUOKA...   YOSHITSUGU  ",
        )

        for name in name_forms:
            with self.subTest(name=name):
                match = resolver.resolve(name, "voice_actor")
                self.assertIsNotNone(match)
                self.assertFalse(match["ambiguous"])
                self.assertEqual(match["entity_id"], 642)
                self.assertEqual(match["matched_name"], "Matsuoka, Yoshitsugu")

    def test_voice_actor_resolver_reports_ambiguous_names(self) -> None:
        catalog = [
            {
                **voice_actor_catalog()[0],
                "id": 301,
                "title": "First Namesake Work",
                "voice_actor_roles": [
                    {
                        "voice_actor_id": 642,
                        "voice_actor": "Matsuoka, Yoshitsugu",
                        "character_id": 3001,
                        "character": "First Character",
                        "language": "Japanese",
                    }
                ],
                "voice_actors": [{"id": 642, "name": "Matsuoka, Yoshitsugu", "language": "Japanese"}],
            },
            {
                **voice_actor_catalog()[1],
                "id": 302,
                "title": "Second Namesake Work",
                "voice_actor_roles": [
                    {
                        "voice_actor_id": 643,
                        "voice_actor": "Matsuoka, Yoshitsugu",
                        "character_id": 3002,
                        "character": "Second Character",
                        "language": "Japanese",
                    }
                ],
                "voice_actors": [{"id": 643, "name": "Matsuoka, Yoshitsugu", "language": "Japanese"}],
            },
        ]

        match = EntityResolver(catalog).resolve("Yoshitsugu Matsuoka", "voice_actor")

        self.assertTrue(match["ambiguous"])
        self.assertEqual({item["entity_id"] for item in match["alternatives"]}, {642, 643})

    def test_required_voice_actor_is_a_hard_filter_with_verified_evidence(self) -> None:
        diagnostics: dict[str, object] = {}
        recommendations = AnimeRecommender(voice_actor_catalog()).recommend(
            required_voice_actors=["Yoshitsugu Matsuoka"],
            query="perfect space action from an unrelated studio",
            top_k=7,
            diagnostics=diagnostics,
        )

        self.assertEqual(len(recommendations), 7)
        self.assertNotIn("Unrelated Perfect Match", [result["title"] for result in recommendations])
        self.assertTrue(all("Matsuoka, Yoshitsugu" in result["matched_voice_actors"] for result in recommendations))
        self.assertTrue(
            all(
                result["voice_actor_roles"][0]["language"] == "Japanese"
                and result["voice_actor_roles"][0]["character"].startswith("Character")
                for result in recommendations
            )
        )
        self.assertTrue(all("Matsuoka, Yoshitsugu" in result["reasons"][0] for result in recommendations))
        self.assertEqual(diagnostics["candidate_count_before_filter"], 9)
        self.assertEqual(diagnostics["candidate_count_after_voice_actor_filter"], 8)

    def test_unknown_voice_actor_returns_no_unverified_recommendations(self) -> None:
        payload = {
            "intent": "recommend",
            "required_voice_actors": ["Missing Voice Actor"],
            "entity_mentions": [
                {"text": "Missing Voice Actor", "entity_type": "voice_actor", "relation": "direct", "index": None}
            ],
            "top_k": 7,
        }
        agent = AnimeAgent(AnimeRecommender(voice_actor_catalog()), client=StructuredOllamaClient(payload))

        response = agent.respond(
            "Recommend 7 anime that have the voice actor Missing Voice Actor involved.",
            debug=True,
        )

        self.assertEqual(response["mode"], "catalog_constraint_error")
        self.assertEqual(response["selected_tool"], "search_entities")
        self.assertEqual(response["candidate_count_after_voice_actor_filter"], 0)
        self.assertFalse(any(step["tool"] == "recommend_anime" for step in response["trace"]))

    def test_required_voice_actor_ignores_previous_session_references(self) -> None:
        payload = {
            "intent": "recommend",
            "reference_titles": ["Unrelated Perfect Match"],
            "required_voice_actors": ["Matsuoka, Yoshitsugu"],
            "entity_mentions": [
                {"text": "Matsuoka, Yoshitsugu", "entity_type": "voice_actor", "relation": "direct", "index": None}
            ],
            "top_k": 7,
        }
        store = SessionStore()
        store.update(
            "voice-session",
            {
                "liked_titles": ["Unrelated Perfect Match"],
                "preferred_genres": ["Sci-Fi"],
                "last_recommendations": ["Unrelated Perfect Match"],
                "last_recommendation_intent": {"reference_titles": ["Unrelated Perfect Match"]},
            },
        )
        agent = AnimeAgent(
            AnimeRecommender(voice_actor_catalog()),
            client=StructuredOllamaClient(payload),
            get_session_profile=store.get,
            update_session_preferences=store.update,
        )

        response = agent.respond(
            "I really like a voice actor called Matsuoka, Yoshitsugu. Could you recommend 7 anime that have him involved?",
            session_id="voice-session",
            debug=True,
        )
        recommendation_step = next(step for step in response["trace"] if step["tool"] == "recommend_anime")

        self.assertEqual(recommendation_step["arguments"]["reference_titles"], [])
        self.assertTrue(recommendation_step["arguments"]["ignore_session_preferences"])
        self.assertIn("liked_titles", response["ignored_session_fields"])
        self.assertNotIn(
            "Unrelated Perfect Match",
            [result["title"] for result in recommendation_step["result"]["results"]],
        )

    def test_original_voice_actor_request_end_to_end(self) -> None:
        payload = {
            "intent": "recommend",
            "required_voice_actors": ["Matsuoka, Yoshitsugu"],
            "entity_mentions": [
                {"text": "Matsuoka, Yoshitsugu", "entity_type": "voice_actor", "relation": "direct", "index": None}
            ],
            "top_k": 7,
        }
        agent = AnimeAgent(AnimeRecommender(voice_actor_catalog()), client=StructuredOllamaClient(payload))

        response = agent.respond(
            "I really like a voice actor called Matsuoka, Yoshitsugu. Could you recommend 7 anime that have him involved?",
            debug=True,
        )
        recommendations = response["trace"][1]["result"]["results"]

        self.assertEqual([step["tool"] for step in response["trace"][:2]], ["search_entities", "recommend_anime"])
        self.assertEqual(response["resolved_entity_id"], 642)
        self.assertEqual(response["entity_type"], "voice_actor")
        self.assertEqual(len(recommendations), 7)
        self.assertTrue(all("Matsuoka, Yoshitsugu" in result["matched_voice_actors"] for result in recommendations))
        self.assertIn("Matsuoka, Yoshitsugu", response["answer"])

    def test_original_voice_actor_request_uses_hard_constraint_in_rule_fallback(self) -> None:
        agent = AnimeAgent(AnimeRecommender(voice_actor_catalog()), client=OfflineOllamaClient())

        response = agent.respond(
            "I really like a voice actor called Matsuoka, Yoshitsugu. Could you recommend 7 anime that have him involved?",
            debug=True,
        )
        recommendation_step = next(step for step in response["trace"] if step["tool"] == "recommend_anime")

        self.assertEqual(response["parser_mode"], "rule_fallback")
        self.assertEqual(response["validated_intent"]["required_voice_actors"], ["Matsuoka, Yoshitsugu"])
        self.assertEqual(recommendation_step["arguments"]["required_voice_actor_ids"], [642])
        self.assertEqual(recommendation_step["arguments"]["free_text_preferences"], "")
        self.assertEqual(len(recommendation_step["result"]["results"]), 7)

    def test_acceptance_a_excludes_franchise_and_uses_hybrid_channels(self) -> None:
        payload = {
            "intent": "recommend",
            "include_genres": ["Supernatural"],
            "excluded_titles": ["Death Note"],
            "exclude_related_series": True,
            "top_k": 10,
        }
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=StructuredOllamaClient(payload))
        response = agent.respond(
            "Recommend 10 supernatural anime, but do not include Death Note or anything from the same series.",
            debug=True,
        )
        results = response["trace"][0]["result"]["results"]
        titles = [item["title"] for item in results]

        self.assertNotIn("Death Note", titles)
        self.assertNotIn("Death Note: Rewrite", titles)
        self.assertTrue(results)
        self.assertIn("metadata", results[0]["active_channels"])
        self.assertNotEqual(results[0]["active_channels"], ["quality"])

    def test_acceptance_b_reference_short_soft_constraint_and_novelty(self) -> None:
        payload = {
            "intent": "recommend",
            "reference_titles": ["Death Note"],
            "novelty_preference": "less_famous",
            "free_text_preferences": "something shorter with a similar atmosphere",
            "inferred_constraints": [
                {"field": "max_episodes", "value": 24, "confidence": 0.72, "source_text": "something shorter"}
            ],
            "top_k": 3,
        }
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=StructuredOllamaClient(payload))
        response = agent.respond("I enjoyed Death Note but want something shorter and less famous.", debug=True)
        intent = response["debug"]["validated_intent"]
        result = response["trace"][0]["result"]["results"][0]

        self.assertIn("Death Note", response["trace"][0]["arguments"]["reference_titles"])
        self.assertEqual(intent["inferred_constraints"][0]["field"], "max_episodes")
        self.assertEqual(intent["novelty_preference"], "less_famous")
        self.assertIn("novelty", result["active_channels"])
        self.assertIn("dense", result["active_channels"])

    def test_enjoyed_reference_is_watched_and_related_entries_are_not_returned(self) -> None:
        store = SessionStore()
        agent = AnimeAgent(
            AnimeRecommender(CATALOG),
            client=OfflineOllamaClient(),
            get_session_profile=store.get,
            update_session_preferences=store.update,
        )
        response = agent.respond(
            "I enjoyed Death Note and want something with a similar atmosphere.",
            session_id="enjoyed",
        )
        titles = [item["title"] for item in response["trace"][0]["result"]["results"]]

        self.assertIn("Death Note", store.get("enjoyed")["seen_titles"])
        self.assertNotIn("Death Note", titles)
        self.assertNotIn("Death Note: Rewrite", titles)

    def test_acceptance_c_free_text_ghost_request_activates_content_channels(self) -> None:
        payload = {
            "intent": "recommend",
            "free_text_preferences": "anime involving ghosts and spirits",
            "top_k": 3,
        }
        agent = AnimeAgent(AnimeRecommender(CATALOG), client=StructuredOllamaClient(payload))
        response = agent.respond("Show me anime involving ghosts and spirits.", debug=True)
        result = response["trace"][0]["result"]["results"][0]

        self.assertEqual(response["trace"][0]["arguments"]["include_genres"], ["Supernatural"])
        self.assertIn("synopsis", result["active_channels"])
        self.assertIn("dense", result["active_channels"])

    def test_acceptance_d_director_is_retrieved_from_catalog(self) -> None:
        catalog = [
            {
                **CATALOG[8],
                "id": 80,
                "title": "Monster",
                "staff": [{"id": 501, "name": "Catalog Director", "role": "Director"}],
                "creators": [{"id": 501, "name": "Catalog Director", "role": "Director"}],
            },
            {
                **CATALOG[5],
                "id": 81,
                "title": "Director's Other Work",
                "staff": [{"id": 501, "name": "Catalog Director", "role": "Director"}],
                "creators": [{"id": 501, "name": "Catalog Director", "role": "Director"}],
            },
        ]
        payload = {
            "intent": "recommend",
            "entity_mentions": [{"text": "Monster", "entity_type": "anime", "relation": "director_of", "index": None}],
            "free_text_preferences": "works connected to this director",
            "top_k": 2,
        }
        agent = AnimeAgent(AnimeRecommender(catalog), client=StructuredOllamaClient(payload))
        response = agent.respond("I like the director of Monster.", debug=True)

        arguments = response["trace"][0]["arguments"]
        director_step = next(step for step in response["trace"] if step["tool"] == "get_anime_details")
        self.assertEqual(arguments["preferred_staff"], ["Catalog Director"])
        self.assertEqual(director_step["result"]["directors"], ["Catalog Director"])
        self.assertIn("creator", response["trace"][0]["result"]["results"][0]["active_channels"])

    def test_acceptance_e_previous_result_indices_update_session(self) -> None:
        first = {"intent": "recommend", "free_text_preferences": "varied anime", "top_k": 4, "one_per_series": True}
        second = {
            "intent": "recommend",
            "reference_result_indices": [4],
            "watched_result_indices": [2],
            "free_text_preferences": "more like the selected result",
            "top_k": 3,
        }
        store = SessionStore()
        agent = AnimeAgent(
            AnimeRecommender(CATALOG),
            client=StructuredOllamaClient(first, second),
            get_session_profile=store.get,
            update_session_preferences=store.update,
        )
        agent.respond("Recommend four varied anime.", session_id="indices")
        prior = store.get("indices")["last_recommendations"]
        response = agent.respond(
            "I already watched the second recommendation. Give me more like the fourth one.",
            session_id="indices",
            debug=True,
        )
        arguments = response["trace"][0]["arguments"]
        profile = store.get("indices")

        self.assertIn(prior[1], arguments["seen_titles"])
        self.assertIn(prior[3], arguments["reference_titles"])
        self.assertIn(prior[1], profile["seen_titles"])

    def test_score_breakdown_reconstructs_final_score(self) -> None:
        result = AnimeRecommender(CATALOG).recommend(
            reference_titles=["Death Note"],
            free_text_preferences="dark psychological supernatural mystery",
            novelty_preference="less_famous",
            limit=3,
        )[1]

        contribution_sum = sum(result["weighted_contributions"].values())
        self.assertAlmostEqual(contribution_sum, result["pre_diversity_score"], places=5)
        self.assertAlmostEqual(
            result["pre_diversity_score"] + result["diversity_adjustment"],
            result["final_score"],
            places=5,
        )
        self.assertAlmostEqual(result["final_score"], result["match_score"], places=4)

    def test_quality_only_request_is_explicit_fallback_mode(self) -> None:
        result = AnimeRecommender(CATALOG).recommend(limit=1)[0]
        self.assertEqual(result["recommendation_mode"], "quality_fallback")
        self.assertEqual(result["active_channels"], ["quality"])
        self.assertEqual(result["effective_weights"]["quality"], 1.0)


if __name__ == "__main__":
    unittest.main()
