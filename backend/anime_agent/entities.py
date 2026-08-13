from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

ENTITY_TYPES = {
    "anime",
    "character",
    "staff",
    "director",
    "original_creator",
    "studio",
    "producer",
    "voice_actor",
    "genre",
    "theme",
    "demographic",
}


@dataclass(frozen=True)
class EntityResolution:
    input_text: str
    entity_type: str
    matched_name: str
    entity_id: int | None
    confidence: float
    resolution_method: str
    anime_id: int | None
    related_anime_ids: list[int]
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _EntityRecord:
    entity_type: str
    name: str
    entity_id: int | None
    anime_id: int | None
    related_anime_ids: set[int]
    variants: set[str]


class EntityResolver:
    """Catalog-backed resolver for anime and the people or labels linked to them."""

    def __init__(self, catalog: list[dict[str, Any]]):
        self.anime_titles = {int(item["id"]): str(item.get("title") or "") for item in catalog}
        grouped: dict[tuple[str, str, int | None], _EntityRecord] = {}

        def add(
            entity_type: str,
            name: Any,
            anime_id: int | None = None,
            entity_id: int | None = None,
        ) -> None:
            text = str(name or "").strip()
            if entity_type not in ENTITY_TYPES or not text:
                return
            key = (entity_type, self._key(text), entity_id)
            record = grouped.get(key)
            if record is None:
                record = _EntityRecord(
                    entity_type,
                    text,
                    entity_id,
                    anime_id if entity_type == "anime" else None,
                    set(),
                    set(),
                )
                grouped[key] = record
            if anime_id is not None:
                record.related_anime_ids.add(anime_id)
            record.variants.update(self._name_variants(text, entity_type))

        for item in catalog:
            anime_id = int(item["id"])
            add("anime", item.get("title"), anime_id, anime_id)
            for alias in item.get("aliases", []):
                add("anime", alias, anime_id, anime_id)
            studio_relationships = item.get("studio_relationships", [])
            for company in studio_relationships:
                add("studio", company.get("name"), anime_id, _optional_entity_id(company.get("id")))
            for name in item.get("studios", []):
                add("studio", name, anime_id)
            producer_relationships = item.get("producer_relationships", [])
            for company in producer_relationships:
                add("producer", company.get("name"), anime_id, _optional_entity_id(company.get("id")))
            for company in item.get("producers", []):
                name = company.get("name") if isinstance(company, dict) else company
                add("producer", name, anime_id)
            character_relationships = item.get("character_relationships", item.get("characters", []))
            for person in character_relationships:
                add("character", person.get("name"), anime_id, _optional_entity_id(person.get("id")))
            for name in item.get("character_names", []):
                add("character", name, anime_id)
            staff_relationships = item.get("staff_relationships", item.get("staff", []))
            for person in staff_relationships:
                add("staff", person.get("name"), anime_id, _optional_entity_id(person.get("id")))
                role = str(person.get("role") or "").casefold()
                if "director" in role:
                    add("director", person.get("name"), anime_id, _optional_entity_id(person.get("id")))
                if "original creator" in role or role.strip() == "creator":
                    add("original_creator", person.get("name"), anime_id, _optional_entity_id(person.get("id")))
            for role in item.get("voice_actor_roles", []):
                add(
                    "voice_actor",
                    role.get("voice_actor"),
                    anime_id,
                    _optional_entity_id(role.get("voice_actor_id")),
                )
            for person in item.get("voice_actors", []):
                add("voice_actor", person.get("name"), anime_id, _optional_entity_id(person.get("id")))
            for name in item.get("genres", []):
                add("genre", name, anime_id)
            for group, names in item.get("genre_groups", {}).items():
                entity_type = {
                    "themes": "theme",
                    "theme": "theme",
                    "demographics": "demographic",
                    "demographic": "demographic",
                }.get(str(group).casefold())
                if entity_type:
                    for name in names:
                        add(entity_type, name, anime_id)

        identified_by_name: dict[tuple[str, str], list[_EntityRecord]] = defaultdict(list)
        anonymous_keys: list[tuple[str, str, int | None]] = []
        for key, record in grouped.items():
            if key[2] is None:
                anonymous_keys.append(key)
            else:
                identified_by_name[key[:2]].append(record)

        for anonymous_key in anonymous_keys:
            record = grouped.get(anonymous_key)
            if record is None:
                continue
            identified = identified_by_name.get(anonymous_key[:2], [])
            if len(identified) == 1:
                identified[0].related_anime_ids.update(record.related_anime_ids)
                identified[0].variants.update(record.variants)
                del grouped[anonymous_key]
                continue
            identified_anime_ids = set().union(*(candidate.related_anime_ids for candidate in identified))
            if identified and record.related_anime_ids <= identified_anime_ids:
                del grouped[anonymous_key]

        self.records = list(grouped.values())
        self.by_type: dict[str, list[_EntityRecord]] = defaultdict(list)
        for record in self.records:
            self.by_type[record.entity_type].append(record)

    def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 10,
        minimum_confidence: float = 0.45,
    ) -> list[dict[str, Any]]:
        query_key = self._key(query)
        if not query_key:
            return []
        allowed = {value.casefold() for value in entity_types or ENTITY_TYPES}
        ranked: list[tuple[float, str, _EntityRecord]] = []
        for record in self.records:
            if record.entity_type not in allowed:
                continue
            confidence, method = self._score(query_key, record.variants)
            if confidence >= minimum_confidence:
                ranked.append((confidence, method, record))
        ranked.sort(key=lambda row: (row[0], row[2].entity_type == "anime", -len(row[2].name)), reverse=True)
        return [
            self._public(query, record, confidence, method) for confidence, method, record in ranked[: max(1, limit)]
        ]

    def resolve(
        self,
        query: str,
        entity_type: str | None = None,
        minimum_confidence: float = 0.58,
    ) -> dict[str, Any] | None:
        matches = self.search(
            query,
            entity_types=[entity_type] if entity_type else None,
            limit=10,
            minimum_confidence=minimum_confidence,
        )
        if not matches:
            return None

        best = dict(matches[0])
        tied = [
            match
            for match in matches[1:]
            if abs(float(match["confidence"]) - float(best["confidence"])) < 0.0001
            and match.get("entity_id") != best.get("entity_id")
        ]
        if tied:
            best["ambiguous"] = True
            best["alternatives"] = [
                {
                    "matched_name": match["matched_name"],
                    "entity_id": match.get("entity_id"),
                    "entity_type": match["entity_type"],
                    "confidence": match["confidence"],
                }
                for match in [best, *tied]
            ]
        return best

    def _public(self, query: str, record: _EntityRecord, confidence: float, method: str) -> dict[str, Any]:
        result = EntityResolution(
            input_text=query,
            entity_type=record.entity_type,
            matched_name=record.name,
            entity_id=record.entity_id,
            confidence=round(confidence, 4),
            resolution_method=method,
            anime_id=record.anime_id,
            related_anime_ids=sorted(record.related_anime_ids),
        ).to_dict()
        result["resolved_entity_type"] = record.entity_type
        result["resolved_name"] = record.name
        result["related_anime"] = [
            {"anime_id": anime_id, "title": self.anime_titles.get(anime_id, "")}
            for anime_id in result["related_anime_ids"]
        ]
        if method == "fuzzy" and record.entity_type == "anime":
            result["resolution_method"] = "fuzzy_title"
        return result

    def _score(self, query: str, variants: set[str]) -> tuple[float, str]:
        if query in variants:
            return 1.0, "exact"
        best = (0.0, "fuzzy")
        query_tokens = set(query.split())
        for variant in variants:
            if len(query) >= 4 and (query in variant or variant in query):
                coverage = min(len(query), len(variant)) / max(len(query), len(variant))
                score = 0.82 + 0.14 * coverage
                if score > best[0]:
                    best = (score, "substring")
            variant_tokens = set(variant.split())
            union = query_tokens | variant_tokens
            token_score = len(query_tokens & variant_tokens) / len(union) if union else 0.0
            sequence_score = SequenceMatcher(None, query, variant).ratio()
            score = max(sequence_score * 0.88, token_score * 0.92)
            if score > best[0]:
                best = (score, "fuzzy")
        return best

    def _name_variants(self, name: str, entity_type: str) -> set[str]:
        variants = {self._key(name)}
        if "," in name:
            last, first = (part.strip() for part in name.split(",", 1))
            variants.add(self._key(f"{first} {last}"))
        elif entity_type in {"staff", "director", "original_creator", "voice_actor"}:
            parts = self._key(name).split()
            if len(parts) == 2:
                variants.add(" ".join(reversed(parts)))
        return {value for value in variants if value}

    @staticmethod
    def _key(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value.casefold())
        plain = "".join(character for character in normalized if not unicodedata.combining(character))
        return " ".join("".join(character if character.isalnum() else " " for character in plain).split())


def _optional_entity_id(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
