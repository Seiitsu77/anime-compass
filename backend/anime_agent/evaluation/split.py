from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import struct
import time
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SPLIT_SCHEMA_VERSION = 3
METHODOLOGY_NOTE = (
    "Because interaction timestamps are unavailable, this evaluation measures preference "
    "reconstruction/generalization under a deterministic user-stratified random holdout, "
    "not chronological next-item prediction."
)
_PAIR = struct.Struct("<IB")


def catalog_ids_sha256(catalog_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for anime_id in sorted({int(value) for value in catalog_ids}):
        digest.update(f"{anime_id}\n".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class FeedbackConfig:
    """Rating-class definitions used by the personalized benchmark.

    Classification is applied in this order: positive, explicit negative,
    neutral, ignored.  The ordering keeps ``positive_threshold=7`` meaningful
    even with the default neutral range of 6--7.  With a threshold of 9, an 8
    is intentionally placed in the ignored class unless the caller changes the
    neutral range.
    """

    positive_threshold: int = 8
    neutral_min: int = 6
    neutral_max: int = 7
    negative_max: int = 5

    def __post_init__(self) -> None:
        if not 1 <= self.positive_threshold <= 10:
            raise ValueError("positive_threshold must be between 1 and 10")
        if not 1 <= self.neutral_min <= self.neutral_max <= 10:
            raise ValueError("neutral range must be within 1..10")
        if not 1 <= self.negative_max <= 10:
            raise ValueError("negative_max must be between 1 and 10")

    def classify(self, rating: int) -> str:
        if not 1 <= rating <= 10:
            raise ValueError(f"rating must be between 1 and 10, got {rating}")
        if rating >= self.positive_threshold:
            return "positive"
        if rating <= self.negative_max:
            return "explicit_negative"
        if self.neutral_min <= rating <= self.neutral_max:
            return "neutral"
        return "ignored"


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    minimum_positives: int = 5
    feedback: FeedbackConfig = FeedbackConfig()

    def __post_init__(self) -> None:
        if self.minimum_positives < 3:
            raise ValueError("minimum_positives must leave room for train, validation, and test")


@dataclass(frozen=True)
class UserSplit:
    user_id: int
    eligible: bool
    train_positive: tuple[tuple[int, int], ...]
    validation_positive: tuple[tuple[int, int], ...]
    test_positive: tuple[tuple[int, int], ...]
    explicit_negative: tuple[tuple[int, int], ...]
    neutral: tuple[tuple[int, int], ...]
    ignored: tuple[tuple[int, int], ...] = ()

    @property
    def train_positive_ids(self) -> tuple[int, ...]:
        return tuple(anime_id for anime_id, _rating in self.train_positive)

    @property
    def validation_positive_ids(self) -> tuple[int, ...]:
        return tuple(anime_id for anime_id, _rating in self.validation_positive)

    @property
    def test_positive_ids(self) -> tuple[int, ...]:
        return tuple(anime_id for anime_id, _rating in self.test_positive)

    @property
    def all_observed_training_ratings(self) -> tuple[tuple[int, int], ...]:
        """Observed ratings available at training time, without held-out positives."""
        return (*self.train_positive, *self.explicit_negative, *self.neutral, *self.ignored)


def holdout_sizes(positive_count: int, minimum_positives: int = 5) -> tuple[int, int]:
    """Return deterministic validation/test counts for one user's positives."""
    if positive_count < minimum_positives:
        return 0, 0
    if positive_count <= 9:
        return 1, 1
    if positive_count <= 19:
        return 1, 2
    validation = max(1, int(positive_count * 0.10))
    test = max(1, int(positive_count * 0.10))
    if validation + test >= positive_count:
        test = 1
        validation = 1
    return validation, test


def split_user_positives(
    user_id: int,
    positives: Sequence[tuple[int, int]],
    config: SplitConfig,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    bool,
]:
    """Split one user's positives without depending on input row order."""
    canonical = sorted((int(anime_id), int(rating)) for anime_id, rating in positives)
    validation_size, test_size = holdout_sizes(len(canonical), config.minimum_positives)
    if not validation_size:
        return tuple(canonical), (), (), False

    stable_seed = int.from_bytes(
        hashlib.blake2b(f"{config.seed}:{user_id}".encode(), digest_size=8).digest(),
        "little",
    )
    indexes = list(range(len(canonical)))
    random.Random(stable_seed).shuffle(indexes)
    validation_indexes = set(indexes[:validation_size])
    test_indexes = set(indexes[validation_size : validation_size + test_size])
    train = tuple(value for index, value in enumerate(canonical) if index not in validation_indexes | test_indexes)
    validation = tuple(canonical[index] for index in sorted(validation_indexes))
    test = tuple(canonical[index] for index in sorted(test_indexes))
    return train, validation, test, True


def _encode_interactions(interactions: Sequence[tuple[int, int]]) -> bytes:
    if not interactions:
        return b""
    payload = bytearray(len(interactions) * _PAIR.size)
    for index, (anime_id, rating) in enumerate(interactions):
        if anime_id < 0 or anime_id > 0xFFFFFFFF:
            raise ValueError(f"anime ID is outside uint32 range: {anime_id}")
        _PAIR.pack_into(payload, index * _PAIR.size, int(anime_id), int(rating))
    return zlib.compress(payload, level=6)


def _decode_interactions(payload: bytes | None) -> tuple[tuple[int, int], ...]:
    if not payload:
        return ()
    raw = zlib.decompress(payload)
    if len(raw) % _PAIR.size:
        raise ValueError("Corrupt interaction blob in split artifact")
    return tuple((anime_id, rating) for anime_id, rating in _PAIR.iter_unpack(raw))


class SplitStore:
    """Read-only interface to a persistent personalized holdout artifact."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def metadata(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value_json FROM metadata ORDER BY key").fetchall()
        metadata = {str(row["key"]): json.loads(str(row["value_json"])) for row in rows}
        if metadata.get("schema_version") != SPLIT_SCHEMA_VERSION:
            raise ValueError("Unsupported personalized split artifact schema")
        return metadata

    def iter_users(self, *, eligible_only: bool = False) -> Iterator[UserSplit]:
        query = "SELECT * FROM user_splits"
        if eligible_only:
            query += " WHERE eligible = 1"
        query += " ORDER BY user_id"
        with self._connect() as connection:
            for row in connection.execute(query):
                yield self._row_to_split(row)

    def iter_users_by_ids(self, user_ids: Sequence[int]) -> Iterator[UserSplit]:
        """Read selected users without decoding every interaction blob.

        SQLite limits the number of bound parameters, so selected IDs are
        queried in deterministic chunks.  Callers should pass sorted IDs when
        they need output aligned across multiple model evaluations.
        """
        requested = [int(user_id) for user_id in user_ids]
        if len(requested) != len(set(requested)):
            raise ValueError("user_ids must be unique")
        if requested != sorted(requested):
            raise ValueError("user_ids must be sorted")
        with self._connect() as connection:
            for start in range(0, len(requested), 900):
                chunk = requested[start : start + 900]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _value in chunk)
                query = f"SELECT * FROM user_splits WHERE user_id IN ({placeholders}) ORDER BY user_id"
                for row in connection.execute(query, chunk):
                    yield self._row_to_split(row)

    def get_user(self, user_id: int) -> UserSplit | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM user_splits WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
        return self._row_to_split(row) if row is not None else None

    def eligible_user_ids(self) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT user_id FROM user_splits WHERE eligible = 1 ORDER BY user_id").fetchall()
        return [int(row[0]) for row in rows]

    def eligible_user_activity(self) -> list[tuple[int, int]]:
        """Return compact (user ID, train-positive count) sampling metadata."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, train_positive_count FROM user_splits WHERE eligible = 1 ORDER BY user_id"
            ).fetchall()
        return [(int(row[0]), int(row[1])) for row in rows]

    def eligible_segment_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN train_positive_count BETWEEN 1 AND 4 THEN 1 ELSE 0 END) AS sparse,
                    SUM(CASE WHEN train_positive_count BETWEEN 5 AND 19 THEN 1 ELSE 0 END) AS medium,
                    SUM(CASE WHEN train_positive_count >= 20 THEN 1 ELSE 0 END) AS heavy
                FROM user_splits
                WHERE eligible = 1
                """
            ).fetchone()
        return {segment: int(row[segment] or 0) for segment in ("sparse", "medium", "heavy")}

    def audit_counts(self) -> dict[str, int | bool]:
        """Reconcile persisted row counts with artifact metadata using SQL."""
        metadata = self.metadata()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS users_before_filter,
                    SUM(eligible) AS users_after_filter,
                    SUM(train_positive_count) AS train_positive_interactions,
                    SUM(validation_positive_count) AS validation_positive_interactions,
                    SUM(test_positive_count) AS test_positive_interactions,
                    SUM(explicit_negative_count) AS explicit_negative_ratings,
                    SUM(neutral_count) AS neutral_ratings,
                    SUM(ignored_count) AS ignored_ratings,
                    SUM(CASE WHEN eligible = 1 AND (
                        train_positive_count < 1 OR validation_positive_count < 1 OR test_positive_count < 1
                    ) THEN 1 ELSE 0 END) AS invalid_eligible_users,
                    SUM(CASE WHEN eligible = 0 AND (
                        validation_positive_count > 0 OR test_positive_count > 0
                    ) THEN 1 ELSE 0 END) AS invalid_ineligible_users
                FROM user_splits
                """
            ).fetchone()
        if row is None:
            raise ValueError("Split artifact contains no aggregate row")
        keys = (
            "users_before_filter",
            "users_after_filter",
            "train_positive_interactions",
            "validation_positive_interactions",
            "test_positive_interactions",
            "explicit_negative_ratings",
            "neutral_ratings",
            "ignored_ratings",
        )
        actual = {key: int(row[key] or 0) for key in keys}
        mismatches = {
            key: {"metadata": int(metadata.get(key, -1)), "stored": actual[key]}
            for key in keys
            if int(metadata.get(key, -1)) != actual[key]
        }
        invalid_eligible = int(row["invalid_eligible_users"] or 0)
        invalid_ineligible = int(row["invalid_ineligible_users"] or 0)
        if mismatches or invalid_eligible or invalid_ineligible:
            raise ValueError(
                "Split artifact count audit failed: "
                f"mismatches={mismatches}, invalid_eligible={invalid_eligible}, "
                f"invalid_ineligible={invalid_ineligible}"
            )
        stored_ratings = sum(
            actual[key]
            for key in (
                "train_positive_interactions",
                "validation_positive_interactions",
                "test_positive_interactions",
                "explicit_negative_ratings",
                "neutral_ratings",
                "ignored_ratings",
            )
        )
        expected_stored = int(metadata["rows_scanned"]) - int(metadata["orphan_rows"])
        if stored_ratings != expected_stored:
            raise ValueError(
                f"Split artifact rating accounting failed: stored={stored_ratings}, expected={expected_stored}"
            )
        return {
            "passed": True,
            "stored_users": actual["users_before_filter"],
            "eligible_users": actual["users_after_filter"],
            "stored_ratings": stored_ratings,
            "orphan_rows": int(metadata["orphan_rows"]),
        }

    @staticmethod
    def _row_to_split(row: sqlite3.Row) -> UserSplit:
        return UserSplit(
            user_id=int(row["user_id"]),
            eligible=bool(row["eligible"]),
            train_positive=_decode_interactions(row["train_positive"]),
            validation_positive=_decode_interactions(row["validation_positive"]),
            test_positive=_decode_interactions(row["test_positive"]),
            explicit_negative=_decode_interactions(row["explicit_negative"]),
            neutral=_decode_interactions(row["neutral"]),
            ignored=_decode_interactions(row["ignored"]),
        )


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE user_splits (
            user_id INTEGER PRIMARY KEY,
            eligible INTEGER NOT NULL,
            positive_count INTEGER NOT NULL,
            train_positive_count INTEGER NOT NULL,
            validation_positive_count INTEGER NOT NULL,
            test_positive_count INTEGER NOT NULL,
            explicit_negative_count INTEGER NOT NULL,
            neutral_count INTEGER NOT NULL,
            ignored_count INTEGER NOT NULL,
            train_positive BLOB NOT NULL,
            validation_positive BLOB NOT NULL,
            test_positive BLOB NOT NULL,
            explicit_negative BLOB NOT NULL,
            neutral BLOB NOT NULL,
            ignored BLOB NOT NULL
        );
        CREATE INDEX ix_user_splits_eligible ON user_splits (eligible, user_id);
        """
    )


def _store_user(
    connection: sqlite3.Connection,
    user_id: int,
    positives: Sequence[tuple[int, int]],
    negatives: Sequence[tuple[int, int]],
    neutrals: Sequence[tuple[int, int]],
    ignored: Sequence[tuple[int, int]],
    config: SplitConfig,
) -> dict[str, int]:
    train, validation, test, eligible = split_user_positives(user_id, positives, config)
    negatives = tuple(sorted(negatives))
    neutrals = tuple(sorted(neutrals))
    ignored = tuple(sorted(ignored))
    connection.execute(
        """
        INSERT INTO user_splits (
            user_id, eligible, positive_count, train_positive_count,
            validation_positive_count, test_positive_count,
            explicit_negative_count, neutral_count, ignored_count,
            train_positive, validation_positive, test_positive,
            explicit_negative, neutral, ignored
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            int(eligible),
            len(positives),
            len(train),
            len(validation),
            len(test),
            len(negatives),
            len(neutrals),
            len(ignored),
            _encode_interactions(train),
            _encode_interactions(validation),
            _encode_interactions(test),
            _encode_interactions(negatives),
            _encode_interactions(neutrals),
            _encode_interactions(ignored),
        ),
    )
    return {
        "users_after_filter": int(eligible),
        "train_positive_interactions": len(train),
        "validation_positive_interactions": len(validation),
        "test_positive_interactions": len(test),
        "explicit_negative_ratings": len(negatives),
        "neutral_ratings": len(neutrals),
        "ignored_ratings": len(ignored),
    }


def build_split_store(
    ratings_path: Path,
    output_path: Path,
    *,
    catalog_ids: set[int],
    config: SplitConfig | None = None,
    source_user_limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stream a user-sorted rating CSV into an atomic SQLite split artifact."""
    config = config or SplitConfig()
    ratings_path = Path(ratings_path)
    output_path = Path(output_path)
    if not ratings_path.exists():
        raise FileNotFoundError(ratings_path)
    if not catalog_ids:
        raise ValueError("catalog_ids cannot be empty")
    if source_user_limit is not None and source_user_limit < 1:
        raise ValueError("source_user_limit must be positive when provided")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    started = time.perf_counter()
    counters = {
        "rows_scanned": 0,
        "users_before_filter": 0,
        "users_after_filter": 0,
        "positive_ratings": 0,
        "train_positive_interactions": 0,
        "validation_positive_interactions": 0,
        "test_positive_interactions": 0,
        "explicit_negative_ratings": 0,
        "neutral_ratings": 0,
        "ignored_ratings": 0,
        "orphan_rows": 0,
    }
    hasher = hashlib.sha256()
    current_user: int | None = None
    positives: list[tuple[int, int]] = []
    negatives: list[tuple[int, int]] = []
    neutrals: list[tuple[int, int]] = []
    ignored: list[tuple[int, int]] = []
    seen_items: set[int] = set()

    connection = sqlite3.connect(temporary_path)
    try:
        _initialize_database(connection)

        def flush_user() -> bool:
            nonlocal current_user
            if current_user is None:
                return True
            if source_user_limit is not None and counters["users_before_filter"] >= source_user_limit:
                return False
            summary = _store_user(
                connection,
                current_user,
                positives,
                negatives,
                neutrals,
                ignored,
                config,
            )
            counters["users_before_filter"] += 1
            for key, value in summary.items():
                counters[key] += value
            if counters["users_before_filter"] % 5_000 == 0:
                connection.commit()
            return True

        with ratings_path.open("rb") as file:
            header = file.readline()
            hasher.update(header)
            if header.rstrip(b"\r\n") != b"user_id,anime_id,rating":
                raise ValueError("Unexpected rating_complete.csv header")
            previous_user = -1
            for line_number, raw_line in enumerate(file, start=2):
                fields = raw_line.rstrip(b"\r\n").split(b",")
                if len(fields) != 3:
                    raise ValueError(f"Malformed rating row at line {line_number}")
                try:
                    user_id, anime_id, rating = (int(value) for value in fields)
                except ValueError as exc:
                    raise ValueError(f"Non-integer rating row at line {line_number}") from exc
                if user_id < previous_user:
                    raise ValueError("rating_complete.csv must be sorted by user_id")
                if current_user is not None and user_id != current_user:
                    if not flush_user():
                        break
                    positives.clear()
                    negatives.clear()
                    neutrals.clear()
                    ignored.clear()
                    seen_items.clear()
                if source_user_limit is not None and counters["users_before_filter"] >= source_user_limit:
                    break
                hasher.update(raw_line)
                current_user = user_id
                previous_user = user_id
                counters["rows_scanned"] += 1
                if anime_id in seen_items:
                    raise ValueError(f"Duplicate user-anime pair for user {user_id}, anime {anime_id}")
                seen_items.add(anime_id)
                if anime_id not in catalog_ids:
                    counters["orphan_rows"] += 1
                    continue
                category = config.feedback.classify(rating)
                interaction = (anime_id, rating)
                if category == "positive":
                    positives.append(interaction)
                    counters["positive_ratings"] += 1
                elif category == "explicit_negative":
                    negatives.append(interaction)
                elif category == "neutral":
                    neutrals.append(interaction)
                else:
                    ignored.append(interaction)
                if progress is not None and counters["rows_scanned"] % 5_000_000 == 0:
                    progress(f"split: scanned {counters['rows_scanned']:,} rating rows")
            else:
                flush_user()

        train_denominator = counters["users_before_filter"] * len(catalog_ids)
        metadata: dict[str, Any] = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "method": "deterministic per-user random positive holdout",
            "methodology_note": METHODOLOGY_NOTE,
            "source_file": ratings_path.name,
            "source_size_bytes": ratings_path.stat().st_size,
            "source_mtime_ns": ratings_path.stat().st_mtime_ns,
            "dataset_sha256": hasher.hexdigest(),
            "dataset_sha256_scope": "full_file" if source_user_limit is None else "header_and_processed_rows",
            "catalog_items": len(catalog_ids),
            "catalog_ids_sha256": catalog_ids_sha256(catalog_ids),
            "split_config": {
                "seed": config.seed,
                "minimum_positives": config.minimum_positives,
                "feedback": asdict(config.feedback),
                "holdout_rules": {
                    "below_configured_minimum": (
                        f"fewer than {config.minimum_positives} positives: excluded from evaluation; "
                        "positives remain training-only"
                    ),
                    "5_to_9": {"validation": 1, "test": 1},
                    "10_to_19": {"validation": 1, "test": 2},
                    "20_plus": "floor(10%) validation and floor(10%) test, minimum one each",
                },
            },
            "source_user_limit": source_user_limit,
            **counters,
            "train_positive_sparsity": (
                counters["train_positive_interactions"] / train_denominator if train_denominator else 0.0
            ),
            "build_duration_seconds": round(time.perf_counter() - started, 6),
        }
        connection.executemany(
            "INSERT INTO metadata (key, value_json) VALUES (?, ?)",
            [(key, json.dumps(value, separators=(",", ":"), sort_keys=True)) for key, value in metadata.items()],
        )
        connection.commit()
        if progress is not None:
            progress(
                f"split: stored {counters['users_before_filter']:,} users; "
                f"{counters['users_after_filter']:,} are evaluation-eligible"
            )
    except Exception:
        connection.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    else:
        connection.close()

    temporary_path.replace(output_path)
    return metadata


def split_store_matches(
    path: Path,
    ratings_path: Path,
    config: SplitConfig,
    *,
    catalog_ids: set[int],
    source_user_limit: int | None,
) -> bool:
    """Cheap reuse check; the full source checksum remains in the artifact."""
    try:
        metadata = SplitStore(path).metadata()
    except (FileNotFoundError, ValueError, sqlite3.DatabaseError):
        return False
    return bool(
        metadata.get("source_size_bytes") == ratings_path.stat().st_size
        and metadata.get("source_mtime_ns") == ratings_path.stat().st_mtime_ns
        and metadata.get("source_user_limit") == source_user_limit
        and metadata.get("catalog_ids_sha256") == catalog_ids_sha256(catalog_ids)
        and metadata.get("split_config", {}).get("seed") == config.seed
        and metadata.get("split_config", {}).get("minimum_positives") == config.minimum_positives
        and metadata.get("split_config", {}).get("feedback") == asdict(config.feedback)
    )


def select_evaluation_users(
    users: Iterable[UserSplit],
    *,
    limit: int | None,
    seed: int,
) -> list[UserSplit]:
    """Choose a deterministic, approximately uniform user sample by stable hash."""
    eligible = [user for user in users if user.eligible]
    if limit is None or limit <= 0 or limit >= len(eligible):
        return eligible

    def sample_key(user: UserSplit) -> tuple[bytes, int]:
        digest = hashlib.blake2b(f"sample:{seed}:{user.user_id}".encode(), digest_size=8).digest()
        return digest, user.user_id

    return sorted(sorted(eligible, key=sample_key)[:limit], key=lambda user: user.user_id)


def select_evaluation_user_ids(
    store: SplitStore,
    *,
    limit: int | None,
    seed: int,
    strategy: str = "uniform",
) -> list[int]:
    """Select identical users for every model while retaining IDs only."""
    if strategy not in {"uniform", "stratified"}:
        raise ValueError("sampling strategy must be 'uniform' or 'stratified'")
    if limit is None or limit <= 0:
        return store.eligible_user_ids()
    activity = store.eligible_user_activity()
    eligible_ids = [user_id for user_id, _count in activity]
    if limit >= len(eligible_ids):
        return eligible_ids

    def sample_key(user_id: int) -> tuple[bytes, int]:
        digest = hashlib.blake2b(f"sample:{seed}:{user_id}".encode(), digest_size=8).digest()
        return digest, user_id

    if strategy == "uniform":
        return sorted(sorted(eligible_ids, key=sample_key)[:limit])

    groups: dict[str, list[int]] = {"sparse": [], "medium": [], "heavy": []}
    for user_id, count in activity:
        segment = "sparse" if count <= 4 else "medium" if count <= 19 else "heavy"
        groups[segment].append(user_id)
    base, remainder = divmod(limit, 3)
    selected: set[int] = set()
    for index, segment in enumerate(("sparse", "medium", "heavy")):
        target = base + int(index < remainder)
        selected.update(sorted(groups[segment], key=sample_key)[:target])
    if len(selected) < limit:
        remaining = (user_id for user_id in eligible_ids if user_id not in selected)
        selected.update(sorted(remaining, key=sample_key)[: limit - len(selected)])
    return sorted(selected)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
