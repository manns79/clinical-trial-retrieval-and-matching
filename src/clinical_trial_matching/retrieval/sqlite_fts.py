from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from clinical_trial_matching.models import SearchResult, Trial
from clinical_trial_matching.retrieval.bm25 import corpus_metadata, tokenize_query

SQLITE_FTS_INDEX_SCHEMA_VERSION = "1.0"
SQLITE_FTS_RETRIEVER_NAME = "sqlite-fts5"
SQLITE_FTS_FIELDS = (
    "title",
    "brief_summary",
    "conditions",
    "interventions",
    "eligibility_criteria",
)
DEFAULT_SQLITE_FTS_FIELD_WEIGHTS = {
    "title": 1.25,
    "brief_summary": 0.75,
    "conditions": 1.5,
    "interventions": 0.25,
    "eligibility_criteria": 0.5,
}


class SQLiteFtsRetriever:
    def __init__(self, index_path: Path, metadata: dict[str, Any]) -> None:
        self.index_path = index_path
        self.metadata = metadata
        self.field_weights = normalize_sqlite_fts_field_weights(
            metadata.get("field_weights")
        )

    def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("Top-K must be at least 1")
        terms = tokenize_query(query)
        if not terms:
            return []
        match_query = " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))
        weights = [self.field_weights[field] for field in SQLITE_FTS_FIELDS]
        rank_expression = "bm25(trials_fts, 0.0, " + ", ".join("?" for _ in weights) + ")"
        sql = (
            "SELECT nct_id, title, "
            f"{rank_expression} AS fts_rank "
            "FROM trials_fts WHERE trials_fts MATCH ? "
            "ORDER BY fts_rank ASC, nct_id ASC LIMIT ?"
        )
        with _read_only_connection(self.index_path) as connection:
            rows = connection.execute(sql, (*weights, match_query, top_k)).fetchall()
        return [
            SearchResult(
                nct_id=str(row["nct_id"]),
                score=-float(row["fts_rank"]),
                rank=rank,
                title=str(row["title"]),
            )
            for rank, row in enumerate(rows, start=1)
        ]


def build_sqlite_fts_index(
    path: Path,
    trials: Iterable[Trial],
    *,
    field_weights: Mapping[str, float] | None = None,
    corpus_path: Path | None = None,
) -> dict[str, Any]:
    trial_list = list(trials)
    if not trial_list:
        raise ValueError("Cannot build an SQLite FTS5 index for an empty corpus")
    if len({trial.nct_id for trial in trial_list}) != len(trial_list):
        raise ValueError("Cannot build an SQLite FTS5 index with duplicate NCT IDs")
    weights = normalize_sqlite_fts_field_weights(field_weights)
    metadata = {
        "schema_version": SQLITE_FTS_INDEX_SCHEMA_VERSION,
        "retriever": SQLITE_FTS_RETRIEVER_NAME,
        "field_weights": weights,
        "tokenizer": "unicode61 remove_diacritics 2",
        "query_operator": "OR",
        "corpus": corpus_metadata(trial_list, corpus_path=corpus_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_index_path(path)
    try:
        with closing(sqlite3.connect(temporary_path)) as connection:
            _require_fts5(connection)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE index_metadata (payload TEXT NOT NULL)")
            connection.execute(
                "CREATE VIRTUAL TABLE trials_fts USING fts5("
                "nct_id UNINDEXED, title, brief_summary, conditions, interventions, "
                "eligibility_criteria, tokenize='unicode61 remove_diacritics 2')"
            )
            connection.execute(
                "INSERT INTO index_metadata(payload) VALUES (?)",
                (json.dumps(metadata, sort_keys=True),),
            )
            connection.executemany(
                "INSERT INTO trials_fts("
                "nct_id, title, brief_summary, conditions, interventions, eligibility_criteria"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (_trial_row(trial) for trial in trial_list),
            )
            connection.execute("INSERT INTO trials_fts(trials_fts) VALUES ('optimize')")
            connection.commit()
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return metadata


def load_sqlite_fts_retriever(
    path: Path,
    trials: Iterable[Trial],
    *,
    field_weights: Mapping[str, float] | None = None,
    corpus_path: Path | None = None,
) -> SQLiteFtsRetriever:
    if not path.is_file():
        raise FileNotFoundError(f"SQLite FTS5 index not found: {path}")
    trial_list = list(trials)
    metadata, indexed_trials = _read_index_metadata(path)
    if not isinstance(metadata, dict):
        raise ValueError("SQLite FTS5 index metadata must be a JSON object")
    _validate_sqlite_fts_index(
        metadata,
        indexed_trials=indexed_trials,
        trials=trial_list,
        field_weights=field_weights,
        corpus_path=corpus_path,
    )
    return SQLiteFtsRetriever(path, metadata)


def load_sqlite_fts_retriever_for_corpus(
    path: Path,
    *,
    corpus: Mapping[str, Any],
    field_weights: Mapping[str, float] | None = None,
) -> SQLiteFtsRetriever:
    if not path.is_file():
        raise FileNotFoundError(f"SQLite FTS5 index not found: {path}")
    metadata, indexed_trials = _read_index_metadata(path)
    if not isinstance(metadata, dict):
        raise ValueError("SQLite FTS5 index metadata must be a JSON object")
    if metadata.get("schema_version") != SQLITE_FTS_INDEX_SCHEMA_VERSION:
        raise ValueError("SQLite FTS5 index schema version is not supported")
    if metadata.get("retriever") != SQLITE_FTS_RETRIEVER_NAME:
        raise ValueError("SQLite FTS5 index retriever metadata is invalid")
    if normalize_sqlite_fts_field_weights(metadata.get("field_weights")) != (
        normalize_sqlite_fts_field_weights(field_weights)
    ):
        raise ValueError("SQLite FTS5 index field weights do not match requested weights")
    observed_corpus = metadata.get("corpus", {})
    for key in ("trials", "unique_nct_ids", "fingerprint"):
        if observed_corpus.get(key) != corpus.get(key):
            raise ValueError(f"SQLite FTS5 index corpus {key} does not match trial store")
    if indexed_trials != int(corpus.get("trials", -1)):
        raise ValueError("SQLite FTS5 row count does not match trial store")
    return SQLiteFtsRetriever(path, metadata)


def _read_index_metadata(path: Path) -> tuple[Any, int]:
    with _read_only_connection(path) as connection:
        try:
            row = connection.execute("SELECT payload FROM index_metadata").fetchone()
            if row is None:
                raise ValueError("SQLite FTS5 index metadata is missing")
            metadata = json.loads(str(row["payload"]))
            indexed_trials = int(
                connection.execute("SELECT count(*) FROM trials_fts").fetchone()[0]
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError("SQLite FTS5 index could not be read") from exc
    return metadata, indexed_trials


def load_or_build_sqlite_fts_retriever(
    *,
    trials: Iterable[Trial],
    index_path: Path,
    field_weights: Mapping[str, float] | None = None,
    corpus_path: Path | None = None,
    rebuild_index: bool = False,
) -> SQLiteFtsRetriever:
    trial_list = list(trials)
    if rebuild_index or not index_path.exists():
        build_sqlite_fts_index(
            index_path,
            trial_list,
            field_weights=field_weights,
            corpus_path=corpus_path,
        )
    return load_sqlite_fts_retriever(
        index_path,
        trial_list,
        field_weights=field_weights,
        corpus_path=corpus_path,
    )


def normalize_sqlite_fts_field_weights(
    field_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    values = field_weights or DEFAULT_SQLITE_FTS_FIELD_WEIGHTS
    unknown = sorted(set(values) - set(SQLITE_FTS_FIELDS))
    missing = sorted(set(SQLITE_FTS_FIELDS) - set(values))
    if unknown:
        raise ValueError(f"Unknown SQLite FTS5 field weight(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"Missing SQLite FTS5 field weight(s): {', '.join(missing)}")
    normalized = {field: float(values[field]) for field in SQLITE_FTS_FIELDS}
    if any(weight < 0 for weight in normalized.values()):
        raise ValueError("SQLite FTS5 field weights cannot be negative")
    if not any(weight > 0 for weight in normalized.values()):
        raise ValueError("At least one SQLite FTS5 field weight must be positive")
    return normalized


def _validate_sqlite_fts_index(
    metadata: dict[str, Any],
    *,
    indexed_trials: int,
    trials: list[Trial],
    field_weights: Mapping[str, float] | None,
    corpus_path: Path | None,
) -> None:
    if metadata.get("schema_version") != SQLITE_FTS_INDEX_SCHEMA_VERSION:
        raise ValueError("SQLite FTS5 index schema version is not supported")
    if metadata.get("retriever") != SQLITE_FTS_RETRIEVER_NAME:
        raise ValueError("SQLite FTS5 index retriever metadata is invalid")
    if normalize_sqlite_fts_field_weights(metadata.get("field_weights")) != (
        normalize_sqlite_fts_field_weights(field_weights)
    ):
        raise ValueError("SQLite FTS5 index field weights do not match requested weights")
    expected_corpus = corpus_metadata(trials, corpus_path=corpus_path)
    observed_corpus = metadata.get("corpus", {})
    for key in ("trials", "unique_nct_ids", "fingerprint"):
        if observed_corpus.get(key) != expected_corpus[key]:
            raise ValueError(f"SQLite FTS5 index corpus {key} does not match current corpus")
    if indexed_trials != len(trials):
        raise ValueError("SQLite FTS5 row count does not match current corpus")


def _trial_row(trial: Trial) -> tuple[str, str, str, str, str, str]:
    return (
        trial.nct_id,
        trial.title,
        trial.brief_summary,
        "; ".join(trial.conditions),
        "; ".join(trial.interventions),
        trial.eligibility_criteria,
    )


@contextmanager
def _read_only_connection(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def _require_fts5(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("CREATE VIRTUAL TABLE temp.fts5_check USING fts5(value)")
        connection.execute("DROP TABLE temp.fts5_check")
    except sqlite3.OperationalError as exc:
        raise RuntimeError("This Python SQLite build does not include FTS5 support") from exc


def _temporary_index_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(name)
    temporary_path.unlink()
    return temporary_path
