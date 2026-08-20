from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from clinical_trial_matching.ingestion.clinicaltrials import (
    trial_from_flat_record,
    trial_to_flat_record,
)
from clinical_trial_matching.models import Trial
from clinical_trial_matching.retrieval.bm25 import file_sha256, update_corpus_fingerprint

TRIAL_STORE_SCHEMA_VERSION = "1.0"
TRIAL_STORE_NAME = "sqlite-trial-metadata"


class SQLiteTrialStore:
    def __init__(self, path: Path, metadata: dict[str, Any]) -> None:
        self.path = path
        self.metadata = metadata
        self.corpus = dict(metadata["corpus"])

    @property
    def count(self) -> int:
        return int(self.corpus["trials"])

    @property
    def unique_nct_ids(self) -> int:
        return int(self.corpus["unique_nct_ids"])

    def get(self, nct_id: str) -> Trial | None:
        normalized = nct_id.strip().upper()
        with _read_only_connection(self.path) as connection:
            row = connection.execute(
                "SELECT payload FROM trials WHERE nct_id = ?",
                (normalized,),
            ).fetchone()
        return None if row is None else _trial_from_payload(str(row["payload"]))

    def get_many(self, nct_ids: Sequence[str]) -> list[Trial]:
        ordered_ids = [nct_id.strip().upper() for nct_id in nct_ids]
        unique_ids = list(dict.fromkeys(ordered_ids))
        if not unique_ids:
            return []
        placeholders = ", ".join("?" for _ in unique_ids)
        with _read_only_connection(self.path) as connection:
            rows = connection.execute(
                f"SELECT nct_id, payload FROM trials WHERE nct_id IN ({placeholders})",
                unique_ids,
            ).fetchall()
        trials_by_id = {
            str(row["nct_id"]): _trial_from_payload(str(row["payload"])) for row in rows
        }
        missing = [nct_id for nct_id in unique_ids if nct_id not in trials_by_id]
        if missing:
            raise ValueError(
                "Trial metadata store is missing requested NCT IDs: " + ", ".join(missing)
            )
        return [trials_by_id[nct_id] for nct_id in ordered_ids]

    def validate_nct_id_order(self, expected_nct_ids: Sequence[str]) -> None:
        if len(expected_nct_ids) != self.count:
            raise ValueError("Trial metadata store and dense index have different row counts")
        with _read_only_connection(self.path) as connection:
            rows = connection.execute(
                "SELECT nct_id FROM trials ORDER BY ordinal"
            )
            for expected, row in zip(expected_nct_ids, rows, strict=True):
                if expected != str(row["nct_id"]):
                    raise ValueError("Trial metadata store NCT ID order does not match dense index")


def build_trial_store(
    path: Path,
    trials: Iterable[Trial],
    *,
    corpus_path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_store_path(path)
    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    trial_count = 0
    try:
        with closing(sqlite3.connect(temporary_path)) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE trials ("
                "nct_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE, payload TEXT NOT NULL)"
            )
            connection.execute("CREATE TABLE store_metadata (payload TEXT NOT NULL)")
            for ordinal, trial in enumerate(trials):
                nct_id = trial.nct_id.strip().upper()
                if not nct_id:
                    raise ValueError("Trial metadata store cannot contain an empty NCT ID")
                if nct_id in seen_ids:
                    raise ValueError(f"Trial metadata store contains duplicate NCT ID: {nct_id}")
                seen_ids.add(nct_id)
                update_corpus_fingerprint(digest, trial)
                connection.execute(
                    "INSERT INTO trials(nct_id, ordinal, payload) VALUES (?, ?, ?)",
                    (
                        nct_id,
                        ordinal,
                        json.dumps(trial_to_flat_record(trial), sort_keys=True),
                    ),
                )
                trial_count += 1
            if trial_count == 0:
                raise ValueError("Cannot build a trial metadata store for an empty corpus")
            metadata = {
                "schema_version": TRIAL_STORE_SCHEMA_VERSION,
                "store": TRIAL_STORE_NAME,
                "record_encoding": "normalized_trial_json",
                "corpus": {
                    "trials": trial_count,
                    "unique_nct_ids": len(seen_ids),
                    "fingerprint": digest.hexdigest(),
                    "path": str(corpus_path),
                    "path_sha256": file_sha256(corpus_path),
                },
            }
            connection.execute(
                "INSERT INTO store_metadata(payload) VALUES (?)",
                (json.dumps(metadata, sort_keys=True),),
            )
            connection.commit()
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return metadata


def load_trial_store(path: Path, *, corpus_path: Path) -> SQLiteTrialStore:
    if not path.is_file():
        raise FileNotFoundError(f"Trial metadata store not found: {path}")
    with _read_only_connection(path) as connection:
        try:
            row = connection.execute("SELECT payload FROM store_metadata").fetchone()
            stored_count = int(connection.execute("SELECT count(*) FROM trials").fetchone()[0])
        except sqlite3.OperationalError as exc:
            raise RuntimeError("Trial metadata store could not be read") from exc
    if row is None:
        raise ValueError("Trial metadata store metadata is missing")
    metadata = json.loads(str(row["payload"]))
    if not isinstance(metadata, dict):
        raise ValueError("Trial metadata store metadata must be a JSON object")
    if metadata.get("schema_version") != TRIAL_STORE_SCHEMA_VERSION:
        raise ValueError("Trial metadata store schema version is not supported")
    if metadata.get("store") != TRIAL_STORE_NAME:
        raise ValueError("Trial metadata store type is invalid")
    corpus = metadata.get("corpus")
    if not isinstance(corpus, dict):
        raise ValueError("Trial metadata store corpus metadata is invalid")
    if int(corpus.get("trials", -1)) != stored_count:
        raise ValueError("Trial metadata store row count is inconsistent")
    if int(corpus.get("unique_nct_ids", -1)) != stored_count:
        raise ValueError("Trial metadata store contains duplicate NCT IDs")
    if corpus.get("path_sha256") != file_sha256(corpus_path):
        raise ValueError("Trial metadata store does not match the configured corpus file")
    return SQLiteTrialStore(path, metadata)


def _trial_from_payload(value: str) -> Trial:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Stored trial payload must be a JSON object")
    return trial_from_flat_record(payload)


@contextmanager
def _read_only_connection(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def _temporary_store_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(name)
    temporary_path.unlink()
    return temporary_path
