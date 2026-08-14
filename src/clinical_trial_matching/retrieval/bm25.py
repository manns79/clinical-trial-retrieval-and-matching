from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clinical_trial_matching.models import SearchResult, Trial

TOKEN_RE = re.compile(r"[a-z0-9]+")
BM25_INDEX_SCHEMA_VERSION = "1.0"
QUERY_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "patient",
    "seeks",
    "she",
    "the",
    "their",
    "to",
    "trial",
    "with",
}
DEFAULT_FIELD_WEIGHTS = {
    "all_text": 1.0,
    "title": 0.75,
    "brief_summary": 0.5,
    "conditions": 0.75,
    "interventions": 0.25,
    "eligibility_criteria": 0.25,
    "demographics": 0.1,
    "status": 0.05,
    "locations": 0.05,
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def tokenize_query(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in QUERY_STOPWORDS]


class BM25Retriever:
    def __init__(
        self,
        trials: Iterable[Trial],
        k1: float = 1.5,
        b: float = 0.75,
        index: BM25FieldIndex | None = None,
    ) -> None:
        self.trials = list(trials)
        self.k1 = k1
        self.b = b
        self.index = index or build_field_index(
            self.trials,
            field_name="all_text",
            weight=1.0,
        )

    @classmethod
    def from_index_record(
        cls,
        trials: Iterable[Trial],
        record: dict[str, Any],
    ) -> BM25Retriever:
        if record.get("retriever") != "bm25":
            raise ValueError("BM25 index record is not for retriever 'bm25'")
        return cls(
            trials,
            k1=float(record.get("k1", 1.5)),
            b=float(record.get("b", 0.75)),
            index=field_index_from_record(record["index"]),
        )

    def to_index_record(self) -> dict[str, Any]:
        return {
            "retriever": "bm25",
            "k1": self.k1,
            "b": self.b,
            "index": field_index_to_record(self.index),
        }

    def _idf(self, term: str) -> float:
        doc_count = len(self.trials)
        df = self.index.document_frequencies.get(term, 0)
        return math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_terms = tokenize_query(query)
        scores: Counter[int] = Counter()
        for term in query_terms:
            idf = self._idf(term)
            for doc_index, frequency in self.index.postings.get(term, ()):
                doc_length = self.index.doc_lengths[doc_index]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.index.avg_doc_length, 1)
                )
                scores[doc_index] += idf * (frequency * (self.k1 + 1)) / denominator

        scored = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(
                nct_id=self.trials[doc_index].nct_id,
                score=score,
                rank=rank,
                title=self.trials[doc_index].title,
            )
            for rank, (doc_index, score) in enumerate(scored[:top_k], start=1)
        ]


@dataclass(frozen=True)
class BM25FieldIndex:
    field_name: str
    weight: float
    document_frequencies: Counter[str]
    doc_lengths: list[int]
    avg_doc_length: float
    postings: dict[str, tuple[tuple[int, int], ...]]


class FieldAwareBM25Retriever:
    def __init__(
        self,
        trials: Iterable[Trial],
        *,
        field_weights: dict[str, float] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        field_indexes: list[BM25FieldIndex] | None = None,
    ) -> None:
        self.trials = list(trials)
        self.k1 = k1
        self.b = b
        self.field_weights = normalized_field_weights(field_weights)
        self.field_indexes = field_indexes or [
            build_field_index(self.trials, field_name=field_name, weight=weight)
            for field_name, weight in self.field_weights.items()
            if weight > 0
        ]

    @classmethod
    def from_index_record(
        cls,
        trials: Iterable[Trial],
        record: dict[str, Any],
    ) -> FieldAwareBM25Retriever:
        if record.get("retriever") != "fielded-bm25":
            raise ValueError("BM25 index record is not for retriever 'fielded-bm25'")
        return cls(
            trials,
            field_weights={str(k): float(v) for k, v in record.get("field_weights", {}).items()},
            k1=float(record.get("k1", 1.5)),
            b=float(record.get("b", 0.75)),
            field_indexes=[
                field_index_from_record(field_record)
                for field_record in record.get("field_indexes", [])
            ],
        )

    def to_index_record(self) -> dict[str, Any]:
        return {
            "retriever": "fielded-bm25",
            "k1": self.k1,
            "b": self.b,
            "field_weights": self.field_weights,
            "field_indexes": [field_index_to_record(index) for index in self.field_indexes],
        }

    def _idf(self, term: str, index: BM25FieldIndex) -> float:
        doc_count = len(self.trials)
        df = index.document_frequencies.get(term, 0)
        return math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

    def _score_field(
        self,
        index: BM25FieldIndex,
        term: str,
        doc_index: int,
        frequency: int,
    ) -> float:
        doc_length = index.doc_lengths[doc_index]
        if doc_length == 0:
            return 0.0
        denominator = frequency + self.k1 * (
            1 - self.b + self.b * doc_length / max(index.avg_doc_length, 1)
        )
        return index.weight * self._idf(term, index) * (frequency * (self.k1 + 1)) / denominator

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_terms = tokenize_query(query)
        scores: Counter[int] = Counter()
        for index in self.field_indexes:
            for term in query_terms:
                for doc_index, frequency in index.postings.get(term, ()):
                    scores[doc_index] += self._score_field(
                        index=index,
                        term=term,
                        doc_index=doc_index,
                        frequency=frequency,
                    )

        scored = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            SearchResult(
                nct_id=self.trials[doc_index].nct_id,
                score=score,
                rank=rank,
                title=self.trials[doc_index].title,
            )
            for rank, (doc_index, score) in enumerate(scored[:top_k], start=1)
        ]


def search_trials(
    trials: Iterable[Trial],
    *,
    query: str,
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
    snippet_chars: int = 240,
    retriever_name: str = "fielded-bm25",
    field_weights: dict[str, float] | None = None,
    retriever: BM25Retriever | FieldAwareBM25Retriever | None = None,
) -> dict[str, Any]:
    trial_list = list(trials)
    trial_by_id = {trial.nct_id: trial for trial in trial_list}
    active_retriever = retriever or build_bm25_retriever(
        trial_list,
        retriever_name=retriever_name,
        field_weights=field_weights,
        k1=k1,
        b=b,
    )
    results = active_retriever.search(query, top_k=top_k)
    normalized_weights = (
        normalized_field_weights(field_weights) if retriever_name == "fielded-bm25" else {}
    )
    return {
        "query": query,
        "retriever": retriever_name,
        "parameters": {
            "top_k": top_k,
            "k1": k1,
            "b": b,
            "field_weights": normalized_weights,
            "query_stopwords": sorted(QUERY_STOPWORDS),
        },
        "corpus": {
            "trials": len(trial_list),
            "unique_nct_ids": len(trial_by_id),
        },
        "results": format_search_results(
            trial_list,
            results,
            query=query,
            snippet_chars=snippet_chars,
        ),
    }


def format_search_results(
    trials: Iterable[Trial],
    results: Iterable[SearchResult],
    *,
    query: str,
    snippet_chars: int,
    result_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    trial_by_id = {trial.nct_id: trial for trial in trials}
    metadata = result_metadata or {}
    return [
        _result_record(
            result,
            trial_by_id[result.nct_id],
            query,
            snippet_chars,
            metadata.get(result.nct_id),
        )
        for result in results
    ]


def _result_record(
    result: SearchResult,
    trial: Trial,
    query: str,
    snippet_chars: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "rank": result.rank,
        "score": round(result.score, 6),
        "nct_id": trial.nct_id,
        "title": trial.title,
        "brief_summary": trial.brief_summary,
        "status": trial.status,
        "conditions": list(trial.conditions),
        "interventions": list(trial.interventions),
        "sex": trial.sex,
        "minimum_age": trial.minimum_age,
        "maximum_age": trial.maximum_age,
        "locations": list(trial.locations),
        "matched_terms": matched_terms(query, trial.searchable_text),
        "snippet": make_snippet(trial.searchable_text, query, snippet_chars),
    }
    if metadata:
        record.update(metadata)
    return record


def matched_terms(query: str, text: str) -> list[str]:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))
    return sorted(query_terms & text_terms)


def make_snippet(text: str, query: str, max_chars: int = 240) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if max_chars < 1 or len(normalized) <= max_chars:
        return normalized

    query_terms = tokenize(query)
    lowered = normalized.lower()
    match_index = min(
        (index for term in query_terms if (index := lowered.find(term)) >= 0),
        default=0,
    )
    start = max(match_index - max_chars // 3, 0)
    end = min(start + max_chars, len(normalized))
    start = max(end - max_chars, 0)
    snippet = normalized[start:end].strip()
    if start > 0:
        snippet = f"... {snippet}"
    if end < len(normalized):
        snippet = f"{snippet} ..."
    return snippet


def build_bm25_retriever(
    trials: Iterable[Trial],
    *,
    retriever_name: str,
    field_weights: dict[str, float] | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Retriever | FieldAwareBM25Retriever:
    if retriever_name == "bm25":
        return BM25Retriever(trials, k1=k1, b=b)
    if retriever_name == "fielded-bm25":
        return FieldAwareBM25Retriever(trials, field_weights=field_weights, k1=k1, b=b)
    raise ValueError(f"Unsupported BM25 retriever: {retriever_name}")


def normalized_field_weights(field_weights: dict[str, float] | None = None) -> dict[str, float]:
    weights = dict(DEFAULT_FIELD_WEIGHTS)
    if field_weights:
        unknown_fields = sorted(set(field_weights) - set(DEFAULT_FIELD_WEIGHTS))
        if unknown_fields:
            raise ValueError(f"Unknown BM25 field weight(s): {', '.join(unknown_fields)}")
        for field_name, weight in field_weights.items():
            if weight < 0:
                raise ValueError("BM25 field weights cannot be negative")
            weights[field_name] = weight
    return weights


def trial_field_texts(trial: Trial) -> dict[str, str]:
    return {
        "all_text": trial.searchable_text,
        "title": trial.title,
        "brief_summary": trial.brief_summary,
        "conditions": " ".join(trial.conditions),
        "interventions": " ".join(trial.interventions),
        "eligibility_criteria": trial.eligibility_criteria,
        "demographics": " ".join(
            part for part in [trial.sex, trial.minimum_age, trial.maximum_age] if part
        ),
        "status": " ".join(
            part for part in [trial.status, *trial.phases, trial.study_type] if part
        ),
        "locations": " ".join(trial.locations),
    }


def build_field_index(
    trials: list[Trial],
    *,
    field_name: str,
    weight: float,
) -> BM25FieldIndex:
    doc_lengths: list[int] = []
    document_frequencies: Counter[str] = Counter()
    mutable_postings: dict[str, list[tuple[int, int]]] = {}
    for doc_index, trial in enumerate(trials):
        tokens = tokenize(trial_field_texts(trial).get(field_name, ""))
        doc_lengths.append(len(tokens))
        term_frequency = Counter(tokens)
        document_frequencies.update(term_frequency.keys())
        for term, frequency in term_frequency.items():
            mutable_postings.setdefault(term, []).append((doc_index, frequency))
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    return BM25FieldIndex(
        field_name=field_name,
        weight=weight,
        document_frequencies=document_frequencies,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
        postings={term: tuple(postings) for term, postings in mutable_postings.items()},
    )


def field_index_to_record(index: BM25FieldIndex) -> dict[str, Any]:
    return {
        "field_name": index.field_name,
        "weight": index.weight,
        "document_frequencies": dict(index.document_frequencies),
        "doc_lengths": index.doc_lengths,
        "avg_doc_length": index.avg_doc_length,
        "postings": {
            term: [[doc_index, frequency] for doc_index, frequency in postings]
            for term, postings in index.postings.items()
        },
    }


def field_index_from_record(record: dict[str, Any]) -> BM25FieldIndex:
    return BM25FieldIndex(
        field_name=str(record["field_name"]),
        weight=float(record["weight"]),
        document_frequencies=Counter(
            {
                str(term): int(frequency)
                for term, frequency in record["document_frequencies"].items()
            }
        ),
        doc_lengths=[int(length) for length in record["doc_lengths"]],
        avg_doc_length=float(record["avg_doc_length"]),
        postings={
            str(term): tuple((int(doc_index), int(frequency)) for doc_index, frequency in postings)
            for term, postings in record["postings"].items()
        },
    )


def build_bm25_index_record(
    trials: Iterable[Trial],
    *,
    retriever_name: str,
    field_weights: dict[str, float] | None = None,
    k1: float = 1.5,
    b: float = 0.75,
    corpus_path: Path | None = None,
) -> dict[str, Any]:
    trial_list = list(trials)
    retriever = build_bm25_retriever(
        trial_list,
        retriever_name=retriever_name,
        field_weights=field_weights,
        k1=k1,
        b=b,
    )
    return {
        "schema_version": BM25_INDEX_SCHEMA_VERSION,
        "corpus": corpus_metadata(trial_list, corpus_path=corpus_path),
        "index": retriever.to_index_record(),
    }


def save_bm25_index(
    path: Path,
    trials: Iterable[Trial],
    *,
    retriever_name: str,
    field_weights: dict[str, float] | None = None,
    k1: float = 1.5,
    b: float = 0.75,
    corpus_path: Path | None = None,
) -> dict[str, Any]:
    record = build_bm25_index_record(
        trials,
        retriever_name=retriever_name,
        field_weights=field_weights,
        k1=k1,
        b=b,
        corpus_path=corpus_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bm25_index_record(path, record)
    return record


def load_bm25_index(
    path: Path,
    trials: Iterable[Trial],
    *,
    retriever_name: str,
    field_weights: dict[str, float] | None = None,
    corpus_path: Path | None = None,
) -> BM25Retriever | FieldAwareBM25Retriever:
    trial_list = list(trials)
    record = read_bm25_index_record(path)
    validate_bm25_index_record(
        record,
        trials=trial_list,
        retriever_name=retriever_name,
        field_weights=field_weights,
        corpus_path=corpus_path,
    )
    index_record = record["index"]
    if retriever_name == "bm25":
        return BM25Retriever.from_index_record(trial_list, index_record)
    return FieldAwareBM25Retriever.from_index_record(trial_list, index_record)


def load_or_build_bm25_retriever(
    *,
    trials: Iterable[Trial],
    retriever_name: str,
    field_weights: dict[str, float] | None = None,
    k1: float = 1.5,
    b: float = 0.75,
    corpus_path: Path | None = None,
    index_path: Path | None = None,
    rebuild_index: bool = False,
) -> BM25Retriever | FieldAwareBM25Retriever:
    trial_list = list(trials)
    if index_path and index_path.exists() and not rebuild_index:
        return load_bm25_index(
            index_path,
            trial_list,
            retriever_name=retriever_name,
            field_weights=field_weights,
            corpus_path=corpus_path,
        )
    if index_path:
        save_bm25_index(
            index_path,
            trial_list,
            retriever_name=retriever_name,
            field_weights=field_weights,
            k1=k1,
            b=b,
            corpus_path=corpus_path,
        )
        return load_bm25_index(
            index_path,
            trial_list,
            retriever_name=retriever_name,
            field_weights=field_weights,
            corpus_path=corpus_path,
        )
    return build_bm25_retriever(
        trial_list,
        retriever_name=retriever_name,
        field_weights=field_weights,
        k1=k1,
        b=b,
    )


def validate_bm25_index_record(
    record: dict[str, Any],
    *,
    trials: list[Trial],
    retriever_name: str,
    field_weights: dict[str, float] | None = None,
    corpus_path: Path | None = None,
) -> None:
    if record.get("schema_version") != BM25_INDEX_SCHEMA_VERSION:
        raise ValueError("BM25 index schema version is not supported")
    index_record = record.get("index", {})
    if index_record.get("retriever") != retriever_name:
        raise ValueError(
            f"BM25 index retriever mismatch: {index_record.get('retriever')} != {retriever_name}"
        )
    if retriever_name == "fielded-bm25":
        expected_weights = normalized_field_weights(field_weights)
        observed_weights = {
            str(k): float(v) for k, v in index_record.get("field_weights", {}).items()
        }
        if observed_weights != expected_weights:
            raise ValueError("BM25 index field weights do not match requested field weights")
    expected_corpus = corpus_metadata(trials, corpus_path=corpus_path)
    observed_corpus = record.get("corpus", {})
    for key in ["trials", "unique_nct_ids", "fingerprint"]:
        if observed_corpus.get(key) != expected_corpus[key]:
            raise ValueError(f"BM25 index corpus {key} does not match current corpus")


def corpus_metadata(trials: list[Trial], *, corpus_path: Path | None = None) -> dict[str, Any]:
    return {
        "trials": len(trials),
        "unique_nct_ids": len({trial.nct_id for trial in trials}),
        "fingerprint": corpus_fingerprint(trials),
        "path": str(corpus_path) if corpus_path else "",
        "path_sha256": file_sha256(corpus_path) if corpus_path else "",
    }


def corpus_fingerprint(trials: list[Trial]) -> str:
    digest = hashlib.sha256()
    for trial in trials:
        payload = {
            "nct_id": trial.nct_id,
            "title": trial.title,
            "brief_summary": trial.brief_summary,
            "status": trial.status,
            "conditions": list(trial.conditions),
            "interventions": list(trial.interventions),
            "eligibility_criteria": trial.eligibility_criteria,
            "sex": trial.sex,
            "minimum_age": trial.minimum_age,
            "maximum_age": trial.maximum_age,
            "phases": list(trial.phases),
            "study_type": trial.study_type,
            "locations": list(trial.locations),
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bm25_index_record(path: Path, record: dict[str, Any]) -> None:
    if path.suffix == ".json":
        path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        return
    with path.open("wb") as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_bm25_index_record(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("BM25 index payload was not a mapping")
    return payload
