from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from clinical_trial_matching.models import SearchResult, Trial

TOKEN_RE = re.compile(r"[a-z0-9]+")
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


class BM25Retriever:
    def __init__(self, trials: Iterable[Trial], k1: float = 1.5, b: float = 0.75) -> None:
        self.trials = list(trials)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(trial.searchable_text) for trial in self.trials]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.document_frequencies = self._document_frequencies()

    def _document_frequencies(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            frequencies.update(set(tokens))
        return frequencies

    def _idf(self, term: str) -> float:
        doc_count = len(self.trials)
        df = self.document_frequencies.get(term, 0)
        return math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_terms = tokenize(query)
        scored: list[tuple[float, Trial]] = []
        for trial, term_frequency, doc_length in zip(
            self.trials, self.term_frequencies, self.doc_lengths, strict=True
        ):
            score = 0.0
            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
                )
                score += self._idf(term) * (frequency * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append((score, trial))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(nct_id=trial.nct_id, score=score, rank=rank, title=trial.title)
            for rank, (score, trial) in enumerate(scored[:top_k], start=1)
        ]


@dataclass(frozen=True)
class BM25FieldIndex:
    field_name: str
    weight: float
    doc_tokens: list[list[str]]
    term_frequencies: list[Counter[str]]
    document_frequencies: Counter[str]
    doc_lengths: list[int]
    avg_doc_length: float


class FieldAwareBM25Retriever:
    def __init__(
        self,
        trials: Iterable[Trial],
        *,
        field_weights: dict[str, float] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.trials = list(trials)
        self.k1 = k1
        self.b = b
        self.field_weights = normalized_field_weights(field_weights)
        self.field_indexes = [
            self._build_field_index(field_name, weight)
            for field_name, weight in self.field_weights.items()
            if weight > 0
        ]

    def _build_field_index(self, field_name: str, weight: float) -> BM25FieldIndex:
        doc_tokens = [
            tokenize(trial_field_texts(trial).get(field_name, "")) for trial in self.trials
        ]
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        term_frequencies = [Counter(tokens) for tokens in doc_tokens]
        document_frequencies: Counter[str] = Counter()
        for tokens in doc_tokens:
            document_frequencies.update(set(tokens))
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        return BM25FieldIndex(
            field_name=field_name,
            weight=weight,
            doc_tokens=doc_tokens,
            term_frequencies=term_frequencies,
            document_frequencies=document_frequencies,
            doc_lengths=doc_lengths,
            avg_doc_length=avg_doc_length,
        )

    def _idf(self, term: str, index: BM25FieldIndex) -> float:
        doc_count = len(self.trials)
        df = index.document_frequencies.get(term, 0)
        return math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

    def _score_field(
        self,
        query_terms: list[str],
        index: BM25FieldIndex,
        trial_index: int,
    ) -> float:
        doc_length = index.doc_lengths[trial_index]
        if doc_length == 0:
            return 0.0
        term_frequency = index.term_frequencies[trial_index]
        score = 0.0
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * doc_length / max(index.avg_doc_length, 1)
            )
            score += index.weight * self._idf(term, index) * (frequency * (self.k1 + 1)) / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_terms = tokenize(query)
        scored: list[tuple[float, Trial]] = []
        for trial_index, trial in enumerate(self.trials):
            score = sum(
                self._score_field(query_terms, index, trial_index) for index in self.field_indexes
            )
            if score > 0:
                scored.append((score, trial))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(nct_id=trial.nct_id, score=score, rank=rank, title=trial.title)
            for rank, (score, trial) in enumerate(scored[:top_k], start=1)
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
) -> dict[str, Any]:
    trial_list = list(trials)
    trial_by_id = {trial.nct_id: trial for trial in trial_list}
    retriever = build_bm25_retriever(
        trial_list,
        retriever_name=retriever_name,
        field_weights=field_weights,
        k1=k1,
        b=b,
    )
    results = retriever.search(query, top_k=top_k)
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
        },
        "corpus": {
            "trials": len(trial_list),
            "unique_nct_ids": len(trial_by_id),
        },
        "results": [
            _result_record(result, trial_by_id[result.nct_id], query, snippet_chars)
            for result in results
        ],
    }


def _result_record(
    result: SearchResult,
    trial: Trial,
    query: str,
    snippet_chars: int,
) -> dict[str, Any]:
    return {
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
        "status": " ".join(part for part in [trial.status, *trial.phases, trial.study_type] if part),
        "locations": " ".join(trial.locations),
    }
