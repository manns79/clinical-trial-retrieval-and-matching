from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from clinical_trial_matching.models import SearchResult, Trial

TOKEN_RE = re.compile(r"[a-z0-9]+")


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


def search_trials(
    trials: Iterable[Trial],
    *,
    query: str,
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
    snippet_chars: int = 240,
) -> dict[str, Any]:
    trial_list = list(trials)
    trial_by_id = {trial.nct_id: trial for trial in trial_list}
    retriever = BM25Retriever(trial_list, k1=k1, b=b)
    results = retriever.search(query, top_k=top_k)
    return {
        "query": query,
        "retriever": "bm25",
        "parameters": {
            "top_k": top_k,
            "k1": k1,
            "b": b,
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
