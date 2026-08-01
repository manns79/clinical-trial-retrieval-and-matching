from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

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
