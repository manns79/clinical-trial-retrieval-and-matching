from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Trial:
    nct_id: str
    title: str
    status: str = ""
    conditions: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    eligibility_criteria: str = ""
    locations: tuple[str, ...] = ()
    source: dict[str, str] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        parts = [
            self.title,
            self.status,
            " ".join(self.conditions),
            " ".join(self.interventions),
            self.eligibility_criteria,
            " ".join(self.locations),
        ]
        return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class Topic:
    topic_id: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    nct_id: str
    score: float
    rank: int
    title: str
