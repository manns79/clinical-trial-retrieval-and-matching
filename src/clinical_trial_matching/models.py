from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Trial:
    nct_id: str
    title: str
    brief_summary: str = ""
    status: str = ""
    conditions: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    eligibility_criteria: str = ""
    sex: str = ""
    minimum_age: str = ""
    maximum_age: str = ""
    phases: tuple[str, ...] = ()
    study_type: str = ""
    locations: tuple[str, ...] = ()
    source: dict[str, str] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        parts = [
            self.title,
            self.brief_summary,
            self.status,
            " ".join(self.conditions),
            " ".join(self.interventions),
            self.eligibility_criteria,
            self.sex,
            self.minimum_age,
            self.maximum_age,
            " ".join(self.phases),
            self.study_type,
            " ".join(self.locations),
        ]
        return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class TopicField:
    name: str
    value: str


@dataclass(frozen=True)
class Topic:
    topic_id: str
    text: str
    year: int | None = None
    fields: tuple[TopicField, ...] = ()
    template: str = ""
    source: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Qrel:
    topic_id: str
    nct_id: str
    relevance: int
    year: int | None = None

    @property
    def label(self) -> str:
        labels = {0: "irrelevant", 1: "excluded", 2: "eligible"}
        return labels.get(self.relevance, "unknown")


@dataclass(frozen=True)
class SearchResult:
    nct_id: str
    score: float
    rank: int
    title: str
