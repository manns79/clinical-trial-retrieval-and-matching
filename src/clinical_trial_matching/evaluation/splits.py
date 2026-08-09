from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

from clinical_trial_matching.models import Qrel, Topic

TOPIC_SPLIT_SCHEMA_VERSION = "1.0"
TOPIC_SPLIT_STRATEGY = "seeded_sha256_rank"


@dataclass(frozen=True)
class TrecTopicSplit:
    development_topics: tuple[Topic, ...]
    development_qrels: tuple[Qrel, ...]
    holdout_topics: tuple[Topic, ...]
    holdout_qrels: tuple[Qrel, ...]
    seed: str
    holdout_fraction: float

    @property
    def development_topic_ids(self) -> tuple[str, ...]:
        return tuple(topic.topic_id for topic in self.development_topics)

    @property
    def holdout_topic_ids(self) -> tuple[str, ...]:
        return tuple(topic.topic_id for topic in self.holdout_topics)


def build_trec_topic_split(
    topics: list[Topic],
    qrels: list[Qrel],
    *,
    seed: str,
    holdout_fraction: float = 0.2,
) -> TrecTopicSplit:
    if not seed.strip():
        raise ValueError("Topic split seed cannot be empty")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("Holdout fraction must be greater than 0 and less than 1")
    if len(topics) < 2:
        raise ValueError("At least two topics are required for development/holdout splitting")

    topic_ids = [topic.topic_id for topic in topics]
    duplicate_topic_ids = sorted(
        topic_id for topic_id, count in Counter(topic_ids).items() if count > 1
    )
    if duplicate_topic_ids:
        raise ValueError(f"Duplicate topic IDs cannot be split: {', '.join(duplicate_topic_ids)}")

    topic_id_set = set(topic_ids)
    unknown_qrel_topic_ids = sorted({qrel.topic_id for qrel in qrels} - topic_id_set)
    if unknown_qrel_topic_ids:
        raise ValueError(
            "Qrels contain topic IDs absent from topics: " + ", ".join(unknown_qrel_topic_ids)
        )

    ranked_topic_ids = sorted(
        topic_ids,
        key=lambda topic_id: (_split_key(seed, topic_id), topic_id),
    )
    holdout_count = round(len(ranked_topic_ids) * holdout_fraction)
    holdout_count = max(1, min(len(ranked_topic_ids) - 1, holdout_count))
    holdout_ids = set(ranked_topic_ids[:holdout_count])
    development_ids = topic_id_set - holdout_ids

    return TrecTopicSplit(
        development_topics=tuple(
            topic for topic in topics if topic.topic_id in development_ids
        ),
        development_qrels=tuple(
            qrel for qrel in qrels if qrel.topic_id in development_ids
        ),
        holdout_topics=tuple(topic for topic in topics if topic.topic_id in holdout_ids),
        holdout_qrels=tuple(qrel for qrel in qrels if qrel.topic_id in holdout_ids),
        seed=seed,
        holdout_fraction=holdout_fraction,
    )


def topic_split_report(
    split: TrecTopicSplit,
    *,
    topics_source: dict[str, str | int],
    qrels_source: dict[str, str | int],
) -> dict[str, Any]:
    development_ids = set(split.development_topic_ids)
    holdout_ids = set(split.holdout_topic_ids)
    return {
        "schema_version": TOPIC_SPLIT_SCHEMA_VERSION,
        "strategy": TOPIC_SPLIT_STRATEGY,
        "seed": split.seed,
        "holdout_fraction": split.holdout_fraction,
        "sources": {
            "topics": topics_source,
            "qrels": qrels_source,
        },
        "partitions": {
            "development": _partition_summary(
                split.development_topics,
                split.development_qrels,
            ),
            "holdout": _partition_summary(split.holdout_topics, split.holdout_qrels),
        },
        "integrity": {
            "topic_overlap": sorted(development_ids & holdout_ids),
            "assigned_topics": len(development_ids | holdout_ids),
            "assigned_qrels": len(split.development_qrels) + len(split.holdout_qrels),
        },
    }


def _split_key(seed: str, topic_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{topic_id}".encode()).hexdigest()


def _partition_summary(topics: tuple[Topic, ...], qrels: tuple[Qrel, ...]) -> dict[str, Any]:
    relevance_counts = Counter(qrel.relevance for qrel in qrels)
    return {
        "topics": len(topics),
        "qrels": len(qrels),
        "topic_ids": sorted(topic.topic_id for topic in topics),
        "relevance_distribution": {
            "irrelevant": relevance_counts.get(0, 0),
            "excluded": relevance_counts.get(1, 0),
            "eligible": relevance_counts.get(2, 0),
            "other": sum(
                count
                for relevance, count in relevance_counts.items()
                if relevance not in {0, 1, 2}
            ),
        },
    }
