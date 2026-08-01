from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from clinical_trial_matching.models import Qrel, Topic, TopicField


def parse_topics_xml(path: Path, year: int | None = None) -> list[Topic]:
    root = ET.parse(path).getroot()
    topics: list[Topic] = []
    for index, topic_element in enumerate(root.findall(".//topic"), start=1):
        topic_id = topic_element.attrib.get("number") or topic_element.attrib.get("id") or str(index)
        template = topic_element.attrib.get("template", "")
        fields = _parse_topic_fields(topic_element)
        if fields:
            text = _fields_to_text(fields)
        else:
            text = _normalize_whitespace("".join(topic_element.itertext()))
        if not text:
            raise ValueError(f"Topic {topic_id!r} in {path} has no text")
        topics.append(
            Topic(
                topic_id=topic_id,
                text=text,
                year=year,
                fields=tuple(fields),
                template=template,
                source={"path": str(path), "format": "trec_topics_xml"},
            )
        )
    if not topics:
        raise ValueError(f"No <topic> elements found in {path}")
    return topics


def parse_qrels(path: Path, year: int | None = None) -> list[Qrel]:
    qrels: list[Qrel] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) == 3:
                topic_id, nct_id, relevance = parts
            elif len(parts) >= 4:
                topic_id, _, nct_id, relevance = parts[:4]
            else:
                raise ValueError(f"Invalid qrels row at {path}:{line_number}")
            qrels.append(Qrel(topic_id=topic_id, nct_id=nct_id, relevance=int(relevance), year=year))
    if not qrels:
        raise ValueError(f"No qrels found in {path}")
    return qrels


def topic_to_json_record(topic: Topic) -> dict[str, Any]:
    return {
        "topic_id": topic.topic_id,
        "year": topic.year,
        "text": topic.text,
        "format": "fields" if topic.fields else "free_text",
        "fields": [{"name": field.name, "value": field.value} for field in topic.fields],
        "template": topic.template,
        "source": topic.source,
    }


def topic_from_json_record(record: dict[str, Any]) -> Topic:
    fields = tuple(
        TopicField(name=str(field["name"]), value=str(field.get("value", "")))
        for field in record.get("fields", [])
    )
    year = record.get("year")
    return Topic(
        topic_id=str(record["topic_id"]),
        text=str(record["text"]),
        year=int(year) if year is not None else None,
        fields=fields,
        template=str(record.get("template", "")),
        source={str(k): str(v) for k, v in dict(record.get("source", {})).items()},
    )


def qrel_to_json_record(qrel: Qrel) -> dict[str, Any]:
    return {
        "topic_id": qrel.topic_id,
        "nct_id": qrel.nct_id,
        "relevance": qrel.relevance,
        "label": qrel.label,
        "year": qrel.year,
    }


def qrel_from_json_record(record: dict[str, Any]) -> Qrel:
    year = record.get("year")
    return Qrel(
        topic_id=str(record["topic_id"]),
        nct_id=str(record["nct_id"]),
        relevance=int(record["relevance"]),
        year=int(year) if year is not None else None,
    )


def qrels_to_mapping(qrels: list[Qrel]) -> dict[str, dict[str, int]]:
    mapping: dict[str, dict[str, int]] = {}
    for qrel in qrels:
        mapping.setdefault(qrel.topic_id, {})[qrel.nct_id] = qrel.relevance
    return mapping


def validate_topics_and_qrels(topics: list[Topic], qrels: list[Qrel]) -> dict[str, Any]:
    topic_ids = {topic.topic_id for topic in topics}
    qrel_topic_ids = {qrel.topic_id for qrel in qrels}
    relevance_counts = Counter(qrel.relevance for qrel in qrels)
    return {
        "topics": len(topics),
        "qrels": len(qrels),
        "judged_topics": len(qrel_topic_ids),
        "topics_without_qrels": sorted(topic_ids - qrel_topic_ids),
        "qrels_with_unknown_topics": sorted(qrel_topic_ids - topic_ids),
        "relevance_distribution": {
            "irrelevant": relevance_counts.get(0, 0),
            "excluded": relevance_counts.get(1, 0),
            "eligible": relevance_counts.get(2, 0),
            "other": sum(count for relevance, count in relevance_counts.items() if relevance not in {0, 1, 2}),
        },
    }


def _parse_topic_fields(topic_element: ET.Element) -> list[TopicField]:
    fields: list[TopicField] = []
    for field_element in topic_element.findall("field"):
        name = field_element.attrib.get("name", "").strip()
        value = _normalize_whitespace("".join(field_element.itertext()))
        if name or value:
            fields.append(TopicField(name=name, value=value))
    return fields


def _fields_to_text(fields: list[TopicField]) -> str:
    populated = [f"{field.name}: {field.value}" for field in fields if field.value]
    return "\n".join(populated)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
