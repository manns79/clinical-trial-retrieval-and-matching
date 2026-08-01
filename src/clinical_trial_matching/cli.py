from __future__ import annotations

import argparse
from pathlib import Path

from clinical_trial_matching.evaluation.metrics import summarize_run
from clinical_trial_matching.ingestion.clinicaltrials import (
    trial_from_flat_record,
    trial_to_flat_record,
)
from clinical_trial_matching.io import read_jsonl, write_json, write_jsonl
from clinical_trial_matching.models import Topic
from clinical_trial_matching.retrieval.bm25 import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(prog="ctmatch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest-sample", help="Normalize synthetic fixture trials.")
    ingest.add_argument("--trials", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate-baseline", help="Evaluate BM25 on fixture data.")
    evaluate.add_argument("--trials", type=Path, required=True)
    evaluate.add_argument("--topics", type=Path, required=True)
    evaluate.add_argument("--qrels", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--top-k", type=int, default=100)

    args = parser.parse_args()

    if args.command == "ingest-sample":
        ingest_sample(args.trials, args.output)
    elif args.command == "evaluate-baseline":
        evaluate_baseline(args.trials, args.topics, args.qrels, args.output, args.top_k)


def ingest_sample(trials_path: Path, output_path: Path) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    write_jsonl(output_path, (trial_to_flat_record(trial) for trial in trials))
    print(f"Wrote {len(trials)} normalized trials to {output_path}")


def evaluate_baseline(
    trials_path: Path,
    topics_path: Path,
    qrels_path: Path,
    output_path: Path,
    top_k: int,
) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    topics = [Topic(topic_id=str(row["topic_id"]), text=str(row["text"])) for row in read_jsonl(topics_path)]
    qrels = read_qrels(qrels_path)

    retriever = BM25Retriever(trials)
    run = {
        topic.topic_id: [result.nct_id for result in retriever.search(topic.text, top_k=top_k)]
        for topic in topics
    }
    metrics = summarize_run(run, qrels)
    payload = {"run_name": "sample_bm25", "metrics": metrics, "topics": len(topics), "trials": len(trials)}
    write_json(output_path, payload)
    print(f"Wrote baseline metrics to {output_path}")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
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
            qrels.setdefault(topic_id, {})[nct_id] = int(relevance)
    return qrels


if __name__ == "__main__":
    main()
