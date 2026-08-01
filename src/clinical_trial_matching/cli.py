from __future__ import annotations

import argparse
from pathlib import Path

from clinical_trial_matching.evaluation.metrics import summarize_run
from clinical_trial_matching.ingestion.clinicaltrials import (
    trial_from_flat_record,
    trial_to_flat_record,
)
from clinical_trial_matching.ingestion.trec import (
    parse_qrels,
    parse_topics_xml,
    qrel_from_json_record,
    qrel_to_json_record,
    qrels_to_mapping,
    topic_from_json_record,
    topic_to_json_record,
    validate_topics_and_qrels,
)
from clinical_trial_matching.io import read_jsonl, write_json, write_jsonl
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

    trec_topics = subparsers.add_parser(
        "ingest-trec-topics", help="Normalize TREC Clinical Trials topics XML to JSONL."
    )
    trec_topics.add_argument("--year", type=int, required=True)
    trec_topics.add_argument("--input", type=Path, required=True)
    trec_topics.add_argument("--output", type=Path, required=True)

    trec_qrels = subparsers.add_parser(
        "ingest-trec-qrels", help="Normalize TREC Clinical Trials qrels to JSONL."
    )
    trec_qrels.add_argument("--year", type=int, required=True)
    trec_qrels.add_argument("--input", type=Path, required=True)
    trec_qrels.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate-trec", help="Validate normalized TREC topics and qrels JSONL files."
    )
    validate.add_argument("--topics", type=Path, required=True)
    validate.add_argument("--qrels", type=Path, required=True)
    validate.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.command == "ingest-sample":
        ingest_sample(args.trials, args.output)
    elif args.command == "evaluate-baseline":
        evaluate_baseline(args.trials, args.topics, args.qrels, args.output, args.top_k)
    elif args.command == "ingest-trec-topics":
        ingest_trec_topics(args.year, args.input, args.output)
    elif args.command == "ingest-trec-qrels":
        ingest_trec_qrels(args.year, args.input, args.output)
    elif args.command == "validate-trec":
        validate_trec(args.topics, args.qrels, args.output)


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
    topics = [topic_from_json_record(row) for row in read_jsonl(topics_path)]
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
    if path.suffix == ".jsonl":
        return qrels_to_mapping([qrel_from_json_record(row) for row in read_jsonl(path)])
    return qrels_to_mapping(parse_qrels(path))


def ingest_trec_topics(year: int, input_path: Path, output_path: Path) -> None:
    topics = parse_topics_xml(input_path, year=year)
    write_jsonl(output_path, (topic_to_json_record(topic) for topic in topics))
    print(f"Wrote {len(topics)} normalized TREC topics to {output_path}")


def ingest_trec_qrels(year: int, input_path: Path, output_path: Path) -> None:
    qrels = parse_qrels(input_path, year=year)
    write_jsonl(output_path, (qrel_to_json_record(qrel) for qrel in qrels))
    print(f"Wrote {len(qrels)} normalized TREC qrels to {output_path}")


def validate_trec(topics_path: Path, qrels_path: Path, output_path: Path | None) -> None:
    topics = [topic_from_json_record(row) for row in read_jsonl(topics_path)]
    qrels = [qrel_from_json_record(row) for row in read_jsonl(qrels_path)]
    summary = validate_topics_and_qrels(topics, qrels)
    if output_path:
        write_json(output_path, summary)
        print(f"Wrote TREC validation summary to {output_path}")
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
