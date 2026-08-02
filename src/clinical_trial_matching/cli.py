from __future__ import annotations

import argparse
from pathlib import Path

from clinical_trial_matching.evaluation.metrics import summarize_run
from clinical_trial_matching.ingestion.clinicaltrials import (
    CTGOV_API_BASE_URL,
    fetch_ctgov_studies,
    parse_studies_json,
    trial_from_flat_record,
    trial_to_flat_record,
)
from clinical_trial_matching.ingestion.manifest import (
    build_source_manifest,
    manifest_to_json_record,
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
from clinical_trial_matching.validation.trials import summarize_trial_corpus


def main() -> None:
    parser = argparse.ArgumentParser(prog="ctmatch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest-sample", help="Normalize synthetic fixture trials.")
    ingest.add_argument("--trials", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)

    ctgov = subparsers.add_parser(
        "ingest-ctgov-studies", help="Normalize ClinicalTrials.gov v2 JSON studies to JSONL."
    )
    ctgov.add_argument("--input", type=Path, required=True)
    ctgov.add_argument("--output", type=Path, required=True)

    ctgov_live = subparsers.add_parser(
        "download-ctgov-studies",
        help="Fetch a small ClinicalTrials.gov v2 query, write raw JSON, manifest, and JSONL.",
    )
    ctgov_live.add_argument("--query", required=True)
    ctgov_live.add_argument("--status", default="RECRUITING")
    ctgov_live.add_argument("--page-size", type=int, default=25)
    ctgov_live.add_argument("--raw-output", type=Path, required=True)
    ctgov_live.add_argument("--manifest-output", type=Path, required=True)
    ctgov_live.add_argument("--processed-output", type=Path, required=True)
    ctgov_live.add_argument("--base-url", default=CTGOV_API_BASE_URL)
    ctgov_live.add_argument("--timeout-seconds", type=float, default=30.0)

    evaluate = subparsers.add_parser("evaluate-baseline", help="Evaluate BM25 on fixture data.")
    evaluate.add_argument("--trials", type=Path, required=True)
    evaluate.add_argument("--topics", type=Path, required=True)
    evaluate.add_argument("--qrels", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--top-k", type=int, default=100)

    trial_report = subparsers.add_parser(
        "report-trial-corpus", help="Summarize a normalized trial JSONL corpus."
    )
    trial_report.add_argument("--trials", type=Path, required=True)
    trial_report.add_argument("--output", type=Path)
    trial_report.add_argument("--sample-size", type=int, default=5)
    trial_report.add_argument("--top-n", type=int, default=10)

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

    manifest = subparsers.add_parser(
        "write-manifest", help="Write a reproducibility manifest for a local source file."
    )
    manifest.add_argument("--name", required=True)
    manifest.add_argument("--source-url", required=True)
    manifest.add_argument("--input", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--dataset", default="")
    manifest.add_argument("--year", type=int)
    manifest.add_argument("--parser", default="")
    manifest.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Optional key=value metadata. May be provided multiple times.",
    )

    args = parser.parse_args()

    if args.command == "ingest-sample":
        ingest_sample(args.trials, args.output)
    elif args.command == "ingest-ctgov-studies":
        ingest_ctgov_studies(args.input, args.output)
    elif args.command == "download-ctgov-studies":
        download_ctgov_studies(
            query=args.query,
            status=args.status,
            page_size=args.page_size,
            raw_output=args.raw_output,
            manifest_output=args.manifest_output,
            processed_output=args.processed_output,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "evaluate-baseline":
        evaluate_baseline(args.trials, args.topics, args.qrels, args.output, args.top_k)
    elif args.command == "report-trial-corpus":
        report_trial_corpus(args.trials, args.output, args.sample_size, args.top_n)
    elif args.command == "ingest-trec-topics":
        ingest_trec_topics(args.year, args.input, args.output)
    elif args.command == "ingest-trec-qrels":
        ingest_trec_qrels(args.year, args.input, args.output)
    elif args.command == "validate-trec":
        validate_trec(args.topics, args.qrels, args.output)
    elif args.command == "write-manifest":
        write_manifest(
            name=args.name,
            source_url=args.source_url,
            input_path=args.input,
            output_path=args.output,
            dataset=args.dataset,
            year=args.year,
            parser=args.parser,
            metadata_items=args.metadata,
        )


def ingest_sample(trials_path: Path, output_path: Path) -> None:
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    write_jsonl(output_path, (trial_to_flat_record(trial) for trial in trials))
    print(f"Wrote {len(trials)} normalized trials to {output_path}")


def ingest_ctgov_studies(input_path: Path, output_path: Path) -> None:
    trials = parse_studies_json(input_path)
    write_jsonl(output_path, (trial_to_flat_record(trial) for trial in trials))
    print(f"Wrote {len(trials)} normalized ClinicalTrials.gov studies to {output_path}")


def download_ctgov_studies(
    *,
    query: str,
    status: str,
    page_size: int,
    raw_output: Path,
    manifest_output: Path,
    processed_output: Path,
    base_url: str,
    timeout_seconds: float,
) -> None:
    result = fetch_ctgov_studies(
        query=query,
        status=status,
        page_size=page_size,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    write_json(raw_output, result.payload)

    trials = parse_studies_json(raw_output)
    write_jsonl(processed_output, (trial_to_flat_record(trial) for trial in trials))

    manifest = build_source_manifest(
        name=f"clinicaltrials_gov_{slugify(query)}_{slugify(status)}_{page_size}",
        source_url=result.request_url,
        input_path=raw_output,
        dataset="clinicaltrials_gov",
        parser="clinicaltrials_gov_v2_studies",
        metadata={
            "query": query,
            "status": status,
            "page_size": str(page_size),
            "study_count": str(result.study_count),
            "total_count": str(result.total_count) if result.total_count is not None else "",
            "next_page_token": result.next_page_token,
        },
    )
    write_json(manifest_output, manifest_to_json_record(manifest))

    print(f"Wrote raw ClinicalTrials.gov response to {raw_output}")
    print(f"Wrote {len(trials)} normalized ClinicalTrials.gov studies to {processed_output}")
    print(f"Wrote source manifest to {manifest_output}")


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


def report_trial_corpus(
    trials_path: Path,
    output_path: Path | None,
    sample_size: int,
    top_n: int,
) -> None:
    if sample_size < 0:
        raise ValueError("Sample size must be non-negative")
    if top_n < 1:
        raise ValueError("Top-N must be at least 1")
    trials = [trial_from_flat_record(row) for row in read_jsonl(trials_path)]
    report = summarize_trial_corpus(trials, sample_size=sample_size, top_n=top_n)
    if output_path:
        write_json(output_path, report)
        print(f"Wrote trial corpus report to {output_path}")
    else:
        for key, value in report.items():
            print(f"{key}: {value}")


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


def write_manifest(
    *,
    name: str,
    source_url: str,
    input_path: Path,
    output_path: Path,
    dataset: str,
    year: int | None,
    parser: str,
    metadata_items: list[str],
) -> None:
    manifest = build_source_manifest(
        name=name,
        source_url=source_url,
        input_path=input_path,
        dataset=dataset,
        year=year,
        parser=parser,
        metadata=parse_metadata_items(metadata_items),
    )
    write_json(output_path, manifest_to_json_record(manifest))
    print(f"Wrote source manifest to {output_path}")


def parse_metadata_items(items: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid metadata item {item!r}; expected key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid metadata item {item!r}; key cannot be empty")
        metadata[key] = value.strip()
    return metadata


def slugify(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in slug.split("_") if part) or "all"


if __name__ == "__main__":
    main()
