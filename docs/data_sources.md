# Data Sources

## ClinicalTrials.gov

ClinicalTrials.gov provides a modern public API under `/api/v2/` and full study downloads in JSON. The ingestion layer should record:

- Source URL or local archive path
- API version and data timestamp when available
- Download timestamp
- File checksum
- Record counts
- Field mapping version

The benchmark pipeline should use frozen snapshots so retrieval metrics remain reproducible.

## TREC Clinical Trials

The TREC Clinical Trials tracks provide patient topics and relevance judgments:

- 2021 and 2022: free-text synthetic patient case descriptions
- 2023: questionnaire-style synthetic patient topics
- Qrels: graded labels where `0` is non-relevant, `1` is excluded, and `2` is eligible

Use one track/year for development and another as a held-out evaluation when possible.

## Local Fixture Data

Files under `data/fixtures/` are synthetic and exist only to exercise code paths in tests and CI. They should not be mixed with benchmark results.
