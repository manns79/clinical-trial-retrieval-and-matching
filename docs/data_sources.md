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

Official source pages:

- TREC Clinical Trials data index: https://trec.nist.gov/data/trials.html
- 2021 topics: https://trec.nist.gov/data/trials/topics2021.xml
- 2021 qrels: https://trec.nist.gov/data/trials/qrels2021.txt
- 2022 topics and qrels: https://trec.nist.gov/data/trials2022.html
- 2023 topics and qrels: https://trec.nist.gov/data/trials2023.html

Recommended local layout:

```text
data/raw/trec/2021/topics2021.xml
data/raw/trec/2021/qrels2021.txt
data/raw/trec/2022/topics2022.xml
data/raw/trec/2022/qrels2022.txt
data/raw/trec/2023/topics2023.xml
data/raw/trec/2023/qrels2023.txt
```

Normalized outputs should be generated under ignored paths:

```text
data/processed/trec/<year>/topics.jsonl
data/processed/trec/<year>/qrels.jsonl
outputs/trec_<year>_validation.json
```

The normalized topic contract is JSONL with:

- `topic_id`
- `year`
- `text`
- `format`, either `free_text` or `fields`
- `fields`, populated for 2023-style questionnaire topics
- `template`, when provided
- `source`

The normalized qrels contract is JSONL with:

- `topic_id`
- `nct_id`
- `relevance`
- `label`, one of `irrelevant`, `excluded`, or `eligible`
- `year`

Public repo rule: commit the parser, tests, docs, and small synthetic fixtures only. Do not commit downloaded TREC files, full ClinicalTrials.gov snapshots, generated indexes, or generated output reports unless they are deliberately small published benchmark summaries.

## Local Fixture Data

Files under `data/fixtures/` are synthetic and exist only to exercise code paths in tests and CI. They should not be mixed with benchmark results.
