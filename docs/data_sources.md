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

The small live-query workflow uses:

- Endpoint: `https://clinicaltrials.gov/api/v2/studies`
- General search parameter: `query.term`
- Recruitment status filter: `filter.overallStatus`
- Page size parameter: `pageSize`
- JSON format parameter: `format=json`
- Total-count flag: `countTotal=true`

Recommended local layout:

```text
data/raw/clinicaltrials/studies.sample.json
data/raw/clinicaltrials/studies.json
data/raw/clinicaltrials/studies.json.zip
data/processed/clinicaltrials/studies.jsonl
data/manifests/clinicaltrials_<snapshot>_studies.json
```

The initial parser supports ClinicalTrials.gov v2 JSON records shaped like:

```text
protocolSection.identificationModule
protocolSection.statusModule
protocolSection.conditionsModule
protocolSection.designModule
protocolSection.armsInterventionsModule
protocolSection.eligibilityModule
protocolSection.contactsLocationsModule
```

The normalized trial contract currently includes:

- `nct_id`
- `title`
- `status`
- `conditions`
- `interventions`
- `eligibility_criteria`
- `sex`
- `minimum_age`
- `maximum_age`
- `phases`
- `study_type`
- `locations`
- `source`

Public repo rule: commit parser code, tests, docs, and tiny synthetic fixtures only. Do not commit full API responses, downloaded JSON archives, normalized corpora, generated indexes, or local manifests without reviewing them first.

Example live query:

```bash
ctmatch download-ctgov-studies \
  --query asthma \
  --status RECRUITING \
  --page-size 25 \
  --raw-output data/raw/clinicaltrials/asthma_recruiting_25.json \
  --manifest-output data/manifests/clinicaltrials_asthma_recruiting_25.json \
  --processed-output data/processed/clinicaltrials/asthma_recruiting_25.jsonl
```

Example validation report:

```bash
ctmatch report-trial-corpus \
  --trials data/processed/clinicaltrials/asthma_recruiting_25.jsonl \
  --output outputs/clinicaltrials_asthma_recruiting_25_report.json
```

Example BM25 search:

```bash
ctmatch search-trials-bm25 \
  --trials data/processed/clinicaltrials/asthma_recruiting_25.jsonl \
  --query "adult persistent asthma inhaled corticosteroid" \
  --top-k 10 \
  --output outputs/clinicaltrials_asthma_recruiting_25_search.json
```

The report summarizes:

- total trial rows and unique NCT IDs
- duplicate NCT IDs
- recruitment status distribution
- missing eligibility criteria
- condition coverage and top conditions
- intervention coverage and top interventions
- a small sample of normalized records

The BM25 search output includes:

- query and retriever parameters
- corpus size
- ranked NCT IDs, titles, statuses, scores, and structured fields
- matched query terms
- a short snippet from searchable trial text

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
data/manifests/trec_<year>_topics.json
data/manifests/trec_<year>_qrels.json
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

### Source Manifests

Source manifests record reproducibility metadata for files stored locally under ignored paths. They include:

- manifest schema version
- dataset/name/year
- source URL
- local path
- SHA256 checksum
- byte size
- creation timestamp in UTC
- parser/schema label
- optional metadata

Example:

```bash
ctmatch write-manifest \
  --name trec_2021_topics \
  --dataset trec_clinical_trials \
  --year 2021 \
  --parser trec_topics_xml \
  --source-url https://trec.nist.gov/data/trials/topics2021.xml \
  --input data/raw/trec/2021/topics2021.xml \
  --output data/manifests/trec_2021_topics.json
```

For now, `data/manifests/` is ignored. Review generated manifests before deciding whether any small manifest files should become tracked project artifacts.

## Local Fixture Data

Files under `data/fixtures/` are synthetic and exist only to exercise code paths in tests and CI. They should not be mixed with benchmark results.
