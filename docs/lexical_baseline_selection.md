# Lexical baseline selection

## Protocol

TREC 2021 topics were deterministically partitioned into 60 development topics and 15 holdout
topics with the tracked `ctmatch-trec-2021-v1` split seed. Candidate selection used only the
development partition. Eligible-only Recall@100 was the primary selection metric because this
stage is intended to retrieve a broad candidate set containing truly eligible trials. Eligible-
only MRR and nDCG were tie-breakers; the broader excluded-or-eligible view remained diagnostic.

The project inspected full TREC 2021 aggregate results before this protocol was introduced.
Therefore, the holdout is a useful workflow check but is not presented as a historically unseen
test set. A later TREC year should be used for stronger external validation.

## Development selection

| Profile | Eligible P@10 | Eligible R@100 | Eligible MRR | Eligible nDCG@10 | Eligible nDCG@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain BM25 | 0.203333 | 0.212451 | 0.408343 | 0.219862 | 0.219378 |
| Fielded BM25 | 0.231667 | 0.259718 | 0.415182 | 0.236307 | 0.259010 |
| Condition/title v1 | 0.231667 | **0.265228** | **0.445300** | **0.242517** | **0.264901** |

`condition_title_v1` was selected and frozen. Relative to the default fielded baseline, it
improved eligible-only Recall@100 by 0.005510 absolute, MRR by 0.030118, nDCG@10 by 0.006210,
and nDCG@100 by 0.005891, while Precision@10 was unchanged.

## Frozen weights

| Field | Weight |
| --- | ---: |
| All text | 1.00 |
| Title | 1.25 |
| Brief summary | 0.75 |
| Conditions | 1.50 |
| Interventions | 0.25 |
| Eligibility criteria | 0.50 |
| Demographics | 0.10 |
| Recruitment status | 0.05 |
| Locations | 0.05 |

## Holdout result

The frozen profile was evaluated once on August 12, 2026. It retrieved 100 trials for each of
15 holdout topics against the 26,150-trial corpus and 6,798 holdout judgments.

| Relevance view | P@10 | R@100 | MRR | nDCG@10 | nDCG@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eligible only | 0.300000 | 0.313912 | 0.483651 | 0.293425 | 0.312885 |
| Excluded or eligible | 0.626667 | 0.344873 | 0.644444 | 0.597100 | 0.514651 |

All 15 topics produced results, and one topic met the existing weak-retrieval diagnostic. The
generated metrics report records experiment config SHA-256
`751a95c86967fcb352dcc26450b76c550639de2867e020f807eec84629f01670`.

This closes lexical field-weight selection on TREC 2021. The holdout result will not be used to
revise those weights. Generated run files, detailed diagnostics, split records, and qrels remain
ignored and are not committed.
