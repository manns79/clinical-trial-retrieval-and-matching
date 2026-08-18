# Dense and hybrid development selection

> Historical experiment note: the original hybrid below uses the frozen Python BM25 run. Serving
> now uses the development-validated SQLite FTS5 lexical component documented in
> `docs/sqlite_fts5_selection.md`; the original result remains intact for reproducibility.

## Scope and selection rule

This record uses only the deterministic 60-topic TREC 2021 development partition. The
15-topic lexical holdout was not read or evaluated while selecting a dense profile or building
the hybrid. Eligible-only Recall@100 was the primary dense selection metric, with MRR and nDCG
used as secondary evidence.

All trial records, benchmark inputs, embeddings, run files, diagnostics, and detailed outputs
remain ignored local artifacts. This tracked record contains aggregate metrics only.

## Dense profile comparison

| profile | eligible P@10 | eligible R@100 | eligible MRR | eligible nDCG@10 | eligible nDCG@100 | weak topics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM, `title_summary_conditions` | 0.303333 | 0.349217 | 0.568825 | 0.327490 | 0.343169 | 2 |
| MedEmbed-small, `eligibility_snapshot` | 0.245000 | 0.243348 | 0.454623 | 0.255993 | 0.238521 | 8 |

The selected dense profile is `sentence-transformers/all-MiniLM-L6-v2` with the
`title_summary_conditions` representation. It leads the biomedical profile on every reported
eligible-only metric and has fewer weak topics.

This is a profile ablation rather than a controlled encoder-only experiment because both the
encoder and text representation changed. It answers which of these two deployable profiles is
stronger, but it does not isolate why. The biomedical profile uses the Apache-2.0 licensed,
medical-retrieval-oriented `abhinand/MedEmbed-small-v0.1` model with a concise title,
conditions, demographics, and eligibility representation.

## Reciprocal-rank fusion

The development hybrid uses equal-weight reciprocal-rank fusion with `k=60`, fusing the top 100
results from:

- Frozen lexical profile `fielded_bm25_condition_title_v1`
- Selected dense profile `dense_all_minilm_l6_v2`

| retriever | eligible P@10 | eligible R@100 | eligible MRR | eligible nDCG@10 | eligible nDCG@100 | weak topics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen lexical | 0.231667 | 0.265228 | 0.445300 | 0.242517 | 0.264901 | 2 |
| selected dense | 0.303333 | 0.349217 | 0.568825 | 0.327490 | 0.343169 | 2 |
| equal-weight RRF | 0.318333 | 0.349395 | 0.514742 | 0.321859 | 0.337757 | 1 |

RRF provides the best eligible Precision@10 and Recall@100 and leaves one weak topic. Dense alone
retains the best eligible MRR and nDCG, so the hybrid is a useful coverage-oriented profile, not
a uniform improvement. On the broader excluded-or-eligible view, RRF improves Recall@100 to
0.364493, MRR to 0.856004, and nDCG@10 to 0.720473.

At the topic level, RRF eligible recall beats/ties/loses to dense on 28/6/26 topics and to lexical
on 42/6/12 topics. The lexical weak-topic set is `{34, 42}`, the dense set is `{42, 66}`, and the
hybrid leaves only `{42}` under the existing diagnostics policy.

## Reproduction

```bash
make run-trec-2021-dense-biomedical
make compare-trec-2021-dense-ablation
make run-trec-2021-hybrid
make compare-trec-2021-retrievers
```

The first biomedical run downloads a public model and builds an ignored local NumPy index. Later
runs validate and reuse the persisted index. The hybrid command requires the existing frozen
lexical and selected dense development run files.
