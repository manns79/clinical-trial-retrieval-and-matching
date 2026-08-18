# SQLite FTS5 serving selection

## Scope

The SQLite FTS5 candidate was evaluated on the 60-topic TREC 2021 development partition only.
It indexes title, brief summary, conditions, interventions, and eligibility criteria with the
selected lexical weights. The existing 15-topic holdout was not read or rerun for this serving
optimization decision.

Both backends used the same 26,150 normalized trials and `top_k=100`. The Python BM25 run remains
the frozen scientific baseline; SQLite FTS5 is a serving backend selected for its quality/resource
tradeoff. These local measurements are not portable service-level objectives.

## Development quality

| view | backend | precision@10 | recall@100 | MRR | nDCG@10 | nDCG@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| eligible only | fielded BM25 | 0.231667 | 0.265228 | 0.445300 | 0.242517 | 0.264901 |
| eligible only | SQLite FTS5 | 0.243333 | 0.273024 | 0.417288 | 0.249625 | 0.269547 |
| excluded or eligible | fielded BM25 | 0.615000 | 0.310313 | 0.773307 | 0.627143 | 0.506062 |
| excluded or eligible | SQLite FTS5 | 0.581667 | 0.316410 | 0.784325 | 0.599117 | 0.505924 |

SQLite improves eligible-only precision, recall, and both nDCG measures while reducing eligible
MRR by 6.29%. In the broader view it improves recall and MRR while reducing precision@10 and
nDCG@10 by 5.42% and 4.47%. This is competitive rather than uniformly superior quality.

The serving hybrid was also reevaluated on development topics after replacing its lexical
component. Equal-weight SQLite FTS5 plus MiniLM achieved eligible-only Recall@100 of 0.355085 and
nDCG@10 of 0.326465, compared with 0.349395 and 0.321859 for BM25 plus MiniLM. Eligible MRR moved
from 0.514742 to 0.506290. In the excluded-or-eligible view, Recall@100 improved from 0.364493 to
0.370910 while nDCG@10 moved from 0.720473 to 0.710610. These modest tradeoffs support adopting
SQLite in hybrid serving as well as standalone lexical serving.

## Resource comparison

Each backend ran in a separate fresh CLI process over five synthetic queries, one warmup round,
five measurement rounds, and `top_k=10`. Operating-system filesystem caches may have been warm.

| backend | cold start ms | retriever RSS delta MiB | p50 ms | p95 ms | req/s | index MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fielded BM25 | 15,013.930 | 2,835.734 | 93.168 | 102.833 | 10.886 | 110.296 |
| SQLite FTS5 | 2,081.544 | 2.949 | 67.154 | 74.844 | 14.877 | 118.016 |

The SQLite file is 7.7 MiB larger, but its retriever retains roughly 0.10% as much process RSS.
It also starts about 7.2 times faster and improves warm latency in this isolated comparison.

After adoption, the full API process with SQLite FTS5, MiniLM, and the dense index peaked at
1,119.102 MiB instead of the prior 3,733.715 MiB. SQLite added 2.973 MiB during preload; the dense
index/model is now the dominant startup resource.

## Decision

SQLite FTS5 is adopted as the default lexical serving backend and as the lexical component of
hybrid RRF. The frozen Python fielded BM25 baseline remains reproducible for evaluation and is
available as an explicit API comparison mode, but it is no longer preloaded.

The next optimization target is dense model loading and corpus duplication. A cross-encoder
should wait until we determine whether the roughly 1.1 GiB full process fits the intended free
deployment tier with enough headroom for request spikes and worker overhead.

All detailed runs, reports, indexes, model files, and benchmark data remain ignored local assets.
