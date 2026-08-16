# Local serving performance baseline

## Scope

This baseline was recorded on 2026-08-16 from the tracked
`trec_2021_local_serving` configuration with SHA-256
`bb6693534aebc204f077f0933c658d8c47931dd8242d13f1fda0d571083b4040`. It used the
26,150-trial local corpus, five synthetic queries, one warmup round, five measurement rounds,
and `top_k=10`.

The machine ran Linux under WSL2 on x86-64 with 16 logical CPUs, Python 3.12.3, NumPy 2.5.1,
PyTorch 2.13.0, and sentence-transformers 5.7.0. These are local observations, not portable
service-level objectives.

Cold start measures API import plus corpus, lexical index, dense index, and locally cached model
preload. Warm latency calls the FastAPI handler in-process and excludes HTTP transport and JSON
serialization. The process was fresh, but operating-system filesystem caches may have been warm.
Throughput is sequential single-process capacity, not concurrent load capacity.

## Results

Cold start was 18.309 seconds. Warm measurements covered 25 requests per mode.

| mode | mean ms | p50 ms | p95 ms | max ms | sequential req/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| fielded BM25 | 149.280 | 146.128 | 180.478 | 186.815 | 6.699 |
| dense | 46.975 | 44.869 | 77.102 | 93.564 | 21.288 |
| hybrid RRF | 192.110 | 182.332 | 241.202 | 301.844 | 5.205 |

Aggregate interleaved throughput was 7.702 requests per second.

## Memory and artifacts

| measurement | MiB |
| --- | ---: |
| process RSS before startup | 22.496 |
| process RSS after startup | 3,678.734 |
| process RSS after benchmark / observed peak | 3,736.758 |
| normalized corpus file | 88.704 |
| persisted fielded BM25 index | 110.296 |
| persisted dense NumPy index | 35.636 |
| selected dense model cache | 87.336 |

Corpus and persisted indexes total 234.636 MiB on disk. The much larger in-memory footprint shows
that deserialized serving structures are the deployment constraint rather than artifact storage.
The expanded Python lexical index and duplicated trial text are the leading suspects, but staged
resource profiling should confirm their individual contributions.

## Decision

Warm latency has plausible room for a small reranker, particularly after dense retrieval, but the
current shared process already requires about 3.7 GiB RSS. Adding a cross-encoder now would increase
both memory and cold-start pressure and would make a free-tier public deployment unlikely.

Deployment optimization should come first. The next investigation should isolate corpus,
lexical-index, and dense-resource memory deltas, then replace or redesign the expanded in-memory
fielded BM25 representation. Dense retrieval itself is fast and its persisted index is compact;
the lexical serving path is the stronger optimization target.

The detailed benchmark JSON remains an ignored local artifact and contains no patient records or
search results.
