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

### Staged startup profile

A fresh-process rerun on 2026-08-17 added checkpoints around the production preload path. The
same tracked configuration and local artifacts were used; filesystem caches may have differed
from the original run.

| startup phase | elapsed seconds | retained RSS delta MiB | peak RSS delta MiB |
| --- | ---: | ---: | ---: |
| API import | 0.627 | 20.141 | 20.168 |
| normalized corpus | 1.616 | 199.145 | 199.352 |
| fielded BM25 | 12.838 | 2,835.586 | 2,842.660 |
| dense index and model | 15.777 | 598.199 | 628.223 |

This run reached 3,675.750 MiB RSS after startup and 3,733.715 MiB after warm measurements.
Fielded BM25 accounted for 78.05% of the positive retained RSS increase across resource phases.
The 110.296 MiB pickle expands into Python counters, dictionaries, strings, tuples, and integer
objects, confirming the lexical representation as the primary deployment-memory constraint.

## 2026-08-17 decision

Warm latency has plausible room for a small reranker, particularly after dense retrieval, but the
current shared process already requires about 3.7 GiB RSS. Adding a cross-encoder now would increase
both memory and cold-start pressure and would make a free-tier public deployment unlikely.

Deployment optimization should come first. The staged profile confirms that the next experiment
should replace the expanded in-memory fielded BM25 representation with a compact disk-backed
lexical backend and compare development retrieval quality, latency, startup, and memory against
the frozen lexical baseline. SQLite FTS5 is the first candidate because it is local, free, and
available through Python's standard library. Dense retrieval itself remains fast and its
persisted index is compact enough that it is not the first optimization target.

The detailed benchmark JSON remains an ignored local artifact and contains no patient records or
search results.

## SQLite FTS5 outcome

On 2026-08-18, the development-quality and isolated resource comparisons supported replacing
preloaded Python fielded BM25 with SQLite FTS5. The full serving benchmark was then rerun with
SQLite FTS5, MiniLM, and the dense index loaded together.

| measurement | Python BM25 serving | SQLite FTS5 serving |
| --- | ---: | ---: |
| lexical preload retained RSS MiB | 2,835.586 | 2.973 |
| process peak RSS MiB | 3,733.715 | 1,119.102 |
| lexical handler p50 ms | 146.128 | 111.962 |
| lexical handler p95 ms | 180.478 | 142.086 |
| hybrid handler p50 ms | 182.332 | 156.071 |
| hybrid handler p95 ms | 241.202 | 176.460 |

These values came from different fresh-process runs and may reflect filesystem-cache and host
variance, but the memory result is large enough to be decisive. The dense index/model is now the
dominant startup resource, retaining 814.586 MiB in the latest run. The detailed quality and
isolated resource comparison is recorded in `docs/sqlite_fts5_selection.md`.

## Dense startup attribution and optimization outcome

On 2026-08-19, the SQLite serving profile was instrumented with separate checkpoints for the
dense embedding index, sentence-transformer encoder, and retriever assembly. The baseline was
then compared with a bit-for-bit-equivalent NumPy memory map and a development-only dynamic-int8
query-encoder ablation. The local deployment planning budget is a 1 GiB single-worker container
with a 900 MiB process peak, leaving 124 MiB for runtime and container overhead.

| profile | index RSS delta MiB | encoder RSS delta MiB | startup RSS MiB | peak RSS MiB | hybrid p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| compressed NPZ + FP32 | 47.332 | 766.102 | 1,059.688 | 1,117.668 | 223.999 |
| memory map + FP32 | 7.555 | 769.219 | 1,022.949 | 1,119.324 | 216.671 |
| memory map + dynamic int8 | 7.523 | 821.898 | 1,075.773 | 1,132.316 | 240.471 |

The retriever assembly retained 0 MiB in all profiles after changing it to preserve the existing
corpus tuple. The mmap conversion preserved NCT order and every float32 embedding exactly. It
reduced pre-search index residency by about 39.8 MiB, but warm retrieval paged the matrix into
memory, so it did not reduce process peak or meet the deployment budget. Artifact size increased
from 35.636 MiB to 38.680 MiB because the mmap array is intentionally uncompressed.

Dynamic int8 is rejected. Even with in-place conversion, this PyTorch/sentence-transformers stack
retained about 52.7 MiB more encoder memory than FP32 and missed the 900 MiB process target.
Development eligible-only Recall@100 also dropped from 0.349217 to 0.310034, while eligible
nDCG@10 dropped from 0.327490 to 0.283981. It fails both the memory and retrieval-quality gates
and is not enabled in the default serving config.

Mmap remains an explicit reproducible candidate rather than the default. It improves idle startup
residency but does not create cross-encoder headroom under the 900 MiB process target. Cold-start
timings varied substantially with filesystem cache state, so the memory and quality findings are
the decision evidence; timing differences are not treated as a reliable win.

All detailed reports, trial records, indexes, model files, and diagnostics remain ignored. This
tracked section contains aggregate measurements and synthetic-query latency only.

## Disk-backed trial metadata outcome

On 2026-08-20, primary serving modes stopped retaining the 26,150 normalized `Trial` objects.
A streamed build now persists normalized records and corpus-integrity metadata in an ignored
SQLite store. Startup validates the source-file checksum, corpus fingerprint, row count, and dense
NCT ID order. Search batch-loads only returned records for snippets, while `/trial/{nct_id}` loads
one record by primary key. Legacy Python BM25 remains available on demand and is the only serving
mode that can still load the full JSONL corpus.

Ranking parity was exact on all five synthetic serving queries for SQLite FTS5 top 100, dense top
100, and hybrid top 10. The adopted FP32/NPZ profile produced these local measurements:

| measurement | in-memory corpus | SQLite trial store |
| --- | ---: | ---: |
| corpus/store startup RSS delta MiB | 199.207 | 0.590 |
| RSS after startup MiB | 1,059.688 | 910.074 |
| warm process peak MiB | 1,117.668 | 968.148 |
| cold start seconds | 27.581 | 7.581 |
| hybrid handler p95 ms | 223.999 | 156.878 |

Filesystem cache state and model allocator variance affect timing and encoder memory, so the
startup-phase reduction and ranking parity are the strongest evidence. The 109.215 MiB store adds
no hosted-service cost. On-demand metadata p95 was 7.461 ms for SQLite FTS5, 11.661 ms for dense,
and 6.997 ms for hybrid in this run.

The mmap/store combination lowered idle RSS after startup further to 871.293 MiB, but warm peak
remained 967.660 MiB once retrieval paged the embedding matrix. It is therefore not adopted over
the simpler compressed NPZ default. The selected profile passes the 15-second cold-start and
250 ms hybrid-p95 targets but remains 68.148 MiB above the 900 MiB peak-process target. A
cross-encoder is still deferred.

## Encoder phase split and ONNX Runtime outcome

On 2026-08-22, encoder startup was split into framework import, model/session construction, and
first-inference/thread-pool initialization. A free local FP32 ONNX Runtime export of the selected
MiniLM query encoder was
then evaluated against the same persisted corpus embeddings. Dense scores are rounded to six
decimals for ranking and exact ties resolve by NCT ID, making near-tie behavior independent of
small backend floating-point differences.

The ONNX development run matched the sentence-transformers run exactly for all 60 development
topics through rank 100. Consequently, eligible-only and excluded-or-eligible metrics were also
identical. No holdout topics were used.

| measurement | sentence-transformers | ONNX Runtime |
| --- | ---: | ---: |
| framework import retained RSS MiB | 776.617 | 18.895 |
| model/session retained RSS MiB | 32.625 | 126.031 |
| first inference retained RSS MiB | 53.480 | 2.492 |
| RSS after startup MiB | 963.098 | 247.793 |
| peak process RSS MiB | 970.090 | 271.902 |
| cold start seconds | 8.447 | 2.489 |
| dense handler p95 ms | 59.319 | 87.811 |
| hybrid handler p95 ms | 181.997 | 226.972 |
| query encoder artifact MiB | 87.0 model cache | 86.879 export |

The sentence-transformers split shows that importing the PyTorch-based framework, rather than
constructing MiniLM alone, caused most retained memory. The ONNX profile passed the versioned
limits of 900 MiB peak RSS, 15 seconds cold start, and 250 ms hybrid p95. It is adopted as the
selected serving backend. Its latency is slower in this local run but remains within budget and
leaves about 628 MiB below the process-memory limit for the next measured reranking experiment.

Exported model files, tokenizers, run files, parity reports, and benchmark JSON remain ignored.
The tracked configs and this aggregate summary contain no patient records or detailed trial data.
