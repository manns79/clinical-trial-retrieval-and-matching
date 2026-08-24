# Cross-encoder reranking baseline

On 2026-08-23, the selected SQLite FTS5 plus MiniLM hybrid development run was reranked with the
public `cross-encoder/ms-marco-MiniLM-L6-v2` model at pinned revision `233902d`. The experiment
used the repository's official FP32 ONNX export through ONNX Runtime, CPU batch size 16, maximum
sequence length 256, and the `clinical_core` trial representation. No holdout topics were read.

The experiment reranked only the configured candidate window and appended the original hybrid
tail through rank 100. This keeps Recall@100 fixed and isolates ordering changes. All model files,
trial records, TREC runs, metrics, diagnostics, and detailed timing samples remain ignored.

## Development quality and latency

| depth | eligible MRR delta | eligible nDCG@10 delta | broad nDCG@10 delta | rerank mean ms/topic | rerank p95 ms/topic |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | +0.044120 | +0.004185 | +0.004129 | 362.967 | 451.980 |
| 25 | +0.035849 | -0.014092 | -0.027112 | 881.245 | 1,050.396 |
| 50 | +0.048605 | -0.000106 | -0.067757 | 1,803.538 | 1,988.468 |

Depth 10 is the only candidate with positive eligible-only and broad nDCG@10 changes. Its
eligible-only MRR gain is meaningful, but its reranking p95 alone is about twice the complete
250 ms hybrid-handler budget. Depths 25 and 50 are slower and degrade multiple ranking metrics.

## Combined serving memory

The headroom check loaded the selected ONNX retrieval stack, then the cross-encoder, then executed
real 10-, 25-, and 50-candidate probes. This captures ONNX Runtime arena growth from larger input
batches rather than reporting only one-pair warmup memory.

| depth probe | combined process RSS/peak MiB | one-topic probe ms |
| ---: | ---: | ---: |
| 10 | 456.051 | 531.323 |
| 25 | 546.699 | 1,174.290 |
| 50 | 635.770 | 1,921.891 |

The selected retrieval stack retained 248.445 MiB before reranker scoring. The depth-50 process
peak was 635.770 MiB, passing the 900 MiB memory limit with about 264 MiB remaining. The ignored
ONNX artifact occupies 87.696 MiB. Memory is therefore acceptable; latency is the adoption blocker.

## Decision

Do not add this FP32 profile to `/search`. Keep depth 10 as the quality reference and next test an
official int8 ONNX export and shorter sequence/text representations. Any serving candidate must
preserve the depth-10 development gain and use an explicit reranked-mode latency budget. The
general-domain MS MARCO model is a baseline, not evidence of clinical suitability or eligibility.

The follow-up optimization is recorded in `cross_encoder_latency_optimization.md`. The tested
official AVX2 int8, 128-token, and shorter-text profiles did not pass both quality and latency
gates, so the serving decision remains unchanged.
