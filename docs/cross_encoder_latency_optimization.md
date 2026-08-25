# Cross-encoder latency optimization

On 2026-08-24, four depth-10 profiles were evaluated on the TREC 2021 development split. The
reference used the FP32 ONNX MS MARCO MiniLM cross-encoder with 256 tokens and `clinical_core`
text. Sequential ablations changed the model to the official AVX2 int8 export, reduced maximum
length to 128, and replaced the trial text with `title_summary_conditions`. No holdout topics were
read.

## Gates

A serving candidate had to match or exceed both reference nDCG@10 gains with zero tolerance:

- eligible-only gain: +0.004185
- excluded-or-eligible gain: +0.004129

The provisional optional reranked mode has a 500 ms p95 planning budget. Reserving the existing
250 ms hybrid allowance leaves 250 ms p95 for incremental reranking. Estimated total p95 below is
the conservative sum of those allowances; it is not a concurrent HTTP load-test measurement.

## Development results

| profile | eligible nDCG@10 gain | broad nDCG@10 gain | reranker p95 ms | estimated mode p95 ms | artifact MiB | peak RSS MiB | quality | latency | adopt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| FP32 / 256 / clinical core | +0.004185 | +0.004129 | 517.192 | 767.192 | 87.703 | 450.660 | pass | fail | no |
| int8 / 256 / clinical core | +0.002039 | +0.006684 | 402.504 | 652.504 | 23.034 | 394.488 | fail | fail | no |
| int8 / 128 / clinical core | -0.001192 | -0.002138 | 270.368 | 520.368 | 23.034 | 382.266 | fail | fail | no |
| int8 / 128 / short text | -0.006382 | -0.015892 | 206.118 | 456.118 | 23.034 | 260.180 | fail | pass | no |

Quantization reduced artifact size by about 74% and improved latency and memory. Reducing context
and removing eligibility/intervention evidence crossed the latency gate, but reversed both nDCG
gains. The int8/256 profile improved broad nDCG beyond the reference, yet retained less than half
the eligible-only gain and remained too slow.

## Decision

Do not enable reranking in `/search` and do not weaken the gates after observing the results. The
selected official int8 file requires AVX2; any eventual deployment image must verify that CPU
capability or provide a different architecture-specific artifact.

## Final small-window gate

On 2026-08-25, int8/256 `clinical_core` depths 3, 5, and 8 were compared with the same FP32/depth-10
quality reference and zero-tolerance gates.

| depth | eligible nDCG@10 gain | broad nDCG@10 gain | reranker p95 ms | estimated mode p95 ms | quality | latency | adopt |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 3 | +0.001747 | +0.002227 | 153.143 | 403.143 | fail | pass | no |
| 5 | +0.003243 | +0.002388 | 186.603 | 436.603 | fail | pass | no |
| 8 | +0.007137 | +0.008893 | 275.995 | 525.995 | pass | fail | no |

No profile passes both gates. Depth 8 is a useful near miss because it improves both quality views,
but the gate was defined before measurement and is not relaxed afterward. The reranking
investigation is therefore closed for the current project phase. The code and experiment records
remain reproducible evidence, while `/search` continues to expose only lexical, dense, and hybrid
retrieval.
