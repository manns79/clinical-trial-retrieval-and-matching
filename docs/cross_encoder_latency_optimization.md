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
best next latency experiment is a sub-10 candidate-depth ablation using int8/256 `clinical_core`,
which preserves more clinical evidence than either 128-token profile. Depths such as 3, 5, and 8
should be compared against the same FP32 depth-10 quality reference. The selected official int8
file requires AVX2; any eventual deployment image must verify that CPU capability or provide a
different architecture-specific artifact.
