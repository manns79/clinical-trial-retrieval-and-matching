# Serving benchmark registry

Each JSON file in this directory defines a reproducible local serving benchmark. A spec pins the
corpus and persisted indexes, dense model configuration, RRF settings, synthetic queries, warmup
and measurement counts, and an ignored output path.

Build the disk-backed metadata store before running the TREC benchmark:

```bash
make build-trec-2021-trial-store
```

Run the TREC 2021 local benchmark:

```bash
ctmatch benchmark-serving \
  --config configs/benchmarks/trec_2021_local_serving.json
```

The command runs with Hugging Face and Transformers offline modes enabled. Build the dense index
and export the selected local ONNX encoder before benchmarking. A fresh CLI process measures API
import, framework import, model/session construction, first-inference/thread-pool initialization,
then interleaves SQLite FTS5, dense, and hybrid requests. Cold startup may still benefit from the
operating system filesystem cache.

```bash
make install-onnx
make export-trec-2021-onnx-encoder
make run-trec-2021-dense-onnx
make check-trec-2021-dense-onnx-parity
make benchmark-trec-2021-serving
make assess-trec-2021-serving-budget
```

The selected profile must pass exact development top-100 NCT-order parity and the versioned
single-worker deployment budget. The PyTorch baseline remains reproducible with
`make benchmark-trec-2021-serving-sentence-transformers`.

Configs and tracked summaries must contain only synthetic queries and aggregate measurements.
Never add patient data, benchmark records, model files, indexes, secrets, or detailed search
results. JSON reports remain under ignored `outputs/` paths.

The rejected mmap/int8 candidate remains separate from the selected serving profile:

```bash
make convert-trec-2021-dense-mmap
make run-trec-2021-dense-mmap-int8
make compare-trec-2021-dense-optimization
make benchmark-trec-2021-serving-mmap-int8
make assess-trec-2021-serving-budget
```

The budget is a local planning target for a single worker in a 1 GiB container. Its 900 MiB peak
process limit intentionally reserves 124 MiB for container/runtime overhead. These commands are
offline and use no paid service. The mmap index, model cache, benchmark records, and reports stay
in ignored paths; only aggregate, non-patient summaries are suitable for version control.
