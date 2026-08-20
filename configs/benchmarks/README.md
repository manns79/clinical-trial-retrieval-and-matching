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
and cache the configured model before benchmarking. A fresh CLI process measures API import and
resource preload once, performs deterministic warmups, and then interleaves SQLite FTS5, dense,
and hybrid requests to reduce run-order bias. Cold process startup may still benefit from the
operating system filesystem cache.

Configs and tracked summaries must contain only synthetic queries and aggregate measurements.
Never add patient data, benchmark records, model files, indexes, secrets, or detailed search
results. JSON reports remain under ignored `outputs/` paths.

The optimized candidate is deliberately separate from the default serving profile. It uses a
directory-backed NumPy memory map and dynamic int8 quantization for the query encoder:

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
