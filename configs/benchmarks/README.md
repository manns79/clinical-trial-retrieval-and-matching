# Serving benchmark registry

Each JSON file in this directory defines a reproducible local serving benchmark. A spec pins the
corpus and persisted indexes, dense model configuration, RRF settings, synthetic queries, warmup
and measurement counts, and an ignored output path.

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
