# BM25 experiment registry

Each JSON file in this directory is a versioned, reviewable BM25 experiment specification.
The config pins the retriever, every field weight, benchmark inputs, and ignored artifact paths.
Paths are resolved from `project_root`, which is relative to the config file.

Run one experiment from any working directory:

```bash
ctmatch run-bm25-experiment \
  --config configs/experiments/trec_2021/fielded_bm25.json
```

Use `--rebuild-index` after changing a corpus or when intentionally replacing an index.
Benchmark data, indexes, run files, metrics, and diagnostics remain ignored. Configs must never
contain patient data, secrets, tokens, or copied benchmark records.

The TREC 2021 tuning specs read only the deterministic development partition. Choose a profile
from development metrics, freeze it, and then create one holdout spec for final evaluation.
Do not repeatedly compare candidates on holdout metrics.
