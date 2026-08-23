# Retrieval experiment registry

Each JSON file in this directory is a versioned, reviewable retrieval experiment specification.
The config pins the retriever or fusion method, benchmark inputs, and ignored artifact paths.
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

The frozen TREC 2021 lexical winner and its aggregate selection evidence are documented in
`docs/lexical_baseline_selection.md`. Its `holdout_` config is retained for reproducibility,
but the holdout result must not be used for further field-weight tuning.

Dense specs pin a sentence-transformer model ID, named trial text representation, batch size,
device, sequence limit, development inputs, and ignored `.npz`/report artifacts. Dense candidates
also stay on the development partition until one is selected and frozen.

RRF specs pin named component run files, component weights, the RRF constant and candidate depth,
benchmark inputs, and ignored output paths. The selected dense profile and hybrid development
evidence are documented in `docs/dense_hybrid_selection.md`. Dense and hybrid candidates must not
be evaluated on the lexical holdout while they are still being selected.

Cross-encoder specs pin a model revision, ignored ONNX artifact, trial text representation,
candidate depths, selected development run, and serving-memory budget. Run the download,
development sweep, and combined-process headroom check with:

```bash
make download-trec-2021-cross-encoder
make run-trec-2021-cross-encoder
make benchmark-trec-2021-cross-encoder-headroom
```

The reranking sweep appends the untouched baseline tail after each reranked candidate window, so
Recall@100 remains comparable. Cross-encoder experiments remain development-only until both
quality and serving-resource gates justify a frozen profile.
