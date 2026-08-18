from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.benchmarking.lexical import (
    benchmark_lexical_backend,
    compare_lexical_backend_reports,
    write_lexical_backend_comparison,
)
from clinical_trial_matching.ingestion.clinicaltrials import trial_from_flat_record
from clinical_trial_matching.io import read_jsonl
from clinical_trial_matching.retrieval.sqlite_fts import (
    DEFAULT_SQLITE_FTS_FIELD_WEIGHTS,
    build_sqlite_fts_index,
)


class LexicalBenchmarkTest(unittest.TestCase):
    def test_sqlite_benchmark_and_comparison_report(self) -> None:
        corpus_path = Path("data/fixtures/trials.sample.jsonl")
        trials = [trial_from_flat_record(row) for row in read_jsonl(corpus_path)]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_path = root / "fts.sqlite"
            output_path = root / "fts.json"
            build_sqlite_fts_index(
                index_path,
                trials,
                corpus_path=corpus_path,
            )
            report = benchmark_lexical_backend(
                backend="sqlite-fts5",
                corpus_path=corpus_path,
                index_path=index_path,
                field_weights=DEFAULT_SQLITE_FTS_FIELD_WEIGHTS,
                queries=["adult asthma inhaler"],
                warmup_rounds=1,
                measurement_rounds=2,
                top_k=2,
                output_path=output_path,
            )
            comparison = compare_lexical_backend_reports(output_path, output_path)
            markdown_path = root / "comparison.md"
            write_lexical_backend_comparison(markdown_path, comparison)

            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(report["backend"], "sqlite-fts5")
        self.assertEqual(report["warm"]["requests"], 2)
        self.assertEqual(report["cost"]["hosted_service_cost_usd"], 0.0)
        self.assertEqual(comparison["candidate_ratios"]["warm_p50"], 1.0)
        self.assertIn("retriever_rss_delta_mib", markdown)


if __name__ == "__main__":
    unittest.main()
