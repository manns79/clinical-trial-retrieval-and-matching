from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.evaluation.parity import trec_run_parity_report


class TrecRunParityTest(unittest.TestCase):
    def test_exact_rankings_pass_even_when_scores_and_run_names_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.run"
            candidate = root / "candidate.run"
            baseline.write_text(
                "1 Q0 NCT1 1 0.9 baseline\n1 Q0 NCT2 2 0.8 baseline\n",
                encoding="utf-8",
            )
            candidate.write_text(
                "1 Q0 NCT1 1 0.91 candidate\n1 Q0 NCT2 2 0.79 candidate\n",
                encoding="utf-8",
            )

            report = trec_run_parity_report(baseline, candidate, depth=2)

        self.assertTrue(report["passed"])
        self.assertEqual(report["topics"]["matching"], 1)

    def test_first_ranking_difference_fails_with_traceable_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.run"
            candidate = root / "candidate.run"
            baseline.write_text(
                "1 Q0 NCT1 1 0.9 baseline\n1 Q0 NCT2 2 0.8 baseline\n",
                encoding="utf-8",
            )
            candidate.write_text(
                "1 Q0 NCT2 1 0.9 candidate\n1 Q0 NCT1 2 0.8 candidate\n",
                encoding="utf-8",
            )

            report = trec_run_parity_report(baseline, candidate, depth=2)

        self.assertFalse(report["passed"])
        self.assertEqual(report["mismatch_sample"][0]["first_mismatch_rank"], 1)
        self.assertEqual(report["mismatch_sample"][0]["baseline_nct_id"], "NCT1")
        self.assertEqual(report["mismatch_sample"][0]["candidate_nct_id"], "NCT2")


if __name__ == "__main__":
    unittest.main()
