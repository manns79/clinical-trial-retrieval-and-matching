from __future__ import annotations

import unittest

from clinical_trial_matching.models import Trial
from clinical_trial_matching.validation.trials import summarize_trial_corpus


class TrialCorpusReportTest(unittest.TestCase):
    def test_summarize_trial_corpus(self) -> None:
        trials = [
            Trial(
                nct_id="NCT1",
                title="Asthma trial",
                status="RECRUITING",
                conditions=("Asthma",),
                interventions=("Drug A",),
                eligibility_criteria="Adults with asthma.",
            ),
            Trial(
                nct_id="NCT2",
                title="Diabetes trial",
                status="COMPLETED",
                conditions=("Type 2 Diabetes",),
            ),
            Trial(
                nct_id="NCT2",
                title="Duplicate diabetes trial",
                status="COMPLETED",
                conditions=(),
                interventions=("Counseling",),
                eligibility_criteria="Adults with diabetes.",
            ),
        ]

        report = summarize_trial_corpus(trials, sample_size=2, top_n=2)

        self.assertEqual(report["trials"], 3)
        self.assertEqual(report["unique_nct_ids"], 2)
        self.assertEqual(report["duplicate_nct_ids"], [{"nct_id": "NCT2", "count": 2}])
        self.assertEqual(report["status_distribution"], {"RECRUITING": 1, "COMPLETED": 2})
        self.assertEqual(report["missing_eligibility_criteria"]["count"], 1)
        self.assertEqual(report["missing_eligibility_criteria"]["rate"], 0.333333)
        self.assertEqual(report["condition_coverage"]["with_values"], 2)
        self.assertEqual(report["condition_coverage"]["missing_values"], 1)
        self.assertEqual(report["intervention_coverage"]["with_values"], 2)
        self.assertEqual(len(report["sample_records"]), 2)

    def test_empty_corpus(self) -> None:
        report = summarize_trial_corpus([], sample_size=5, top_n=10)

        self.assertEqual(report["trials"], 0)
        self.assertEqual(report["unique_nct_ids"], 0)
        self.assertEqual(report["missing_eligibility_criteria"]["rate"], 0.0)
        self.assertEqual(report["condition_coverage"]["coverage_rate"], 0.0)
        self.assertEqual(report["sample_records"], [])


if __name__ == "__main__":
    unittest.main()
