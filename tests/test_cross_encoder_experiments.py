from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.evaluation.experiments import load_cross_encoder_experiment
from clinical_trial_matching.io import read_json, write_json


class CrossEncoderExperimentTest(unittest.TestCase):
    def test_tracked_experiment_is_development_only_and_pins_depths(self) -> None:
        experiment = load_cross_encoder_experiment(
            Path(
                "configs/experiments/trec_2021/"
                "development_cross_encoder_minilm_l6_v2.json"
            )
        )

        self.assertEqual(experiment.candidate_depths, (10, 25, 50))
        self.assertEqual(experiment.model_revision, "233902d")
        self.assertEqual(experiment.text_representation, "clinical_core")
        self.assertIn("/development/", experiment.topics_path.as_posix())
        self.assertNotIn("/holdout/", experiment.topics_path.as_posix())

    def test_loader_rejects_duplicate_or_unsorted_depths(self) -> None:
        source = Path(
            "configs/experiments/trec_2021/development_cross_encoder_minilm_l6_v2.json"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = read_json(source)
            payload["project_root"] = "."
            payload["reranking"]["candidate_depths"] = [25, 10, 25]
            config_path = root / "experiment.json"
            write_json(config_path, payload)

            with self.assertRaisesRegex(ValueError, "unique and ascending"):
                load_cross_encoder_experiment(config_path)


if __name__ == "__main__":
    unittest.main()
