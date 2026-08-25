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
        self.assertEqual(experiment.model_file, "onnx/model.onnx")
        self.assertEqual(experiment.model_precision, "fp32")
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

    def test_int8_profile_pins_official_model_file_and_development_split(self) -> None:
        experiment = load_cross_encoder_experiment(
            Path(
                "configs/experiments/trec_2021/"
                "development_cross_encoder_depth10_int8_128_short.json"
            )
        )

        self.assertEqual(experiment.candidate_depths, (10,))
        self.assertEqual(experiment.model_file, "onnx/model_quint8_avx2.onnx")
        self.assertEqual(experiment.model_precision, "int8")
        self.assertEqual(experiment.max_length, 128)
        self.assertEqual(experiment.text_representation, "title_summary_conditions")
        self.assertIn("/development/", experiment.topics_path.as_posix())

    def test_small_depth_profile_is_int8_256_clinical_core(self) -> None:
        experiment = load_cross_encoder_experiment(
            Path(
                "configs/experiments/trec_2021/"
                "development_cross_encoder_int8_256_core_small_depths.json"
            )
        )

        self.assertEqual(experiment.candidate_depths, (3, 5, 8))
        self.assertEqual(experiment.model_precision, "int8")
        self.assertEqual(experiment.max_length, 256)
        self.assertEqual(experiment.text_representation, "clinical_core")
        self.assertIn("/development/", experiment.qrels_path.as_posix())

    def test_loader_rejects_unsafe_model_file(self) -> None:
        source = Path(
            "configs/experiments/trec_2021/development_cross_encoder_minilm_l6_v2.json"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = read_json(source)
            payload["project_root"] = "."
            payload["model"]["file"] = "../model.onnx"
            config_path = Path(tmpdir) / "experiment.json"
            write_json(config_path, payload)

            with self.assertRaisesRegex(ValueError, "safe relative path"):
                load_cross_encoder_experiment(config_path)


if __name__ == "__main__":
    unittest.main()
