from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from clinical_trial_matching.cli import evaluate_trec_dense
from clinical_trial_matching.evaluation.trec import build_dense_trec_run
from clinical_trial_matching.ingestion.clinicaltrials import trial_from_flat_record
from clinical_trial_matching.io import read_json, read_jsonl
from clinical_trial_matching.models import Topic, Trial
from clinical_trial_matching.retrieval.dense import (
    DenseRetriever,
    build_dense_index,
    load_dense_index,
    load_or_build_dense_retriever,
    save_dense_index,
    trial_text,
)

try:
    import numpy as np
except ImportError:
    np = None


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int, bool]] = []

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Any:
        if np is None:
            raise RuntimeError("NumPy is required for this fixture")
        self.calls.append((list(texts), batch_size, show_progress_bar))
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    2.0 if "asthma" in lowered else 0.1,
                    2.0 if "diabetes" in lowered else 0.1,
                    1.0 if "adult" in lowered else 0.1,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


@unittest.skipIf(np is None, "NumPy dense extra is not installed")
class DenseRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trials = [
            Trial(
                nct_id="NCT1",
                title="Adult asthma inhaler study",
                brief_summary="Treatment for persistent wheezing.",
                conditions=("Asthma",),
            ),
            Trial(
                nct_id="NCT2",
                title="Type 2 diabetes lifestyle study",
                conditions=("Diabetes Mellitus, Type 2",),
            ),
        ]

    def test_trial_text_uses_named_representation(self) -> None:
        text = trial_text(self.trials[0], "title_summary_conditions")

        self.assertIn("Title: Adult asthma inhaler study", text)
        self.assertIn("Brief Summary: Treatment for persistent wheezing.", text)
        self.assertIn("Conditions: Asthma", text)
        self.assertNotIn("Eligibility Criteria", text)

    def test_build_and_search_normalizes_embeddings_and_batches_queries(self) -> None:
        encoder = FakeEncoder()
        index = build_dense_index(
            self.trials,
            encoder=encoder,
            model_name="fixture-model",
            text_representation="title_summary_conditions",
            batch_size=2,
            device="cpu",
            max_seq_length=128,
            show_progress_bar=False,
        )
        retriever = DenseRetriever(
            self.trials,
            index=index,
            encoder=encoder,
            batch_size=2,
        )

        rankings = retriever.search_many(["adult asthma", "diabetes"], top_k=2)

        self.assertEqual(rankings[0][0].nct_id, "NCT1")
        self.assertEqual(rankings[1][0].nct_id, "NCT2")
        self.assertTrue(np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0))
        self.assertEqual(encoder.calls[0][1], 2)
        self.assertEqual(len(encoder.calls[1][0]), 2)

    def test_persisted_numpy_index_round_trips_without_pickle(self) -> None:
        index = self._build_index()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dense.npz"
            save_dense_index(path, index)
            loaded = load_dense_index(
                path,
                self.trials,
                model_name="fixture-model",
                text_representation="title_summary_conditions",
                max_seq_length=128,
            )

        self.assertEqual(loaded.nct_ids, ("NCT1", "NCT2"))
        self.assertTrue(np.allclose(loaded.embeddings, index.embeddings))
        self.assertEqual(loaded.metadata["embedding_dimension"], 3)

    def test_persisted_index_rejects_model_or_corpus_drift(self) -> None:
        index = self._build_index()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dense.npz"
            save_dense_index(path, index)
            with self.assertRaisesRegex(ValueError, "model_name"):
                load_dense_index(
                    path,
                    self.trials,
                    model_name="different-model",
                    text_representation="title_summary_conditions",
                    max_seq_length=128,
                )
            with self.assertRaisesRegex(ValueError, "corpus_fingerprint"):
                load_dense_index(
                    path,
                    [Trial(nct_id="NCT1", title="Changed"), self.trials[1]],
                    model_name="fixture-model",
                    text_representation="title_summary_conditions",
                    max_seq_length=128,
                )

    def test_build_rejects_duplicate_nct_ids(self) -> None:
        duplicate_trials = [self.trials[0], self.trials[0]]

        with self.assertRaisesRegex(ValueError, "duplicate NCT IDs"):
            build_dense_index(
                duplicate_trials,
                encoder=FakeEncoder(),
                model_name="fixture-model",
                text_representation="title_summary_conditions",
                batch_size=2,
                device="cpu",
                max_seq_length=128,
                show_progress_bar=False,
            )

    def test_load_or_build_reuses_index_and_trec_run_format(self) -> None:
        index = self._build_index()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dense.npz"
            save_dense_index(path, index)
            encoder = FakeEncoder()
            retriever = load_or_build_dense_retriever(
                trials=self.trials,
                model_name="fixture-model",
                text_representation="title_summary_conditions",
                batch_size=2,
                device="cpu",
                max_seq_length=128,
                index_path=path,
                encoder_factory=lambda _model, _device, _length: encoder,
                show_progress_bar=False,
            )
            rows = build_dense_trec_run(
                retriever=retriever,
                topics=[Topic(topic_id="1", text="adult asthma")],
                run_name="dense_fixture",
                top_k=2,
            )

        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(rows[0].nct_id, "NCT1")
        self.assertRegex(rows[0].to_trec_line(), r"^1 Q0 NCT1 1 [0-9.]+ dense_fixture$")

    def test_dense_evaluation_writes_traceable_metrics_and_diagnostics(self) -> None:
        trials = [
            trial_from_flat_record(row)
            for row in read_jsonl(Path("data/fixtures/trials.sample.jsonl"))
        ]
        encoder = FakeEncoder()
        index = build_dense_index(
            trials,
            encoder=encoder,
            model_name="fixture-model",
            text_representation="title_summary_conditions",
            batch_size=2,
            device="cpu",
            max_seq_length=128,
            show_progress_bar=False,
        )
        retriever = DenseRetriever(trials, index=index, encoder=encoder, batch_size=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_path = root / "metrics.json"
            diagnostics_path = root / "diagnostics.json"
            with patch(
                "clinical_trial_matching.cli.load_or_build_dense_retriever",
                return_value=retriever,
            ):
                evaluate_trec_dense(
                    trials_path=Path("data/fixtures/trials.sample.jsonl"),
                    topics_path=Path("data/fixtures/topics.sample.jsonl"),
                    qrels_path=Path("data/fixtures/qrels.sample.tsv"),
                    index_path=root / "dense.npz",
                    run_output_path=root / "dense.run",
                    metrics_output_path=metrics_path,
                    diagnostics_output_path=diagnostics_path,
                    run_name="dense_fixture",
                    top_k=3,
                    model_name="fixture-model",
                    text_representation="title_summary_conditions",
                    batch_size=2,
                    device="cpu",
                    max_seq_length=128,
                    experiment_metadata={"name": "dense_fixture"},
                )
            metrics = read_json(metrics_path)
            diagnostics = read_json(diagnostics_path)

        self.assertEqual(metrics["retriever"], "dense-bi-encoder")
        self.assertEqual(metrics["topics"], 2)
        self.assertEqual(metrics["run_rows"], 6)
        self.assertIn("eligible_only", metrics["metrics"])
        self.assertEqual(metrics["experiment"]["name"], "dense_fixture")
        self.assertEqual(diagnostics["retriever_parameters"]["embedding_dimension"], 3)

    def _build_index(self) -> Any:
        return build_dense_index(
            self.trials,
            encoder=FakeEncoder(),
            model_name="fixture-model",
            text_representation="title_summary_conditions",
            batch_size=2,
            device="cpu",
            max_seq_length=128,
            show_progress_bar=False,
        )


if __name__ == "__main__":
    unittest.main()
