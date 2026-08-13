from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from clinical_trial_matching.cli import run_rrf_experiment
from clinical_trial_matching.evaluation.experiments import load_rrf_experiment
from clinical_trial_matching.io import read_json, write_json
from clinical_trial_matching.retrieval.hybrid import (
    RankedRun,
    read_trec_rankings,
    reciprocal_rank_fusion,
)


class ReciprocalRankFusionTest(unittest.TestCase):
    def test_rrf_experiment_writes_traceable_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shutil.copyfile("data/fixtures/trials.sample.jsonl", root / "trials.jsonl")
            shutil.copyfile("data/fixtures/topics.sample.jsonl", root / "topics.jsonl")
            shutil.copyfile("data/fixtures/qrels.sample.tsv", root / "qrels.tsv")
            lexical_path = root / "lexical.run"
            dense_path = root / "dense.run"
            lexical_path.write_text(
                "1 Q0 NCT99990001 1 2 lexical\n"
                "1 Q0 NCT99990002 2 1 lexical\n"
                "2 Q0 NCT99990002 1 2 lexical\n"
                "2 Q0 NCT99990003 2 1 lexical\n",
                encoding="utf-8",
            )
            dense_path.write_text(
                "1 Q0 NCT99990002 1 2 dense\n"
                "1 Q0 NCT99990001 2 1 dense\n"
                "2 Q0 NCT99990003 1 2 dense\n"
                "2 Q0 NCT99990002 2 1 dense\n",
                encoding="utf-8",
            )
            config_path = root / "rrf.json"
            write_json(config_path, rrf_payload())

            experiment = load_rrf_experiment(config_path)
            run_rrf_experiment(config_path)
            metrics = read_json(root / "hybrid_metrics.json")
            diagnostics = read_json(root / "hybrid_diagnostics.json")

        self.assertEqual(experiment.rrf_k, 60)
        self.assertEqual(len(experiment.components), 2)
        self.assertEqual(metrics["retriever"], "reciprocal-rank-fusion")
        self.assertEqual(metrics["topics"], 2)
        self.assertEqual(metrics["run_rows"], 4)
        self.assertRegex(
            metrics["retriever_parameters"]["components"][0]["run_sha256"],
            r"^[a-f0-9]{64}$",
        )
        self.assertEqual(diagnostics["experiment"]["name"], "fixture_rrf")

    def test_rrf_experiment_rejects_component_topics_outside_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shutil.copyfile("data/fixtures/trials.sample.jsonl", root / "trials.jsonl")
            shutil.copyfile("data/fixtures/topics.sample.jsonl", root / "topics.jsonl")
            shutil.copyfile("data/fixtures/qrels.sample.tsv", root / "qrels.tsv")
            invalid_run = "99 Q0 NCT99990001 1 1 fixture\n"
            (root / "lexical.run").write_text(invalid_run, encoding="utf-8")
            (root / "dense.run").write_text(invalid_run, encoding="utf-8")
            config_path = root / "rrf.json"
            write_json(config_path, rrf_payload())

            with self.assertRaisesRegex(ValueError, "does not match benchmark topics"):
                run_rrf_experiment(config_path)

    def test_rrf_rewards_documents_retrieved_by_both_components(self) -> None:
        lexical = RankedRun(
            name="lexical",
            weight=1.0,
            rankings={"1": ("NCT_LEX", "NCT_BOTH", "NCT_X")},
        )
        dense = RankedRun(
            name="dense",
            weight=1.0,
            rankings={"1": ("NCT_DENSE", "NCT_BOTH", "NCT_Y")},
        )

        rows = reciprocal_rank_fusion(
            [lexical, dense],
            run_name="hybrid_fixture",
            rrf_k=60,
            top_k=3,
        )

        self.assertEqual(rows[0].nct_id, "NCT_BOTH")
        self.assertEqual(rows[0].rank, 1)
        self.assertEqual(rows[0].run_name, "hybrid_fixture")

    def test_rrf_uses_deterministic_best_rank_and_id_tie_breaks(self) -> None:
        run_a = RankedRun(name="a", weight=1.0, rankings={"2": ("NCT_B", "NCT_A")})
        run_b = RankedRun(name="b", weight=1.0, rankings={"2": ("NCT_A", "NCT_B")})

        rows = reciprocal_rank_fusion([run_a, run_b], run_name="hybrid", top_k=2)

        self.assertEqual([row.nct_id for row in rows], ["NCT_A", "NCT_B"])

    def test_rrf_rejects_mismatched_topic_sets(self) -> None:
        run_a = RankedRun(name="a", weight=1.0, rankings={"1": ("NCT1",)})
        run_b = RankedRun(name="b", weight=1.0, rankings={"2": ("NCT1",)})

        with self.assertRaisesRegex(ValueError, "identical topic sets"):
            reciprocal_rank_fusion([run_a, run_b], run_name="hybrid")

    def test_read_trec_rankings_orders_by_rank_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.txt"
            path.write_text(
                "1 Q0 NCT2 2 0.5 fixture\n1 Q0 NCT1 1 0.7 fixture\n",
                encoding="utf-8",
            )
            rankings = read_trec_rankings(path)
            duplicate_path = Path(tmpdir) / "duplicate.txt"
            duplicate_path.write_text(
                "1 Q0 NCT1 1 0.7 fixture\n1 Q0 NCT1 2 0.6 fixture\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate NCT ID"):
                read_trec_rankings(duplicate_path)

        self.assertEqual(rankings["1"], ("NCT1", "NCT2"))


def rrf_payload() -> dict:
    return {
        "schema_version": 1,
        "name": "fixture_rrf",
        "description": "Fixture hybrid experiment.",
        "project_root": ".",
        "rrf_k": 60,
        "candidate_depth": 100,
        "components": [
            {"name": "lexical", "run": "lexical.run", "weight": 1.0},
            {"name": "dense", "run": "dense.run", "weight": 1.0},
        ],
        "benchmark": {
            "trials": "trials.jsonl",
            "topics": "topics.jsonl",
            "qrels": "qrels.tsv",
            "top_k": 2,
        },
        "artifacts": {
            "run": "hybrid.run",
            "metrics": "hybrid_metrics.json",
            "diagnostics": "hybrid_diagnostics.json",
        },
    }


if __name__ == "__main__":
    unittest.main()
