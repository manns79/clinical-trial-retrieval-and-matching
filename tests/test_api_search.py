from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clinical_trial_matching.ingestion.clinicaltrials import (
    trial_from_flat_record,
    trial_to_flat_record,
)
from clinical_trial_matching.io import read_jsonl, write_jsonl
from clinical_trial_matching.models import SearchResult, Trial
from clinical_trial_matching.retrieval.sqlite_fts import build_sqlite_fts_index
from clinical_trial_matching.trial_store import build_trial_store

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_FASTAPI, "FastAPI is not installed")
class ApiSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        from clinical_trial_matching.api.main import (
            load_dense_encoder_framework,
            load_dense_search_encoder,
            load_dense_search_index,
            load_dense_search_retriever,
            load_search_retriever,
            load_sqlite_search_retriever,
            load_trial_corpus,
            load_trial_metadata_store,
            warm_dense_search_encoder,
        )

        load_trial_corpus.cache_clear()
        load_trial_metadata_store.cache_clear()
        load_search_retriever.cache_clear()
        load_sqlite_search_retriever.cache_clear()
        load_dense_search_index.cache_clear()
        load_dense_encoder_framework.cache_clear()
        load_dense_search_encoder.cache_clear()
        warm_dense_search_encoder.cache_clear()
        load_dense_search_retriever.cache_clear()
        self.environment_names = (
            "TRIAL_CORPUS_PATH",
            "TRIAL_STORE_PATH",
            "BM25_INDEX_PATH",
            "PLAIN_BM25_INDEX_PATH",
            "SQLITE_FTS_INDEX_PATH",
            "DENSE_INDEX_PATH",
            "DENSE_MODEL_NAME",
            "DENSE_TEXT_REPRESENTATION",
            "DENSE_BATCH_SIZE",
            "DENSE_DEVICE",
            "DENSE_MAX_SEQ_LENGTH",
            "DENSE_DYNAMIC_QUANTIZATION",
            "DENSE_ENCODER_BACKEND",
            "DENSE_ONNX_MODEL_PATH",
            "RRF_K",
            "RRF_CANDIDATE_DEPTH",
        )
        self.previous_environment = {
            name: os.environ.get(name) for name in self.environment_names
        }

    def tearDown(self) -> None:
        from clinical_trial_matching.api.main import (
            load_dense_encoder_framework,
            load_dense_search_encoder,
            load_dense_search_index,
            load_dense_search_retriever,
            load_search_retriever,
            load_sqlite_search_retriever,
            load_trial_corpus,
            load_trial_metadata_store,
            warm_dense_search_encoder,
        )

        for name, previous_value in self.previous_environment.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value
        load_trial_corpus.cache_clear()
        load_trial_metadata_store.cache_clear()
        load_search_retriever.cache_clear()
        load_sqlite_search_retriever.cache_clear()
        load_dense_search_index.cache_clear()
        load_dense_encoder_framework.cache_clear()
        load_dense_search_encoder.cache_clear()
        warm_dense_search_encoder.cache_clear()
        load_dense_search_retriever.cache_clear()

    def test_search_returns_traceable_sqlite_records_from_configured_corpus(self) -> None:
        from clinical_trial_matching.api.main import (
            SearchRequest,
            load_sqlite_search_retriever,
            load_trial_corpus,
            search,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "trials.jsonl"
            write_jsonl(
                corpus_path,
                [
                    trial_to_flat_record(
                        Trial(
                            nct_id="NCT1",
                            title="Asthma inhaler study",
                            status="RECRUITING",
                            conditions=("Asthma",),
                            interventions=("Inhaled corticosteroid",),
                            eligibility_criteria="Adults with persistent asthma and wheezing.",
                        )
                    ),
                    trial_to_flat_record(
                        Trial(
                            nct_id="NCT2",
                            title="Migraine prevention study",
                            status="COMPLETED",
                            conditions=("Migraine",),
                        )
                    ),
                ],
            )
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            self._configure_trial_store(corpus_path, build_fts=True)
            load_trial_corpus.cache_clear()
            load_sqlite_search_retriever.cache_clear()

            response = search(
                SearchRequest(
                    query="adult persistent asthma inhaled corticosteroid",
                    top_k=1,
                )
            )

        payload = response.model_dump()
        self.assertEqual(payload["retriever"], "sqlite-fts5")
        self.assertEqual(payload["corpus"], {"trials": 2, "unique_nct_ids": 2})
        self.assertEqual(payload["results"][0]["nct_id"], "NCT1")
        self.assertEqual(payload["results"][0]["rank"], 1)
        self.assertIn("asthma", payload["results"][0]["matched_terms"])
        self.assertIn("snippet", payload["results"][0])
        self.assertIn("latency_ms", payload)
        self.assertGreaterEqual(payload["latency_ms"]["corpus_load"], 0)
        self.assertGreaterEqual(payload["latency_ms"]["index_load"], 0)
        self.assertGreaterEqual(payload["latency_ms"]["lexical"], 0)
        self.assertEqual(payload["latency_ms"]["embedding"], 0)
        self.assertEqual(payload["latency_ms"]["fusion"], 0)
        self.assertGreaterEqual(payload["latency_ms"]["metadata"], 0)
        self.assertGreaterEqual(payload["latency_ms"]["retrieval"], 0)
        self.assertGreaterEqual(payload["latency_ms"]["total"], 0)

    def test_serving_field_weights_match_frozen_lexical_profile(self) -> None:
        from clinical_trial_matching.api.main import SERVING_FIELD_WEIGHTS
        from clinical_trial_matching.evaluation.experiments import load_bm25_experiment

        experiment = load_bm25_experiment(
            Path(
                "configs/experiments/trec_2021/"
                "fielded_bm25_condition_title_v1.json"
            )
        )

        self.assertEqual(SERVING_FIELD_WEIGHTS, experiment.field_weights)

    def test_dense_search_uses_cached_retriever_and_reports_embedding_latency(self) -> None:
        from clinical_trial_matching.api.main import SearchRequest, load_trial_corpus, search

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = self._write_search_corpus(Path(tmpdir))
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            self._configure_trial_store(corpus_path)
            load_trial_corpus.cache_clear()
            dense_retriever = FakeServingRetriever(
                [SearchResult("NCT1", 0.91, 1, "Asthma inhaler study")]
            )
            with patch(
                "clinical_trial_matching.api.main.load_dense_search_retriever",
                return_value=dense_retriever,
            ):
                response = search(
                    SearchRequest(query="adult asthma", top_k=1, retriever="dense")
                )

        payload = response.model_dump()
        self.assertEqual(payload["retriever"], "dense")
        self.assertEqual(payload["results"][0]["nct_id"], "NCT1")
        self.assertEqual(payload["parameters"]["model_name"], "fixture-model")
        self.assertEqual(payload["latency_ms"]["lexical"], 0)
        self.assertGreaterEqual(payload["latency_ms"]["embedding"], 0)
        self.assertEqual(payload["latency_ms"]["fusion"], 0)

    def test_hybrid_search_returns_component_ranks_and_stage_latencies(self) -> None:
        from clinical_trial_matching.api.main import SearchRequest, load_trial_corpus, search

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = self._write_search_corpus(Path(tmpdir))
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            self._configure_trial_store(corpus_path)
            load_trial_corpus.cache_clear()
            lexical_retriever = FakeServingRetriever(
                [
                    SearchResult("NCT1", 2.0, 1, "Asthma inhaler study"),
                    SearchResult("NCT2", 1.0, 2, "Migraine prevention study"),
                ]
            )
            dense_retriever = FakeServingRetriever(
                [
                    SearchResult("NCT2", 0.9, 1, "Migraine prevention study"),
                    SearchResult("NCT1", 0.8, 2, "Asthma inhaler study"),
                ]
            )
            with (
                patch(
                    "clinical_trial_matching.api.main.load_sqlite_search_retriever",
                    return_value=lexical_retriever,
                ),
                patch(
                    "clinical_trial_matching.api.main.load_dense_search_retriever",
                    return_value=dense_retriever,
                ),
            ):
                response = search(
                    SearchRequest(query="adult asthma", top_k=2, retriever="hybrid")
                )

        payload = response.model_dump()
        self.assertEqual(payload["retriever"], "hybrid")
        self.assertEqual(payload["parameters"]["rrf_k"], 60)
        self.assertEqual(
            payload["results"][0]["component_ranks"],
            {"sqlite-fts5": 1, "dense": 2},
        )
        self.assertGreaterEqual(payload["latency_ms"]["lexical"], 0)
        self.assertGreaterEqual(payload["latency_ms"]["embedding"], 0)
        self.assertGreaterEqual(payload["latency_ms"]["fusion"], 0)

    def test_dense_loader_initializes_model_and_index_once(self) -> None:
        from clinical_trial_matching.api.main import (
            load_dense_encoder_framework,
            load_dense_search_encoder,
            load_dense_search_index,
            load_dense_search_retriever,
            load_trial_corpus,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corpus_path = self._write_search_corpus(root)
            index_path = root / "dense.npz"
            index_path.touch()
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            self._configure_trial_store(corpus_path)
            os.environ["DENSE_INDEX_PATH"] = str(index_path)
            load_trial_corpus.cache_clear()
            load_dense_search_index.cache_clear()
            load_dense_encoder_framework.cache_clear()
            load_dense_search_encoder.cache_clear()
            load_dense_search_retriever.cache_clear()
            index = SimpleNamespace(nct_ids=("NCT1", "NCT2"), metadata={})
            encoder = object()
            framework = object()
            with (
                patch(
                    "clinical_trial_matching.api.main.load_dense_index_for_corpus",
                    return_value=index,
                ) as index_loader,
                patch(
                    "clinical_trial_matching.api.main.load_encoder_framework",
                    return_value=framework,
                ) as framework_loader,
                patch(
                    "clinical_trial_matching.api.main.construct_text_encoder",
                    return_value=encoder,
                ) as encoder_loader,
            ):
                first = load_dense_search_retriever()
                second = load_dense_search_retriever()

        self.assertIs(first, second)
        self.assertIs(first.index, index)
        self.assertIs(first.encoder, encoder)
        index_loader.assert_called_once()
        framework_loader.assert_called_once_with("sentence-transformers")
        encoder_loader.assert_called_once()

    def test_dense_search_returns_503_without_configured_index(self) -> None:
        from fastapi import HTTPException

        from clinical_trial_matching.api.main import load_dense_search_retriever

        os.environ.pop("DENSE_INDEX_PATH", None)
        load_dense_search_retriever.cache_clear()

        with self.assertRaises(HTTPException) as context:
            load_dense_search_retriever()

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("DENSE_INDEX_PATH", context.exception.detail)

    def test_lifespan_preloads_search_resources_once(self) -> None:
        import asyncio

        from clinical_trial_matching.api.main import app, lifespan

        async def exercise_lifespan() -> None:
            async with lifespan(app):
                pass

        with patch("clinical_trial_matching.api.main.preload_search_resources") as preload:
            asyncio.run(exercise_lifespan())

        preload.assert_called_once_with()

    def test_preload_reports_completed_resource_phases_in_order(self) -> None:
        from clinical_trial_matching.api.main import preload_search_resources

        completed: list[str] = []
        with (
            patch(
                "clinical_trial_matching.api.main.load_trial_metadata_store",
                return_value=SimpleNamespace(count=1),
            ),
            patch("clinical_trial_matching.api.main.load_sqlite_search_retriever"),
            patch(
                "clinical_trial_matching.api.main.get_dense_index_path",
                return_value=Path("dense.npz"),
            ),
            patch("clinical_trial_matching.api.main.load_dense_search_index"),
            patch("clinical_trial_matching.api.main.load_dense_encoder_framework"),
            patch("clinical_trial_matching.api.main.load_dense_search_encoder"),
            patch("clinical_trial_matching.api.main.warm_dense_search_encoder"),
            patch("clinical_trial_matching.api.main.load_dense_search_retriever"),
        ):
            preload_search_resources(on_phase_complete=completed.append)

        self.assertEqual(
            completed,
            [
                "trial_metadata_store",
                "sqlite_fts5",
                "dense_embedding_index",
                "dense_encoder_framework",
                "dense_encoder_model",
                "dense_encoder_first_inference_thread_pool",
                "dense_retriever_assembly",
            ],
        )

    def test_health_only_advertises_dense_modes_when_index_is_configured(self) -> None:
        from clinical_trial_matching.api.main import metrics_health

        without_dense = metrics_health()
        self.assertEqual(
            without_dense["available_retrievers"],
            ["sqlite-fts5", "fielded-bm25", "bm25"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corpus_path = self._write_search_corpus(root)
            index_path = root / "dense.npz"
            index_path.touch()
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            self._configure_trial_store(corpus_path)
            os.environ["DENSE_INDEX_PATH"] = str(index_path)
            with_dense = metrics_health()

        self.assertEqual(
            with_dense["available_retrievers"],
            ["sqlite-fts5", "fielded-bm25", "bm25", "dense", "hybrid"],
        )

    def test_health_does_not_advertise_onnx_without_local_artifact(self) -> None:
        from clinical_trial_matching.api.main import metrics_health

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            corpus_path = self._write_search_corpus(root)
            index_path = root / "dense.npz"
            index_path.touch()
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            self._configure_trial_store(corpus_path)
            os.environ["DENSE_INDEX_PATH"] = str(index_path)
            os.environ["DENSE_ENCODER_BACKEND"] = "onnxruntime"
            os.environ["DENSE_ONNX_MODEL_PATH"] = str(root / "missing-onnx")

            health = metrics_health()

        self.assertNotIn("dense", health["available_retrievers"])
        self.assertFalse(health["checks"]["dense_encoder_artifact_exists"])

    def test_timing_middleware_adds_process_time_header(self) -> None:
        import asyncio

        from fastapi import Response

        from clinical_trial_matching.api.main import timing_middleware

        class FakeUrl:
            path = "/health"

        class FakeRequest:
            method = "GET"
            url = FakeUrl()

        async def call_next(_: FakeRequest) -> Response:
            return Response(status_code=200)

        async def call_middleware() -> str:
            response = await timing_middleware(FakeRequest(), call_next)
            return response.headers["X-Process-Time-Ms"]

        process_time_ms = float(asyncio.run(call_middleware()))

        self.assertGreaterEqual(process_time_ms, 0)

    def test_get_trial_returns_normalized_trial_details(self) -> None:
        from clinical_trial_matching.api.main import get_trial

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "trials.jsonl"
            write_jsonl(
                corpus_path,
                [
                    trial_to_flat_record(
                        Trial(
                            nct_id="NCT1",
                            title="Asthma inhaler study",
                            brief_summary="Brief asthma trial summary.",
                            status="RECRUITING",
                            conditions=("Asthma",),
                            interventions=("Inhaled corticosteroid",),
                            eligibility_criteria="Adults with persistent asthma and wheezing.",
                            sex="ALL",
                            minimum_age="18 Years",
                            maximum_age="65 Years",
                            phases=("PHASE2",),
                            study_type="INTERVENTIONAL",
                            locations=("Boston, Massachusetts, United States",),
                            source={"kind": "test"},
                        )
                    )
                ],
            )
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            self._configure_trial_store(corpus_path)

            response = get_trial("nct1")

        payload = response.model_dump()
        self.assertEqual(payload["nct_id"], "NCT1")
        self.assertEqual(payload["title"], "Asthma inhaler study")
        self.assertEqual(payload["brief_summary"], "Brief asthma trial summary.")
        self.assertEqual(payload["conditions"], ["Asthma"])
        self.assertEqual(payload["interventions"], ["Inhaled corticosteroid"])
        self.assertEqual(payload["minimum_age"], "18 Years")
        self.assertEqual(payload["phases"], ["PHASE2"])
        self.assertEqual(payload["source"], {"kind": "test"})

    def test_get_trial_returns_404_when_nct_id_is_missing(self) -> None:
        from fastapi import HTTPException

        from clinical_trial_matching.api.main import get_trial

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = Path(tmpdir) / "trials.jsonl"
            write_jsonl(
                corpus_path,
                [
                    trial_to_flat_record(
                        Trial(
                            nct_id="NCT1",
                            title="Asthma inhaler study",
                        )
                    )
                ],
            )
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            self._configure_trial_store(corpus_path)

            with self.assertRaises(HTTPException) as context:
                get_trial("NCT404")

        self.assertEqual(context.exception.status_code, 404)
        self.assertIn("Trial not found", context.exception.detail)

    def test_primary_search_and_trial_lookup_do_not_load_jsonl_corpus(self) -> None:
        from clinical_trial_matching.api.main import (
            SearchRequest,
            get_trial,
            load_trial_corpus,
            search,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = self._write_search_corpus(Path(tmpdir))
            os.environ["TRIAL_CORPUS_PATH"] = str(corpus_path)
            self._configure_trial_store(corpus_path)
            load_trial_corpus.cache_clear()
            dense_retriever = FakeServingRetriever(
                [SearchResult("NCT1", 0.91, 1, "")]
            )
            with patch(
                "clinical_trial_matching.api.main.load_dense_search_retriever",
                return_value=dense_retriever,
            ):
                response = search(
                    SearchRequest(query="adult asthma", top_k=1, retriever="dense")
                )
            trial = get_trial("NCT1")

        self.assertEqual(response.results[0]["nct_id"], "NCT1")
        self.assertEqual(trial.nct_id, "NCT1")
        self.assertEqual(load_trial_corpus.cache_info().currsize, 0)

    def test_search_returns_503_when_configured_corpus_is_missing(self) -> None:
        from fastapi import HTTPException

        from clinical_trial_matching.api.main import SearchRequest, load_trial_corpus, search

        os.environ["TRIAL_CORPUS_PATH"] = "/tmp/clinical-trial-missing-corpus.jsonl"
        load_trial_corpus.cache_clear()

        with self.assertRaises(HTTPException) as context:
            search(SearchRequest(query="asthma"))

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("Trial corpus not found", context.exception.detail)

    @staticmethod
    def _configure_trial_store(corpus_path: Path, *, build_fts: bool = False) -> None:
        from clinical_trial_matching.api.main import load_trial_metadata_store

        trials = [trial_from_flat_record(row) for row in read_jsonl(corpus_path)]
        store_path = corpus_path.with_name("trial_store.sqlite")
        build_trial_store(store_path, trials, corpus_path=corpus_path)
        os.environ["TRIAL_STORE_PATH"] = str(store_path)
        load_trial_metadata_store.cache_clear()
        if build_fts:
            fts_path = corpus_path.with_name("trials_fts.sqlite")
            build_sqlite_fts_index(fts_path, trials, corpus_path=corpus_path)
            os.environ["SQLITE_FTS_INDEX_PATH"] = str(fts_path)

    @staticmethod
    def _write_search_corpus(root: Path) -> Path:
        corpus_path = root / "trials.jsonl"
        write_jsonl(
            corpus_path,
            [
                trial_to_flat_record(
                    Trial(
                        nct_id="NCT1",
                        title="Asthma inhaler study",
                        conditions=("Asthma",),
                    )
                ),
                trial_to_flat_record(
                    Trial(
                        nct_id="NCT2",
                        title="Migraine prevention study",
                        conditions=("Migraine",),
                    )
                ),
            ],
        )
        return corpus_path


class FakeServingRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.index = SimpleNamespace(
            metadata={
                "model_name": "fixture-model",
                "text_representation": "title_summary_conditions",
                "max_seq_length": 128,
                "embedding_dimension": 3,
                "normalize_embeddings": True,
            }
        )

    def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        return self.results[:top_k]


if __name__ == "__main__":
    unittest.main()
