from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from clinical_trial_matching.evaluation.trec import TrecRunRow
from clinical_trial_matching.models import Topic, Trial
from clinical_trial_matching.retrieval.dense import trial_text

CROSS_ENCODER_ARTIFACT_SCHEMA_VERSION = 1
CROSS_ENCODER_RETRIEVER_NAME = "cross-encoder-reranker"
RERANK_SCORE_TIE_DECIMALS = 6


class PairScorer(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]], *, batch_size: int) -> Any: ...


@dataclass(frozen=True)
class RerankerFramework:
    modules: dict[str, Any]


@dataclass(frozen=True)
class RerankedTopic:
    rows: tuple[TrecRunRow, ...]
    inference_ms: float
    total_ms: float


class OnnxCrossEncoder:
    def __init__(
        self,
        artifact_path: Path,
        *,
        model_name: str,
        model_revision: str,
        model_file: str,
        max_length: int,
        framework: RerankerFramework,
    ) -> None:
        metadata_path = artifact_path / "metadata.json"
        model_path = artifact_path / model_file
        tokenizer_path = artifact_path / "tokenizer.json"
        missing_files = [
            path.name for path in (metadata_path, model_path, tokenizer_path) if not path.is_file()
        ]
        if missing_files:
            raise ValueError(
                "Cross-encoder artifact is missing files: " + ", ".join(missing_files)
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Cross-encoder metadata must be a JSON object")
        expected = {
            "schema_version": CROSS_ENCODER_ARTIFACT_SCHEMA_VERSION,
            "backend": "onnxruntime",
            "model_name": model_name,
            "model_revision": model_revision,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if metadata.get("model_file", "onnx/model.onnx") != model_file:
            mismatches.append("model_file")
        if mismatches:
            raise ValueError(
                "Cross-encoder artifact is incompatible with the experiment: "
                + ", ".join(mismatches)
            )
        _validate_artifact_checksum(metadata, model_file, model_path)
        _validate_artifact_checksum(metadata, "tokenizer.json", tokenizer_path)

        np = framework.modules["numpy"]
        onnxruntime = framework.modules["onnxruntime"]
        tokenizers = framework.modules["tokenizers"]
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", pad_type_id=0)
        self.session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = tokenizer
        self.input_names = tuple(item.name for item in self.session.get_inputs())
        self.output_name = self.session.get_outputs()[0].name
        self.np = np
        self.metadata = metadata
        self.backend = "onnxruntime"

    def predict(self, pairs: Sequence[tuple[str, str]], *, batch_size: int) -> Any:
        if batch_size < 1:
            raise ValueError("Cross-encoder batch size must be at least 1")
        outputs = []
        for offset in range(0, len(pairs), batch_size):
            encodings = self.tokenizer.encode_batch(list(pairs[offset : offset + batch_size]))
            values = {
                "input_ids": self.np.asarray(
                    [encoding.ids for encoding in encodings], dtype=self.np.int64
                ),
                "attention_mask": self.np.asarray(
                    [encoding.attention_mask for encoding in encodings], dtype=self.np.int64
                ),
                "token_type_ids": self.np.asarray(
                    [encoding.type_ids for encoding in encodings], dtype=self.np.int64
                ),
            }
            feeds = {name: values[name] for name in self.input_names}
            outputs.append(self.session.run([self.output_name], feeds)[0])
        if not outputs:
            return self.np.empty((0,), dtype=self.np.float32)
        return self.np.concatenate(outputs, axis=0).reshape(-1)


def download_cross_encoder_artifact(
    *,
    model_name: str,
    model_revision: str,
    model_file: str,
    output_path: Path,
) -> dict[str, Any]:
    try:
        huggingface_hub = import_module("huggingface_hub")
    except ImportError as exc:
        raise RuntimeError(
            'Install local reranking dependencies with `python -m pip install -e ".[onnx]"`.'
        ) from exc
    output_path.mkdir(parents=True, exist_ok=True)
    huggingface_hub.snapshot_download(
        repo_id=model_name,
        revision=model_revision,
        local_dir=output_path,
        allow_patterns=[
            "config.json",
            model_file,
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ],
    )
    artifact_files = {}
    for relative_path in (
        "config.json",
        model_file,
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        path = output_path / relative_path
        if path.is_file():
            artifact_files[relative_path] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    required = {"config.json", model_file, "tokenizer.json"}
    missing = required - set(artifact_files)
    if missing:
        raise ValueError(
            "Downloaded cross-encoder artifact is incomplete: " + ", ".join(sorted(missing))
        )
    metadata = {
        "schema_version": CROSS_ENCODER_ARTIFACT_SCHEMA_VERSION,
        "backend": "onnxruntime",
        "model_name": model_name,
        "model_revision": model_revision,
        "model_file": model_file,
        "files": artifact_files,
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_reranker_framework() -> RerankerFramework:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    try:
        return RerankerFramework(
            modules={
                "numpy": import_module("numpy"),
                "onnxruntime": import_module("onnxruntime"),
                "tokenizers": import_module("tokenizers"),
            }
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install local reranking dependencies with "
            '`python -m pip install -e ".[onnx-runtime]"`.'
        ) from exc


def rerank_topic(
    *,
    topic: Topic,
    baseline_nct_ids: Sequence[str],
    candidate_trials: Sequence[Trial],
    reranker: PairScorer,
    candidate_depth: int,
    top_k: int,
    text_representation: str,
    batch_size: int,
    run_name: str,
    clock: Any,
) -> RerankedTopic:
    if candidate_depth < 1 or top_k < 1:
        raise ValueError("Candidate depth and top-k must be positive")
    candidates = tuple(baseline_nct_ids[:candidate_depth])
    if tuple(trial.nct_id for trial in candidate_trials) != candidates:
        raise ValueError("Candidate trial order does not match the baseline ranking")
    total_start = clock()
    inference_start = clock()
    scores = reranker.predict(
        [(topic.text, trial_text(trial, text_representation)) for trial in candidate_trials],
        batch_size=batch_size,
    )
    inference_end = clock()
    if len(scores) != len(candidates):
        raise ValueError("Cross-encoder score count does not match the candidate count")
    rounded_scores = [round(float(score), RERANK_SCORE_TIE_DECIMALS) for score in scores]
    ranked_candidates = sorted(
        zip(candidates, rounded_scores, strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    reranked_ids = [nct_id for nct_id, _score in ranked_candidates]
    reranked_ids.extend(baseline_nct_ids[candidate_depth:top_k])
    rows = tuple(
        TrecRunRow(
            topic_id=topic.topic_id,
            nct_id=nct_id,
            rank=rank,
            score=float(top_k - rank + 1),
            run_name=run_name,
        )
        for rank, nct_id in enumerate(reranked_ids, start=1)
    )
    return RerankedTopic(
        rows=rows,
        inference_ms=(inference_end - inference_start) * 1000,
        total_ms=(clock() - total_start) * 1000,
    )


def _validate_artifact_checksum(
    metadata: dict[str, Any],
    relative_path: str,
    path: Path,
) -> None:
    files = metadata.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(relative_path), dict):
        raise ValueError(f"Cross-encoder metadata is missing {relative_path}")
    expected = files[relative_path].get("sha256")
    if expected != _sha256_file(path):
        raise ValueError(f"Cross-encoder checksum mismatch for {relative_path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
