from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from clinical_trial_matching.models import SearchResult, Trial
from clinical_trial_matching.retrieval.bm25 import corpus_fingerprint

DENSE_INDEX_SCHEMA_VERSION = "1.0"
DENSE_RETRIEVER_NAME = "dense-bi-encoder"
DENSE_INDEX_MMAP_SUFFIX = ".mmap"
DENSE_SCORE_TIE_DECIMALS = 6
ONNX_ENCODER_SCHEMA_VERSION = "1.0"
ENCODER_BACKENDS = {"sentence-transformers", "onnxruntime"}
TEXT_REPRESENTATIONS = {
    "title": ("title",),
    "title_summary_conditions": ("title", "brief_summary", "conditions"),
    "eligibility_snapshot": (
        "title",
        "conditions",
        "demographics",
        "eligibility_criteria",
    ),
    "clinical_core": (
        "title",
        "conditions",
        "demographics",
        "eligibility_criteria",
        "interventions",
        "brief_summary",
    ),
    "all_fields": (
        "title",
        "brief_summary",
        "conditions",
        "interventions",
        "eligibility_criteria",
        "demographics",
        "status",
        "locations",
    ),
}


class TextEncoder(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class DenseIndex:
    embeddings: Any
    nct_ids: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EncoderFramework:
    backend: str
    modules: dict[str, Any]


class SentenceTransformerEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        max_seq_length: int | None,
        sentence_transformers_module: Any | None = None,
    ) -> None:
        sentence_transformers = sentence_transformers_module or _sentence_transformers()
        self.model = sentence_transformers.SentenceTransformer(model_name, device=device)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.quantization = "fp32"
        self.backend = "sentence-transformers"

    def quantize_dynamic_int8(self) -> None:
        torch = import_module("torch")
        quantization = getattr(getattr(torch, "ao", None), "quantization", None)
        quantize_dynamic = getattr(quantization, "quantize_dynamic", None)
        if quantize_dynamic is None:
            quantize_dynamic = torch.quantization.quantize_dynamic
        self.model = quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8,
            inplace=True,
        )
        self.quantization = "dynamic_int8"

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Any:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class OnnxRuntimeEncoder:
    def __init__(
        self,
        artifact_path: Path,
        *,
        model_name: str,
        max_seq_length: int | None,
        framework: EncoderFramework,
    ) -> None:
        if framework.backend != "onnxruntime":
            raise ValueError("ONNX encoder requires the onnxruntime framework")
        metadata_path = artifact_path / "metadata.json"
        model_path = artifact_path / "model.onnx"
        tokenizer_path = artifact_path / "tokenizer.json"
        missing = [
            path.name for path in (metadata_path, model_path, tokenizer_path) if not path.is_file()
        ]
        if missing:
            raise ValueError(f"ONNX encoder artifact is missing files: {', '.join(missing)}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("ONNX encoder metadata must be a JSON object")
        expected = {
            "schema_version": ONNX_ENCODER_SCHEMA_VERSION,
            "backend": "onnxruntime",
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "normalize_embeddings": True,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError(
                "ONNX encoder artifact is incompatible with the serving config: "
                + ", ".join(mismatches)
            )
        if _sha256_file(model_path) != metadata.get("model_sha256"):
            raise ValueError("ONNX encoder model checksum does not match its metadata")
        if _sha256_file(tokenizer_path) != metadata.get("tokenizer_sha256"):
            raise ValueError("ONNX encoder tokenizer checksum does not match its metadata")

        tokenizers = framework.modules["tokenizers"]
        onnxruntime = framework.modules["onnxruntime"]
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=int(metadata["max_seq_length"]))
        self.tokenizer.enable_padding(
            pad_id=int(metadata["pad_token_id"]),
            pad_token=str(metadata["pad_token"]),
            pad_type_id=int(metadata.get("pad_token_type_id", 0)),
        )
        self.session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = tuple(input_value.name for input_value in self.session.get_inputs())
        self.output_name = self.session.get_outputs()[0].name
        self.np = framework.modules["numpy"]
        self.metadata = metadata
        self.quantization = str(metadata.get("quantization", "fp32"))
        self.backend = "onnxruntime"

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Any:
        del show_progress_bar
        if batch_size < 1:
            raise ValueError("Dense batch size must be at least 1")
        batches = []
        for offset in range(0, len(texts), batch_size):
            encodings = self.tokenizer.encode_batch(list(texts[offset : offset + batch_size]))
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
            batches.append(self.session.run([self.output_name], feeds)[0])
        if not batches:
            dimension = int(self.metadata["embedding_dimension"])
            return self.np.empty((0, dimension), dtype=self.np.float32)
        return self.np.concatenate(batches, axis=0)


def load_encoder_framework(backend: str) -> EncoderFramework:
    _validate_encoder_backend(backend)
    if backend == "sentence-transformers":
        return EncoderFramework(
            backend=backend,
            modules={"sentence_transformers": _sentence_transformers()},
        )
    try:
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        return EncoderFramework(
            backend=backend,
            modules={
                "numpy": import_module("numpy"),
                "onnxruntime": import_module("onnxruntime"),
                "tokenizers": import_module("tokenizers"),
            },
        )
    except ImportError as exc:
        raise RuntimeError(
            'Install the local ONNX runtime with `python -m pip install -e ".[onnx]"`.'
        ) from exc


def construct_text_encoder(
    *,
    backend: str,
    framework: EncoderFramework,
    model_name: str,
    device: str,
    max_seq_length: int | None,
    onnx_model_path: Path | None = None,
) -> TextEncoder:
    _validate_encoder_backend(backend)
    if framework.backend != backend:
        raise ValueError("Dense encoder framework and configured backend do not match")
    if backend == "sentence-transformers":
        return SentenceTransformerEncoder(
            model_name,
            device=device,
            max_seq_length=max_seq_length,
            sentence_transformers_module=framework.modules["sentence_transformers"],
        )
    if device != "cpu":
        raise ValueError("The ONNX Runtime prototype currently supports only the CPU device")
    if onnx_model_path is None:
        raise ValueError("The ONNX Runtime backend requires an ONNX model artifact path")
    return OnnxRuntimeEncoder(
        onnx_model_path,
        model_name=model_name,
        max_seq_length=max_seq_length,
        framework=framework,
    )


def warm_up_text_encoder(encoder: TextEncoder, *, batch_size: int) -> None:
    encoder.encode(
        ["Adult with a documented condition seeking a clinical trial."],
        batch_size=batch_size,
        show_progress_bar=False,
    )


class DenseRetriever:
    def __init__(
        self,
        trials: Iterable[Trial] | None,
        *,
        index: DenseIndex,
        encoder: TextEncoder,
        batch_size: int,
    ) -> None:
        self.trials = (
            None
            if trials is None
            else trials if isinstance(trials, tuple) else tuple(trials)
        )
        self.index = index
        self.encoder = encoder
        self.batch_size = batch_size
        if self.trials is not None and len(self.trials) != len(self.index.nct_ids):
            raise ValueError("Dense index and trial corpus have different lengths")

    def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        return self.search_many([query], top_k=top_k)[0]

    def search_many(self, queries: Sequence[str], *, top_k: int = 10) -> list[list[SearchResult]]:
        if top_k < 1:
            raise ValueError("Top-K must be at least 1")
        if not queries:
            return []

        np = _numpy()
        query_embeddings = _normalized_embeddings(
            self.encoder.encode(
                queries,
                batch_size=self.batch_size,
                show_progress_bar=False,
            ),
            expected_rows=len(queries),
        )
        if query_embeddings.shape[1] != self.index.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match persisted dense index dimension"
            )

        scores = query_embeddings @ self.index.embeddings.T
        result_count = min(top_k, len(self.index.nct_ids))
        rankings: list[list[SearchResult]] = []
        nct_id_values = np.asarray(self.index.nct_ids)
        for query_scores in scores:
            rounded_scores = np.round(query_scores, decimals=DENSE_SCORE_TIE_DECIMALS)
            ranked_indexes = np.lexsort((nct_id_values, -rounded_scores))[:result_count]
            rankings.append(
                [
                    SearchResult(
                        nct_id=self.index.nct_ids[int(index)],
                        score=float(query_scores[int(index)]),
                        rank=rank,
                        title=(
                            ""
                            if self.trials is None
                            else self.trials[int(index)].title
                        ),
                    )
                    for rank, index in enumerate(ranked_indexes, start=1)
                ]
            )
        return rankings


def build_dense_index(
    trials: Iterable[Trial],
    *,
    encoder: TextEncoder,
    model_name: str,
    text_representation: str,
    batch_size: int,
    device: str,
    max_seq_length: int | None,
    show_progress_bar: bool = True,
) -> DenseIndex:
    trial_list = list(trials)
    if not trial_list:
        raise ValueError("Cannot build a dense index for an empty trial corpus")
    if len({trial.nct_id for trial in trial_list}) != len(trial_list):
        raise ValueError("Cannot build a dense index for a corpus with duplicate NCT IDs")
    _validate_dense_parameters(
        model_name=model_name,
        text_representation=text_representation,
        batch_size=batch_size,
        device=device,
        max_seq_length=max_seq_length,
    )
    embeddings = _normalized_embeddings(
        encoder.encode(
            [trial_text(trial, text_representation) for trial in trial_list],
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        ),
        expected_rows=len(trial_list),
    )
    return DenseIndex(
        embeddings=embeddings,
        nct_ids=tuple(trial.nct_id for trial in trial_list),
        metadata={
            "schema_version": DENSE_INDEX_SCHEMA_VERSION,
            "retriever": DENSE_RETRIEVER_NAME,
            "model_name": model_name,
            "text_representation": text_representation,
            "batch_size": batch_size,
            "device": device,
            "max_seq_length": max_seq_length,
            "normalize_embeddings": True,
            "trials": len(trial_list),
            "unique_nct_ids": len({trial.nct_id for trial in trial_list}),
            "embedding_dimension": int(embeddings.shape[1]),
            "corpus_fingerprint": corpus_fingerprint(trial_list),
        },
    )


def save_dense_index(path: Path, index: DenseIndex) -> None:
    if path.suffix.lower() == DENSE_INDEX_MMAP_SUFFIX:
        _save_mmap_dense_index(path, index)
        return
    if path.suffix.lower() != ".npz":
        raise ValueError("Dense index output must use the .npz or .mmap suffix")
    np = _numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                embeddings=index.embeddings,
                nct_ids=np.asarray(index.nct_ids),
                metadata=np.asarray(json.dumps(index.metadata, sort_keys=True)),
            )
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_dense_index(
    path: Path,
    trials: Iterable[Trial],
    *,
    model_name: str,
    text_representation: str,
    max_seq_length: int | None,
) -> DenseIndex:
    trial_list = list(trials)
    index = _read_dense_index(path)
    _validate_loaded_index(
        embeddings=index.embeddings,
        nct_ids=index.nct_ids,
        metadata=index.metadata,
        expected_nct_ids=tuple(trial.nct_id for trial in trial_list),
        expected_trials=len(trial_list),
        expected_corpus_fingerprint=corpus_fingerprint(trial_list),
        model_name=model_name,
        text_representation=text_representation,
        max_seq_length=max_seq_length,
        validate_embedding_values=path.suffix.lower() != DENSE_INDEX_MMAP_SUFFIX,
    )
    return index


def load_dense_index_for_corpus(
    path: Path,
    *,
    trials_count: int,
    corpus_fingerprint_value: str,
    model_name: str,
    text_representation: str,
    max_seq_length: int | None,
) -> DenseIndex:
    index = _read_dense_index(path)
    _validate_loaded_index(
        embeddings=index.embeddings,
        nct_ids=index.nct_ids,
        metadata=index.metadata,
        expected_nct_ids=None,
        expected_trials=trials_count,
        expected_corpus_fingerprint=corpus_fingerprint_value,
        model_name=model_name,
        text_representation=text_representation,
        max_seq_length=max_seq_length,
        validate_embedding_values=path.suffix.lower() != DENSE_INDEX_MMAP_SUFFIX,
    )
    return index


def _read_dense_index(path: Path) -> DenseIndex:
    if path.suffix.lower() == DENSE_INDEX_MMAP_SUFFIX:
        return _read_mmap_dense_index(path)
    np = _numpy()
    with np.load(path, allow_pickle=False) as payload:
        required_arrays = {"embeddings", "nct_ids", "metadata"}
        missing_arrays = required_arrays - set(payload.files)
        if missing_arrays:
            raise ValueError(f"Dense index is missing arrays: {', '.join(sorted(missing_arrays))}")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        nct_ids = tuple(str(value) for value in payload["nct_ids"].tolist())
        metadata_value = json.loads(str(payload["metadata"].item()))
        if not isinstance(metadata_value, dict):
            raise ValueError("Dense index metadata must be a JSON object")
        metadata = metadata_value

    return DenseIndex(embeddings=embeddings, nct_ids=nct_ids, metadata=metadata)


def load_or_build_dense_retriever(
    *,
    trials: Iterable[Trial],
    model_name: str,
    text_representation: str,
    batch_size: int,
    device: str,
    max_seq_length: int | None,
    index_path: Path,
    rebuild_index: bool = False,
    encoder_factory: Callable[[str, str, int | None], TextEncoder] | None = None,
    dynamic_quantization: bool = False,
    encoder_backend: str = "sentence-transformers",
    onnx_model_path: Path | None = None,
    show_progress_bar: bool = True,
) -> DenseRetriever:
    trial_list = list(trials)
    _validate_dense_parameters(
        model_name=model_name,
        text_representation=text_representation,
        batch_size=batch_size,
        device=device,
        max_seq_length=max_seq_length,
    )
    if encoder_factory is not None:
        encoder = encoder_factory(model_name, device, max_seq_length)
    else:
        framework = load_encoder_framework(encoder_backend)
        encoder = construct_text_encoder(
            backend=encoder_backend,
            framework=framework,
            model_name=model_name,
            device=device,
            max_seq_length=max_seq_length,
            onnx_model_path=onnx_model_path,
        )

    if index_path.exists() and not rebuild_index:
        index = load_dense_index(
            index_path,
            trial_list,
            model_name=model_name,
            text_representation=text_representation,
            max_seq_length=max_seq_length,
        )
    else:
        index = build_dense_index(
            trial_list,
            encoder=encoder,
            model_name=model_name,
            text_representation=text_representation,
            batch_size=batch_size,
            device=device,
            max_seq_length=max_seq_length,
            show_progress_bar=show_progress_bar,
        )
        save_dense_index(index_path, index)

    if dynamic_quantization:
        if encoder_backend != "sentence-transformers":
            raise ValueError(
                "Dynamic PyTorch quantization is only supported by sentence-transformers"
            )
        quantize = getattr(encoder, "quantize_dynamic_int8", None)
        if quantize is None:
            raise ValueError("Configured dense encoder does not support dynamic int8 quantization")
        quantize()

    return DenseRetriever(
        trial_list,
        index=index,
        encoder=encoder,
        batch_size=batch_size,
    )


def export_onnx_encoder(
    *,
    model_name: str,
    output_path: Path,
    device: str,
    max_seq_length: int,
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("The ONNX Runtime prototype currently supports only CPU export")
    if max_seq_length < 1:
        raise ValueError("ONNX encoder max sequence length must be at least 1")
    try:
        torch = import_module("torch")
        import_module("onnx")
    except ImportError as exc:
        raise RuntimeError(
            'Install export dependencies with `python -m pip install -e ".[dense,onnx]"`.'
        ) from exc

    framework = load_encoder_framework("sentence-transformers")
    encoder = construct_text_encoder(
        backend="sentence-transformers",
        framework=framework,
        model_name=model_name,
        device=device,
        max_seq_length=max_seq_length,
    )
    model = encoder.model
    if len(model) != 3:
        raise ValueError("ONNX export expects Transformer, mean Pooling, and Normalize modules")
    transformer = model[0]
    pooling = model[1]
    normalize = model[2]
    pooling_mode = getattr(pooling, "pooling_mode", None)
    legacy_mean_pooling = getattr(pooling, "pooling_mode_mean_tokens", False)
    if pooling_mode != "mean" and not legacy_mean_pooling:
        raise ValueError("ONNX export currently supports sentence-transformers mean pooling")
    if normalize.__class__.__name__ != "Normalize":
        raise ValueError("ONNX export expects a normalized sentence-transformer model")

    class MeanPoolingEncoder(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, auto_model: Any) -> None:
            super().__init__()
            self.auto_model = auto_model

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any,
        ) -> Any:
            token_embeddings = self.auto_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )[0]
            expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = torch.sum(token_embeddings * expanded_mask, 1) / torch.clamp(
                expanded_mask.sum(1),
                min=1e-9,
            )
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    export_model = MeanPoolingEncoder(transformer.auto_model).eval()
    tokenizer = transformer.tokenizer
    encoded = tokenizer(
        ["Clinical trial encoder export."],
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
    input_names = ("input_ids", "attention_mask", "token_type_ids")
    output_path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output_path,
        prefix=".model.",
        suffix=".onnx.tmp",
        delete=False,
    ) as handle:
        temporary_model_path = Path(handle.name)
    try:
        torch.onnx.export(
            export_model,
            tuple(encoded[name] for name in input_names),
            temporary_model_path,
            input_names=list(input_names),
            output_names=["sentence_embedding"],
            dynamic_axes={
                name: {0: "batch", 1: "sequence"} for name in input_names
            }
            | {"sentence_embedding": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        temporary_model_path.replace(output_path / "model.onnx")
    finally:
        if temporary_model_path.exists():
            temporary_model_path.unlink()

    tokenizer_path = output_path / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tokenizer_path))
    with torch.inference_mode():
        embedding_dimension = int(export_model(*(encoded[name] for name in input_names)).shape[1])
    metadata = {
        "schema_version": ONNX_ENCODER_SCHEMA_VERSION,
        "backend": "onnxruntime",
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "embedding_dimension": embedding_dimension,
        "normalize_embeddings": True,
        "pooling": "mean",
        "quantization": "fp32",
        "opset_version": 17,
        "input_names": list(input_names),
        "output_name": "sentence_embedding",
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "pad_token_type_id": tokenizer.pad_token_type_id,
        "model_sha256": _sha256_file(output_path / "model.onnx"),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
    }
    _atomic_write_text(output_path / "metadata.json", json.dumps(metadata, sort_keys=True))
    return metadata


def trial_text(trial: Trial, representation: str) -> str:
    fields = TEXT_REPRESENTATIONS.get(representation)
    if fields is None:
        raise ValueError(
            f"Unknown dense text representation {representation!r}; expected one of "
            + ", ".join(sorted(TEXT_REPRESENTATIONS))
        )
    values = _trial_field_values(trial)
    parts = [f"{_field_label(field)}: {values[field]}" for field in fields if values[field]]
    return "\n".join(parts) or f"NCT ID: {trial.nct_id}"


def _trial_field_values(trial: Trial) -> dict[str, str]:
    return {
        "title": trial.title,
        "brief_summary": trial.brief_summary,
        "conditions": "; ".join(trial.conditions),
        "interventions": "; ".join(trial.interventions),
        "eligibility_criteria": trial.eligibility_criteria,
        "demographics": "; ".join(
            value for value in (trial.sex, trial.minimum_age, trial.maximum_age) if value
        ),
        "status": "; ".join(
            value for value in (trial.status, *trial.phases, trial.study_type) if value
        ),
        "locations": "; ".join(trial.locations),
    }


def _field_label(field_name: str) -> str:
    return field_name.replace("_", " ").title()


def _normalized_embeddings(values: Any, *, expected_rows: int) -> Any:
    np = _numpy()
    embeddings = np.asarray(values, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Encoder returned shape {embeddings.shape}; expected {expected_rows} rows"
        )
    if embeddings.shape[1] < 1:
        raise ValueError("Encoder returned zero-dimensional embeddings")
    if not np.isfinite(embeddings).all():
        raise ValueError("Encoder returned non-finite embeddings")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encoder returned a zero-length embedding")
    return np.asarray(embeddings / norms, dtype=np.float32)


def _validate_loaded_index(
    *,
    embeddings: Any,
    nct_ids: tuple[str, ...],
    metadata: dict[str, Any],
    expected_nct_ids: tuple[str, ...] | None,
    expected_trials: int,
    expected_corpus_fingerprint: str,
    model_name: str,
    text_representation: str,
    max_seq_length: int | None,
    validate_embedding_values: bool,
) -> None:
    expected = {
        "schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "retriever": DENSE_RETRIEVER_NAME,
        "model_name": model_name,
        "text_representation": text_representation,
        "max_seq_length": max_seq_length,
        "corpus_fingerprint": expected_corpus_fingerprint,
    }
    mismatches = [
        key for key, expected_value in expected.items() if metadata.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "Dense index is incompatible with the requested corpus/config: "
            + ", ".join(mismatches)
        )
    if expected_nct_ids is not None and nct_ids != expected_nct_ids:
        raise ValueError("Dense index NCT ID order does not match the trial corpus")
    if len(nct_ids) != expected_trials:
        raise ValueError("Dense index NCT ID count does not match the trial corpus")
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_trials:
        raise ValueError("Dense index embedding shape does not match the trial corpus")
    if int(metadata.get("embedding_dimension", 0)) != embeddings.shape[1]:
        raise ValueError("Dense index embedding dimension metadata is inconsistent")
    if validate_embedding_values:
        if not _numpy().isfinite(embeddings).all():
            raise ValueError("Dense index contains non-finite embeddings")
        norms = _numpy().linalg.norm(embeddings, axis=1)
        if not all(
            math.isclose(float(norm), 1.0, rel_tol=1e-5, abs_tol=1e-5)
            for norm in norms
        ):
            raise ValueError("Dense index embeddings are not normalized")


def _save_mmap_dense_index(path: Path, index: DenseIndex) -> None:
    np = _numpy()
    path.mkdir(parents=True, exist_ok=True)
    embeddings_path = path / "embeddings.npy"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path,
        prefix=".embeddings.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_embeddings = Path(handle.name)
        np.save(handle, np.asarray(index.embeddings, dtype=np.float32), allow_pickle=False)
    temporary_embeddings.replace(embeddings_path)
    metadata = {**index.metadata, "storage_format": "npy_memmap"}
    _atomic_write_text(path / "nct_ids.json", json.dumps(index.nct_ids))
    _atomic_write_text(path / "metadata.json", json.dumps(metadata, sort_keys=True))


def _read_mmap_dense_index(path: Path) -> DenseIndex:
    np = _numpy()
    required = (path / "embeddings.npy", path / "nct_ids.json", path / "metadata.json")
    missing = [candidate.name for candidate in required if not candidate.is_file()]
    if missing:
        raise ValueError(f"Memory-mapped dense index is missing files: {', '.join(missing)}")
    embeddings = np.load(path / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if embeddings.dtype != np.float32:
        raise ValueError("Memory-mapped dense embeddings must use float32")
    nct_ids_value = json.loads((path / "nct_ids.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(nct_ids_value, list) or not all(
        isinstance(value, str) for value in nct_ids_value
    ):
        raise ValueError("Memory-mapped dense index NCT IDs must be a JSON string list")
    if not isinstance(metadata, dict):
        raise ValueError("Memory-mapped dense index metadata must be a JSON object")
    return DenseIndex(
        embeddings=embeddings,
        nct_ids=tuple(nct_ids_value),
        metadata=metadata,
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(value)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_dense_parameters(
    *,
    model_name: str,
    text_representation: str,
    batch_size: int,
    device: str,
    max_seq_length: int | None,
) -> None:
    if not model_name.strip():
        raise ValueError("Dense model name cannot be empty")
    if text_representation not in TEXT_REPRESENTATIONS:
        raise ValueError(
            f"Unknown dense text representation {text_representation!r}; expected one of "
            + ", ".join(sorted(TEXT_REPRESENTATIONS))
        )
    if batch_size < 1:
        raise ValueError("Dense batch size must be at least 1")
    if not device.strip():
        raise ValueError("Dense device cannot be empty")
    if max_seq_length is not None and max_seq_length < 1:
        raise ValueError("Dense max sequence length must be at least 1")


def _validate_encoder_backend(backend: str) -> None:
    if backend not in ENCODER_BACKENDS:
        raise ValueError(
            f"Unknown dense encoder backend {backend!r}; expected one of "
            + ", ".join(sorted(ENCODER_BACKENDS))
        )


def _sentence_transformer_encoder(
    model_name: str,
    device: str,
    max_seq_length: int | None,
) -> TextEncoder:
    return SentenceTransformerEncoder(
        model_name,
        device=device,
        max_seq_length=max_seq_length,
    )


def _sentence_transformers() -> Any:
    try:
        return import_module("sentence_transformers")
    except ImportError as exc:
        raise RuntimeError(
            'Install dense retrieval dependencies with `python -m pip install -e ".[dense]"`.'
        ) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy() -> Any:
    try:
        return import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            'Install dense retrieval dependencies with `python -m pip install -e ".[dense]"`.'
        ) from exc
