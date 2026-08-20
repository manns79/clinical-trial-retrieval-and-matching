from __future__ import annotations

import json
import math
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


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, *, device: str, max_seq_length: int | None) -> None:
        try:
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                'Install dense retrieval dependencies with `python -m pip install -e ".[dense]"`.'
            ) from exc

        self.model = sentence_transformers.SentenceTransformer(model_name, device=device)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.quantization = "fp32"

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
        for query_scores in scores:
            ranked_indexes = np.argsort(-query_scores, kind="stable")[:result_count]
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
    factory = encoder_factory or _sentence_transformer_encoder
    encoder = factory(model_name, device, max_seq_length)

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


def _numpy() -> Any:
    try:
        return import_module("numpy")
    except ImportError as exc:
        raise RuntimeError(
            'Install dense retrieval dependencies with `python -m pip install -e ".[dense]"`.'
        ) from exc
