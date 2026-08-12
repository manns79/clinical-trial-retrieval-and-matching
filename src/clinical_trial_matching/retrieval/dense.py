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
TEXT_REPRESENTATIONS = {
    "title": ("title",),
    "title_summary_conditions": ("title", "brief_summary", "conditions"),
    "clinical_core": (
        "title",
        "brief_summary",
        "conditions",
        "interventions",
        "eligibility_criteria",
        "demographics",
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
        trials: Iterable[Trial],
        *,
        index: DenseIndex,
        encoder: TextEncoder,
        batch_size: int,
    ) -> None:
        self.trials = list(trials)
        self.index = index
        self.encoder = encoder
        self.batch_size = batch_size
        if len(self.trials) != len(self.index.nct_ids):
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
        result_count = min(top_k, len(self.trials))
        rankings: list[list[SearchResult]] = []
        for query_scores in scores:
            ranked_indexes = np.argsort(-query_scores, kind="stable")[:result_count]
            rankings.append(
                [
                    SearchResult(
                        nct_id=self.trials[int(index)].nct_id,
                        score=float(query_scores[int(index)]),
                        rank=rank,
                        title=self.trials[int(index)].title,
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
    if path.suffix.lower() != ".npz":
        raise ValueError("Dense index output must use the .npz suffix")
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
    np = _numpy()
    trial_list = list(trials)
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

    _validate_loaded_index(
        embeddings=embeddings,
        nct_ids=nct_ids,
        metadata=metadata,
        trials=trial_list,
        model_name=model_name,
        text_representation=text_representation,
        max_seq_length=max_seq_length,
    )
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
    trials: list[Trial],
    model_name: str,
    text_representation: str,
    max_seq_length: int | None,
) -> None:
    expected = {
        "schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "retriever": DENSE_RETRIEVER_NAME,
        "model_name": model_name,
        "text_representation": text_representation,
        "max_seq_length": max_seq_length,
        "corpus_fingerprint": corpus_fingerprint(trials),
    }
    mismatches = [
        key for key, expected_value in expected.items() if metadata.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "Dense index is incompatible with the requested corpus/config: "
            + ", ".join(mismatches)
        )
    expected_ids = tuple(trial.nct_id for trial in trials)
    if nct_ids != expected_ids:
        raise ValueError("Dense index NCT ID order does not match the trial corpus")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(trials):
        raise ValueError("Dense index embedding shape does not match the trial corpus")
    if int(metadata.get("embedding_dimension", 0)) != embeddings.shape[1]:
        raise ValueError("Dense index embedding dimension metadata is inconsistent")
    if not _numpy().isfinite(embeddings).all():
        raise ValueError("Dense index contains non-finite embeddings")
    norms = _numpy().linalg.norm(embeddings, axis=1)
    if not all(math.isclose(float(norm), 1.0, rel_tol=1e-5, abs_tol=1e-5) for norm in norms):
        raise ValueError("Dense index embeddings are not normalized")


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
