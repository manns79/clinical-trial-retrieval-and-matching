from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SourceManifest:
    name: str
    source_url: str
    local_path: str
    sha256: str
    bytes: int
    created_at_utc: str
    schema_version: str = MANIFEST_SCHEMA_VERSION
    dataset: str = ""
    year: int | None = None
    parser: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def build_source_manifest(
    *,
    name: str,
    source_url: str,
    input_path: Path,
    dataset: str = "",
    year: int | None = None,
    parser: str = "",
    metadata: dict[str, str] | None = None,
) -> SourceManifest:
    if not input_path.exists():
        raise FileNotFoundError(f"Manifest input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Manifest input path is not a file: {input_path}")

    stat = input_path.stat()
    return SourceManifest(
        name=name,
        source_url=source_url,
        local_path=str(input_path),
        sha256=sha256_file(input_path),
        bytes=stat.st_size,
        created_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        dataset=dataset,
        year=year,
        parser=parser,
        metadata=metadata or {},
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_to_json_record(manifest: SourceManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "name": manifest.name,
        "dataset": manifest.dataset,
        "year": manifest.year,
        "source_url": manifest.source_url,
        "local_path": manifest.local_path,
        "sha256": manifest.sha256,
        "bytes": manifest.bytes,
        "created_at_utc": manifest.created_at_utc,
        "parser": manifest.parser,
        "metadata": manifest.metadata,
    }
