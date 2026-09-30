"""Read and write the ``<file>.yaml`` metadata sidecar stored next to each source PDF."""

from pathlib import Path

import yaml

from reglens.models import DocumentMetadata


def sidecar_path(pdf: Path) -> Path:
    return pdf.with_suffix(".yaml")


def read_sidecar(pdf: Path) -> DocumentMetadata:
    data = yaml.safe_load(sidecar_path(pdf).read_text(encoding="utf-8"))
    return DocumentMetadata.model_validate(data)


def write_sidecar(pdf: Path, metadata: DocumentMetadata) -> None:
    write_yaml(sidecar_path(pdf), metadata.model_dump(mode="json"))


def write_yaml(path: Path, data: object) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8"
    )
