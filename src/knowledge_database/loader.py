import json
from pathlib import Path
from typing import Any, Dict, Union
import math

import yaml


class KnowledgeValidationError(ValueError):
    """Raised when the knowledge index or its references are invalid."""

def validate_candidate_metrics(record: Dict[str, Any]) -> None:
    candidate_id = record.get("candidate_id", "<unknown>")

    metric_rules = [
        ("accuracy", "map50_95", 0, 1),
        ("benchmark", "median_latency_ms", 0, None),
        ("model", "model_size_mb", 0, None),
    ]

    for section_name, field_name, minimum, maximum in metric_rules:
        section = record.get(section_name)

        if not isinstance(section, dict):
            raise KnowledgeValidationError(
                f"{candidate_id}: {section_name} must be an object."
            )

        value = section.get(field_name)
        metric_name = f"{section_name}.{field_name}"

        # bool phải được kiểm tra riêng vì isinstance(True, int) là True.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise KnowledgeValidationError(
                f"{candidate_id}: {metric_name} must be a number."
            )

        if isinstance(value, float) and not math.isfinite(value):
            raise KnowledgeValidationError(
                f"{candidate_id}: {metric_name} must be finite."
            )

        if maximum is not None:
            if not minimum <= value <= maximum:
                raise KnowledgeValidationError(
                    f"{candidate_id}: {metric_name} must be "
                    f"between {minimum} and {maximum}."
                )
        elif value <= minimum:
            raise KnowledgeValidationError(
                f"{candidate_id}: {metric_name} must be greater than {minimum}."
            )
def load_knowledge_base(
    index_file: Union[str, Path],
    project_root: Union[str, Path],
) -> Dict[str, Any]:
    root = Path(project_root).resolve()

    index_path = Path(index_file)
    if not index_path.is_absolute():
        index_path = root / index_path

    with index_path.open(encoding="utf-8") as file:
        index = yaml.safe_load(file)

    # Kiểm tra cấu trúc index.
    if not isinstance(index, dict):
        raise KnowledgeValidationError(
            "Knowledge index must be a mapping."
        )

    expected_fields = {
        "schema_version",
        "knowledge_base_id",
        "candidate_records",
    }
    if set(index) != expected_fields:
        raise KnowledgeValidationError(
            f"Knowledge index must contain exactly: {sorted(expected_fields)}"
        )

    if index["schema_version"] != "1.0":
        raise KnowledgeValidationError(
            "Unsupported knowledge schema version."
        )

    knowledge_id = index["knowledge_base_id"]
    if not isinstance(knowledge_id, str) or not knowledge_id.strip():
        raise KnowledgeValidationError(
            "knowledge_base_id must be a non-empty string."
        )

    entries = index["candidate_records"]
    if not isinstance(entries, list):
        raise KnowledgeValidationError(
            "candidate_records must be a list."
        )

    candidates = []
    seen_ids = set()

    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "candidate_id",
            "record_path",
        }:
            raise KnowledgeValidationError(
                "Each entry must contain candidate_id and record_path."
            )

        candidate_id = entry["candidate_id"]
        record_path = entry["record_path"]

        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise KnowledgeValidationError(
                "candidate_id must be a non-empty string."
            )

        if candidate_id in seen_ids:
            raise KnowledgeValidationError(
                f"Duplicate candidate ID: {candidate_id}"
            )

        if not isinstance(record_path, str) or not record_path.strip():
            raise KnowledgeValidationError(
                f"Invalid record_path for {candidate_id}."
            )

        # Các đường dẫn record phải tương đối với project root.
        relative_path = Path(record_path)
        if relative_path.is_absolute():
            raise KnowledgeValidationError(
                f"record_path must be relative: {record_path}"
            )

        full_path = (root / relative_path).resolve()

        try:
            normalized_path = full_path.relative_to(root)
        except ValueError:
            raise KnowledgeValidationError(
                f"Record path is outside the project: {record_path}"
            ) from None

        with full_path.open(encoding="utf-8") as file:
            record = json.load(file)

        if not isinstance(record, dict):
            raise KnowledgeValidationError(
                f"Candidate record must be an object: {record_path}"
            )

        if record.get("candidate_id") != candidate_id:
            raise KnowledgeValidationError(
                f"Candidate ID mismatch: {record_path}"
            )
        validate_candidate_metrics(record)

        seen_ids.add(candidate_id)

        candidates.append({
            "record_path": normalized_path.as_posix(),
            "record": record,
        })

    return {
        "schema_version": index["schema_version"],
        "knowledge_base_id": knowledge_id,
        "candidates": candidates,
    }