import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from src.requirement_parser.parser import (
    RequirementValidationError,
    load_requirement,
)
from src.requirement_parser.schema import Requirement


class CandidateValidationError(ValueError):
    """Raised when a candidate record is missing or contains invalid data."""


def get_nested_value(
    data: Dict[str, Any],
    path: Tuple[str, ...],
) -> Any:
    """Read a value from a nested dictionary."""

    current: Any = data

    for key in path:
        if not isinstance(current, dict) or key not in current:
            field_name = ".".join(path)
            raise CandidateValidationError(
                f"Missing candidate field: '{field_name}'."
            )

        current = current[key]

    return current


def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateValidationError(
            f"'{field_name}' must be a non-empty string."
        )

    return value.strip()


def require_number(
    value: Any,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise CandidateValidationError(
            f"'{field_name}' must be a finite number."
        )

    return float(value)


def load_candidate(
    file_path: Union[str, Path],
) -> Dict[str, Any]:
    """Load a candidate JSON file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {path}"
        )

    if path.suffix.lower() != ".json":
        raise CandidateValidationError(
            "Candidate file must use the .json extension."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as error:
        raise CandidateValidationError(
            f"Invalid candidate JSON syntax: {error}"
        ) from error

    if not isinstance(raw_data, dict):
        raise CandidateValidationError(
            "Candidate root must be a mapping/object."
        )

    return raw_data


def evaluate_candidate(
    requirement: Requirement,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one candidate against one validated requirement."""

    candidate_id = require_non_empty_string(
        get_nested_value(candidate, ("candidate_id",)),
        "candidate_id",
    )

    candidate_dataset = require_non_empty_string(
        get_nested_value(candidate, ("accuracy", "dataset")),
        "accuracy.dataset",
    )

    candidate_device = require_non_empty_string(
        get_nested_value(candidate, ("benchmark", "device")),
        "benchmark.device",
    )

    if candidate_dataset != requirement.target.dataset:
        raise CandidateValidationError(
            "Dataset mismatch: requirement expects "
            f"'{requirement.target.dataset}', but candidate uses "
            f"'{candidate_dataset}'."
        )

    if candidate_device != requirement.target.device:
        raise CandidateValidationError(
            "Device mismatch: requirement expects "
            f"'{requirement.target.device}', but candidate was "
            f"benchmarked on '{candidate_device}'."
        )

    actual_map50_95 = require_number(
        get_nested_value(
            candidate,
            ("accuracy", "map50_95"),
        ),
        "accuracy.map50_95",
    )

    actual_median_latency_ms = require_number(
        get_nested_value(
            candidate,
            ("benchmark", "median_latency_ms"),
        ),
        "benchmark.median_latency_ms",
    )

    actual_model_size_mb = require_number(
        get_nested_value(
            candidate,
            ("model", "model_size_mb"),
        ),
        "model.model_size_mb",
    )

    if not 0.0 <= actual_map50_95 <= 1.0:
        raise CandidateValidationError(
            "'accuracy.map50_95' must be between 0 and 1."
        )

    if actual_median_latency_ms <= 0:
        raise CandidateValidationError(
            "'benchmark.median_latency_ms' must be greater than 0."
        )

    if actual_model_size_mb <= 0:
        raise CandidateValidationError(
            "'model.model_size_mb' must be greater than 0."
        )

    constraints = requirement.constraints

    checks = {
        "minimum_map50_95": {
            "metric_path": "accuracy.map50_95",
            "operator": ">=",
            "required": constraints.minimum_map50_95,
            "actual": actual_map50_95,
            "passed": (
                actual_map50_95
                >= constraints.minimum_map50_95
            ),
        },
        "maximum_median_latency_ms": {
            "metric_path": "benchmark.median_latency_ms",
            "operator": "<=",
            "required": constraints.maximum_median_latency_ms,
            "actual": actual_median_latency_ms,
            "passed": (
                actual_median_latency_ms
                <= constraints.maximum_median_latency_ms
            ),
        },
        "maximum_model_size_mb": {
            "metric_path": "model.model_size_mb",
            "operator": "<=",
            "required": constraints.maximum_model_size_mb,
            "actual": actual_model_size_mb,
            "passed": (
                actual_model_size_mb
                <= constraints.maximum_model_size_mb
            ),
        },
    }

    constraints_satisfied = all(
        check["passed"] for check in checks.values()
    )

    return {
        "schema_version": "1.0",
        "evaluation_id": (
            f"{requirement.request_id}__{candidate_id}"
        ),
        "request_id": requirement.request_id,
        "candidate_id": candidate_id,
        "target": {
            "task": requirement.target.task,
            "device": requirement.target.device,
            "dataset": requirement.target.dataset,
        },
        "checks": checks,
        "constraints_satisfied": constraints_satisfied,
    }


def save_evaluation(
    evaluation: Dict[str, Any],
    output_file: Union[str, Path],
) -> None:
    """Save an evaluation result as JSON."""

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            evaluation,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Evaluate an EdgeNAS-Lite candidate "
            "against a requirement."
        )
    )

    argument_parser.add_argument(
        "request_file",
        help="Path to the requirement YAML or JSON file.",
    )

    argument_parser.add_argument(
        "candidate_file",
        help="Path to the candidate JSON file.",
    )

    argument_parser.add_argument(
        "--output",
        help="Optional path for the evaluation JSON file.",
    )

    arguments = argument_parser.parse_args()

    try:
        requirement = load_requirement(
            arguments.request_file
        )
        candidate = load_candidate(
            arguments.candidate_file
        )
        evaluation = evaluate_candidate(
            requirement,
            candidate,
        )

        if arguments.output:
            save_evaluation(
                evaluation,
                arguments.output,
            )

    except (
        FileNotFoundError,
        RequirementValidationError,
        CandidateValidationError,
    ) as error:
        argument_parser.error(str(error))

    print(
        json.dumps(
            evaluation,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()