import argparse
import json
from pathlib import Path
from typing import Any, Dict, Set, Union

import yaml

from .schema import Constraints, Preferences, Requirement, Target


SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
SUPPORTED_TASKS = {"object_detection"}
SUPPORTED_DEVICES = {"cpu"}
SUPPORTED_DATASETS = {"KITTI"}
SUPPORTED_OPTIMIZATION_GOALS = {
    "accuracy",
    "latency",
    "model_size",
    "balanced",
}


class RequirementValidationError(ValueError):
    """Raised when a requirement file fails validation."""


def require_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RequirementValidationError(
            f"'{field_name}' must be a mapping/object."
        )

    return value


def validate_keys(
    data: Dict[str, Any],
    required_keys: Set[str],
    field_name: str,
) -> None:
    actual_keys = set(data.keys())

    missing_keys = required_keys - actual_keys
    unknown_keys = actual_keys - required_keys

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise RequirementValidationError(
            f"Missing field(s) in '{field_name}': {missing}"
        )

    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise RequirementValidationError(
            f"Unknown field(s) in '{field_name}': {unknown}"
        )


def require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequirementValidationError(
            f"'{field_name}' must be a non-empty string."
        )

    return value.strip()


def require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequirementValidationError(
            f"'{field_name}' must be a number."
        )

    return float(value)


def parse_requirement_data(data: Dict[str, Any]) -> Requirement:
    validate_keys(
        data,
        {
            "schema_version",
            "request_id",
            "target",
            "constraints",
            "preferences",
        },
        "root",
    )

    schema_version = require_non_empty_string(
        data["schema_version"],
        "schema_version",
    )

    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise RequirementValidationError(
            f"Unsupported schema_version: {schema_version}"
        )

    request_id = require_non_empty_string(
        data["request_id"],
        "request_id",
    )

    target_data = require_mapping(data["target"], "target")
    validate_keys(
        target_data,
        {"task", "device", "dataset"},
        "target",
    )

    task = require_non_empty_string(
        target_data["task"],
        "target.task",
    )
    device = require_non_empty_string(
        target_data["device"],
        "target.device",
    )
    dataset = require_non_empty_string(
        target_data["dataset"],
        "target.dataset",
    )

    if task not in SUPPORTED_TASKS:
        raise RequirementValidationError(
            f"Unsupported task: {task}"
        )

    if device not in SUPPORTED_DEVICES:
        raise RequirementValidationError(
            f"Unsupported target device: {device}"
        )

    if dataset not in SUPPORTED_DATASETS:
        raise RequirementValidationError(
            f"Unsupported dataset: {dataset}"
        )

    constraints_data = require_mapping(
        data["constraints"],
        "constraints",
    )
    validate_keys(
        constraints_data,
        {
            "minimum_map50_95",
            "maximum_median_latency_ms",
            "maximum_model_size_mb",
        },
        "constraints",
    )

    minimum_map50_95 = require_number(
        constraints_data["minimum_map50_95"],
        "constraints.minimum_map50_95",
    )
    maximum_median_latency_ms = require_number(
        constraints_data["maximum_median_latency_ms"],
        "constraints.maximum_median_latency_ms",
    )
    maximum_model_size_mb = require_number(
        constraints_data["maximum_model_size_mb"],
        "constraints.maximum_model_size_mb",
    )

    if not 0.0 <= minimum_map50_95 <= 1.0:
        raise RequirementValidationError(
            "'constraints.minimum_map50_95' must be between 0 and 1."
        )

    if maximum_median_latency_ms <= 0:
        raise RequirementValidationError(
            "'constraints.maximum_median_latency_ms' must be greater than 0."
        )

    if maximum_model_size_mb <= 0:
        raise RequirementValidationError(
            "'constraints.maximum_model_size_mb' must be greater than 0."
        )

    preferences_data = require_mapping(
        data["preferences"],
        "preferences",
    )
    validate_keys(
        preferences_data,
        {"optimization_goal"},
        "preferences",
    )

    optimization_goal = require_non_empty_string(
        preferences_data["optimization_goal"],
        "preferences.optimization_goal",
    )

    if optimization_goal not in SUPPORTED_OPTIMIZATION_GOALS:
        supported = ", ".join(
            sorted(SUPPORTED_OPTIMIZATION_GOALS)
        )
        raise RequirementValidationError(
            f"Unsupported optimization goal: {optimization_goal}. "
            f"Supported values: {supported}"
        )

    return Requirement(
        schema_version=schema_version,
        request_id=request_id,
        target=Target(
            task=task,
            device=device,
            dataset=dataset,
        ),
        constraints=Constraints(
            minimum_map50_95=minimum_map50_95,
            maximum_median_latency_ms=maximum_median_latency_ms,
            maximum_model_size_mb=maximum_model_size_mb,
        ),
        preferences=Preferences(
            optimization_goal=optimization_goal,
        ),
    )


def load_requirement(
    file_path: Union[str, Path],
) -> Requirement:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Requirement file not found: {path}"
        )

    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise RequirementValidationError(
            "Requirement file must use .yaml, .yml, or .json."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise RequirementValidationError(
            f"Invalid YAML or JSON syntax: {error}"
        ) from error

    root_data = require_mapping(raw_data, "root")
    return parse_requirement_data(root_data)


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Parse and validate an EdgeNAS-Lite requirement."
    )
    argument_parser.add_argument(
        "request_file",
        help="Path to a requirement YAML or JSON file.",
    )
    arguments = argument_parser.parse_args()

    try:
        requirement = load_requirement(arguments.request_file)
    except (
        FileNotFoundError,
        RequirementValidationError,
    ) as error:
        argument_parser.error(str(error))

    print(json.dumps(requirement.to_dict(), indent=2))


if __name__ == "__main__":
    main()