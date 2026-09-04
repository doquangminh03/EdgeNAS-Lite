import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import yaml


SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
SUPPORTED_SEARCH_TYPES = {
    "deployment_configuration",
}
SUPPORTED_STRATEGIES = {"grid"}
SUPPORTED_TASKS = {"object_detection"}
SUPPORTED_DEVICES = {"cpu"}
SUPPORTED_DATASETS = {"KITTI"}
SUPPORTED_ACCURACY_METRICS = {"map50_95"}
SUPPORTED_ACCURACY_SOURCES = {
    "full_validation",
}

REQUIRED_METRICS = {
    "accuracy.map50_95",
    "benchmark.median_latency_ms",
    "benchmark.p95_latency_ms",
    "model.model_size_mb",
}


class SearchSpaceValidationError(ValueError):
    """Raised when a search-space file is invalid."""


def require_mapping(
    value: Any,
    field_name: str,
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SearchSpaceValidationError(
            f"'{field_name}' must be a mapping/object."
        )

    return value


def require_list(
    value: Any,
    field_name: str,
) -> List[Any]:
    if not isinstance(value, list):
        raise SearchSpaceValidationError(
            f"'{field_name}' must be a list."
        )

    return value


def require_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SearchSpaceValidationError(
            f"'{field_name}' must be a non-empty string."
        )

    return value.strip()


def require_integer(
    value: Any,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SearchSpaceValidationError(
            f"'{field_name}' must be an integer."
        )

    return value


def require_number(
    value: Any,
    field_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise SearchSpaceValidationError(
            f"'{field_name}' must be a number."
        )

    return float(value)


def require_boolean(
    value: Any,
    field_name: str,
) -> bool:
    if not isinstance(value, bool):
        raise SearchSpaceValidationError(
            f"'{field_name}' must be true or false."
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
        raise SearchSpaceValidationError(
            f"Missing field(s) in '{field_name}': "
            f"{missing}"
        )

    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise SearchSpaceValidationError(
            f"Unknown field(s) in '{field_name}': "
            f"{unknown}"
        )


def validate_supported_value(
    value: str,
    supported_values: Set[str],
    field_name: str,
) -> None:
    if value not in supported_values:
        supported = ", ".join(
            sorted(supported_values)
        )

        raise SearchSpaceValidationError(
            f"Unsupported value for '{field_name}': "
            f"{value}. Supported values: {supported}"
        )


def parse_search_space_data(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    validate_keys(
        data,
        {
            "schema_version",
            "search_space_id",
            "search",
            "fixed_configuration",
            "variable_dimensions",
            "evaluation",
            "candidate_naming",
            "existing_candidates",
        },
        "root",
    )

    schema_version = require_string(
        data["schema_version"],
        "schema_version",
    )
    validate_supported_value(
        schema_version,
        SUPPORTED_SCHEMA_VERSIONS,
        "schema_version",
    )

    require_string(
        data["search_space_id"],
        "search_space_id",
    )

    search = require_mapping(
        data["search"],
        "search",
    )
    validate_keys(
        search,
        {
            "type",
            "strategy",
            "experiment_budget",
        },
        "search",
    )

    search_type = require_string(
        search["type"],
        "search.type",
    )
    validate_supported_value(
        search_type,
        SUPPORTED_SEARCH_TYPES,
        "search.type",
    )

    strategy = require_string(
        search["strategy"],
        "search.strategy",
    )
    validate_supported_value(
        strategy,
        SUPPORTED_STRATEGIES,
        "search.strategy",
    )

    experiment_budget = require_integer(
        search["experiment_budget"],
        "search.experiment_budget",
    )

    if experiment_budget <= 0:
        raise SearchSpaceValidationError(
            "'search.experiment_budget' "
            "must be greater than 0."
        )

    fixed = require_mapping(
        data["fixed_configuration"],
        "fixed_configuration",
    )
    validate_keys(
        fixed,
        {
            "task",
            "model",
            "dataset",
            "deployment",
            "prediction",
        },
        "fixed_configuration",
    )

    task = require_string(
        fixed["task"],
        "fixed_configuration.task",
    )
    validate_supported_value(
        task,
        SUPPORTED_TASKS,
        "fixed_configuration.task",
    )

    model = require_mapping(
        fixed["model"],
        "fixed_configuration.model",
    )
    validate_keys(
        model,
        {
            "family",
            "scale",
            "checkpoint",
            "retrain",
        },
        "fixed_configuration.model",
    )

    require_string(
        model["family"],
        "fixed_configuration.model.family",
    )
    require_string(
        model["scale"],
        "fixed_configuration.model.scale",
    )
    require_string(
        model["checkpoint"],
        "fixed_configuration.model.checkpoint",
    )

    retrain = require_boolean(
        model["retrain"],
        "fixed_configuration.model.retrain",
    )

    if (
        search_type == "deployment_configuration"
        and retrain
    ):
        raise SearchSpaceValidationError(
            "A deployment-configuration search "
            "must use 'retrain: false'."
        )

    dataset = require_mapping(
        fixed["dataset"],
        "fixed_configuration.dataset",
    )
    validate_keys(
        dataset,
        {
            "name",
            "data_config",
            "validation_split",
            "validation_images",
            "classes",
            "class_ids",
        },
        "fixed_configuration.dataset",
    )

    dataset_name = require_string(
        dataset["name"],
        "fixed_configuration.dataset.name",
        
    )
    require_string(
        dataset["data_config"],
        "fixed_configuration.dataset.data_config",
    )
    validate_supported_value(
        dataset_name,
        SUPPORTED_DATASETS,
        "fixed_configuration.dataset.name",
    )

    require_string(
        dataset["validation_split"],
        "fixed_configuration.dataset.validation_split",
    )

    validation_images = require_integer(
        dataset["validation_images"],
        "fixed_configuration.dataset.validation_images",
    )

    if validation_images <= 0:
        raise SearchSpaceValidationError(
            "'validation_images' must be greater than 0."
        )

    classes = require_list(
        dataset["classes"],
        "fixed_configuration.dataset.classes",
    )

    if not classes:
        raise SearchSpaceValidationError(
            "'fixed_configuration.dataset.classes' "
            "cannot be empty."
        )

    normalized_classes = [
        require_string(
            class_name,
            "fixed_configuration.dataset.classes",
        )
        for class_name in classes
    ]

    if len(normalized_classes) != len(
        set(normalized_classes)
    ):
        raise SearchSpaceValidationError(
            "Dataset classes must be unique."
        )

    class_ids = require_list(
        dataset["class_ids"],
        "fixed_configuration.dataset.class_ids",
    )

    normalized_class_ids = [
        require_integer(
            class_id,
            "fixed_configuration.dataset.class_ids",
        )
        for class_id in class_ids
    ]

    if any(
        class_id < 0
        for class_id in normalized_class_ids
    ):
        raise SearchSpaceValidationError(
            "Dataset class IDs cannot be negative."
        )

    if len(normalized_class_ids) != len(
        set(normalized_class_ids)
    ):
        raise SearchSpaceValidationError(
            "Dataset class IDs must be unique."
        )

    if len(normalized_classes) != len(
        normalized_class_ids
    ):
        raise SearchSpaceValidationError(
            "Dataset classes and class IDs "
            "must have the same length."
        )

    deployment = require_mapping(
        fixed["deployment"],
        "fixed_configuration.deployment",
    )
    validate_keys(
        deployment,
        {"device", "batch_size"},
        "fixed_configuration.deployment",
    )

    device = require_string(
        deployment["device"],
        "fixed_configuration.deployment.device",
    )
    validate_supported_value(
        device,
        SUPPORTED_DEVICES,
        "fixed_configuration.deployment.device",
    )

    batch_size = require_integer(
        deployment["batch_size"],
        "fixed_configuration.deployment.batch_size",
    )

    if batch_size != 1:
        raise SearchSpaceValidationError(
            "The current CPU search requires "
            "'batch_size: 1'."
        )

    prediction = require_mapping(
        fixed["prediction"],
        "fixed_configuration.prediction",
    )
    validate_keys(
        prediction,
        {
            "confidence_threshold",
            "iou_threshold",
            "max_detections",
        },
        "fixed_configuration.prediction",
    )

    confidence = require_number(
        prediction["confidence_threshold"],
        (
            "fixed_configuration.prediction."
            "confidence_threshold"
        ),
    )
    iou = require_number(
        prediction["iou_threshold"],
        (
            "fixed_configuration.prediction."
            "iou_threshold"
        ),
    )
    max_detections = require_integer(
        prediction["max_detections"],
        (
            "fixed_configuration.prediction."
            "max_detections"
        ),
    )

    if not 0.0 <= confidence <= 1.0:
        raise SearchSpaceValidationError(
            "'confidence_threshold' must be "
            "between 0 and 1."
        )

    if not 0.0 <= iou <= 1.0:
        raise SearchSpaceValidationError(
            "'iou_threshold' must be between 0 and 1."
        )

    if max_detections <= 0:
        raise SearchSpaceValidationError(
            "'max_detections' must be greater than 0."
        )

    variable_dimensions = require_mapping(
        data["variable_dimensions"],
        "variable_dimensions",
    )
    validate_keys(
        variable_dimensions,
        {"image_size"},
        "variable_dimensions",
    )

    image_size_dimension = require_mapping(
        variable_dimensions["image_size"],
        "variable_dimensions.image_size",
    )
    validate_keys(
        image_size_dimension,
        {"values"},
        "variable_dimensions.image_size",
    )

    image_sizes_raw = require_list(
        image_size_dimension["values"],
        "variable_dimensions.image_size.values",
    )

    if not image_sizes_raw:
        raise SearchSpaceValidationError(
            "At least one image size is required."
        )

    image_sizes = [
        require_integer(
            value,
            "variable_dimensions.image_size.values",
        )
        for value in image_sizes_raw
    ]

    if any(value <= 0 for value in image_sizes):
        raise SearchSpaceValidationError(
            "Every image size must be greater than 0."
        )

    if any(value % 32 != 0 for value in image_sizes):
        raise SearchSpaceValidationError(
            "Every image size must be divisible by 32."
        )

    if len(image_sizes) != len(set(image_sizes)):
        raise SearchSpaceValidationError(
            "Image-size values must be unique."
        )

    if experiment_budget != len(image_sizes):
        raise SearchSpaceValidationError(
            "'experiment_budget' must equal the "
            "number of grid-search candidates."
        )

    evaluation = require_mapping(
        data["evaluation"],
        "evaluation",
    )
    validate_keys(
        evaluation,
        {
            "accuracy_metric",
            "accuracy_source",
            "validation_output_dir",
            "benchmark_config",
            "required_metrics",
        },
        "evaluation",
    )

    accuracy_metric = require_string(
        evaluation["accuracy_metric"],
        "evaluation.accuracy_metric",
    )
    validate_supported_value(
        accuracy_metric,
        SUPPORTED_ACCURACY_METRICS,
        "evaluation.accuracy_metric",
    )

    accuracy_source = require_string(
        evaluation["accuracy_source"],
        "evaluation.accuracy_source",
    )
    require_string(
        evaluation["validation_output_dir"],
        "evaluation.validation_output_dir",
    )
    validate_supported_value(
        accuracy_source,
        SUPPORTED_ACCURACY_SOURCES,
        "evaluation.accuracy_source",
    )

    require_string(
        evaluation["benchmark_config"],
        "evaluation.benchmark_config",
    )

    required_metrics = require_list(
        evaluation["required_metrics"],
        "evaluation.required_metrics",
    )

    normalized_metrics = [
        require_string(
            metric,
            "evaluation.required_metrics",
        )
        for metric in required_metrics
    ]

    if len(normalized_metrics) != len(
        set(normalized_metrics)
    ):
        raise SearchSpaceValidationError(
            "Required metrics must be unique."
        )

    if set(normalized_metrics) != REQUIRED_METRICS:
        raise SearchSpaceValidationError(
            "'evaluation.required_metrics' does not "
            "match the supported metric contract."
        )

    candidate_naming = require_mapping(
        data["candidate_naming"],
        "candidate_naming",
    )
    validate_keys(
        candidate_naming,
        {"template"},
        "candidate_naming",
    )

    template = require_string(
        candidate_naming["template"],
        "candidate_naming.template",
    )

    if "{image_size}" not in template:
        raise SearchSpaceValidationError(
            "Candidate template must contain "
            "'{image_size}'."
        )

    existing_candidates = require_list(
        data["existing_candidates"],
        "existing_candidates",
    )

    existing_ids = set()
    existing_sizes = set()

    for index, item in enumerate(existing_candidates):
        field_name = f"existing_candidates[{index}]"

        candidate = require_mapping(
            item,
            field_name,
        )
        validate_keys(
            candidate,
            {
                "candidate_id",
                "matches",
                "reuse_accuracy",
                "reuse_benchmark",
            },
            field_name,
        )

        candidate_id = require_string(
            candidate["candidate_id"],
            f"{field_name}.candidate_id",
        )

        matches = require_mapping(
            candidate["matches"],
            f"{field_name}.matches",
        )
        validate_keys(
            matches,
            {"image_size"},
            f"{field_name}.matches",
        )

        matched_size = require_integer(
            matches["image_size"],
            f"{field_name}.matches.image_size",
        )

        require_boolean(
            candidate["reuse_accuracy"],
            f"{field_name}.reuse_accuracy",
        )
        require_boolean(
            candidate["reuse_benchmark"],
            f"{field_name}.reuse_benchmark",
        )

        if matched_size not in image_sizes:
            raise SearchSpaceValidationError(
                f"{field_name} uses image size "
                f"{matched_size}, which is not in "
                "the search space."
            )

        if candidate_id in existing_ids:
            raise SearchSpaceValidationError(
                "Existing candidate IDs must be unique."
            )

        if matched_size in existing_sizes:
            raise SearchSpaceValidationError(
                "Only one existing candidate may match "
                "each image size."
            )

        existing_ids.add(candidate_id)
        existing_sizes.add(matched_size)

    generated_ids = []

    for image_size in image_sizes:
        try:
            candidate_id = template.format(
                image_size=image_size
            )
        except (KeyError, ValueError) as error:
            raise SearchSpaceValidationError(
                f"Invalid candidate template: {error}"
            ) from error

        generated_ids.append(candidate_id)

    if len(generated_ids) != len(
        set(generated_ids)
    ):
        raise SearchSpaceValidationError(
            "Candidate template produces duplicate IDs."
        )

    return data


def expand_candidates(
    search_space: Dict[str, Any],
) -> List[Dict[str, Any]]:
    image_sizes = search_space[
        "variable_dimensions"
    ]["image_size"]["values"]

    template = search_space[
        "candidate_naming"
    ]["template"]

    existing_by_size = {
        item["matches"]["image_size"]: item
        for item in search_space[
            "existing_candidates"
        ]
    }

    candidates = []

    for image_size in image_sizes:
        generated_id = template.format(
            image_size=image_size
        )

        existing = existing_by_size.get(image_size)

        if existing:
            candidates.append(
                {
                    "candidate_id": (
                        existing["candidate_id"]
                    ),
                    "generated_alias": generated_id,
                    "image_size": image_size,
                    "status": "reuse_existing",
                    "reuse_accuracy": existing[
                        "reuse_accuracy"
                    ],
                    "reuse_benchmark": existing[
                        "reuse_benchmark"
                    ],
                }
            )
        else:
            candidates.append(
                {
                    "candidate_id": generated_id,
                    "image_size": image_size,
                    "status": "pending_evaluation",
                    "reuse_accuracy": False,
                    "reuse_benchmark": False,
                }
            )

    return candidates


def validate_local_paths(
    search_space: Dict[str, Any],
    config_path: Path,
) -> None:
    project_root = config_path.resolve().parent.parent

    checkpoint = project_root / search_space[
        "fixed_configuration"
    ]["model"]["checkpoint"]

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    benchmark_config = project_root / search_space[
        "evaluation"
    ]["benchmark_config"]

    if not benchmark_config.exists():
        raise FileNotFoundError(
            "Benchmark config not found: "
            f"{benchmark_config}"
        )

    for item in search_space["existing_candidates"]:
        candidate_id = item["candidate_id"]

        candidate_path = (
            project_root
            / "results"
            / "candidates"
            / f"{candidate_id}.json"
        )

        if not candidate_path.exists():
            raise FileNotFoundError(
                "Existing candidate file not found: "
                f"{candidate_path}"
            )

        with candidate_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            candidate_data = json.load(file)

        actual_id = candidate_data.get(
            "candidate_id"
        )

        if actual_id != candidate_id:
            raise SearchSpaceValidationError(
                "Existing candidate ID does not match "
                f"its filename: {candidate_path}"
            )

        expected_size = item[
            "matches"
        ]["image_size"]

        actual_size = (
            candidate_data
            .get("benchmark", {})
            .get("image_size")
        )

        if actual_size != expected_size:
            raise SearchSpaceValidationError(
                f"Existing candidate '{candidate_id}' "
                f"uses image_size={actual_size}, "
                f"expected {expected_size}."
            )


def load_search_space(
    file_path: Union[str, Path],
    check_paths: bool = False,
) -> Dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Search-space file not found: {path}"
        )

    if path.suffix.lower() not in {
        ".yaml",
        ".yml",
        ".json",
    }:
        raise SearchSpaceValidationError(
            "Search-space file must use "
            ".yaml, .yml, or .json."
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise SearchSpaceValidationError(
            f"Invalid YAML or JSON syntax: {error}"
        ) from error

    root_data = require_mapping(
        raw_data,
        "root",
    )

    search_space = parse_search_space_data(
        root_data
    )

    if check_paths:
        validate_local_paths(
            search_space,
            path,
        )

    return search_space


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Validate and expand an "
            "EdgeNAS-Lite search space."
        )
    )

    argument_parser.add_argument(
        "search_space_file",
        help="Path to the search-space YAML or JSON.",
    )

    argument_parser.add_argument(
        "--check-paths",
        action="store_true",
        help=(
            "Verify local checkpoint, benchmark "
            "config, and existing candidate files."
        ),
    )

    arguments = argument_parser.parse_args()

    try:
        search_space = load_search_space(
            arguments.search_space_file,
            check_paths=arguments.check_paths,
        )

        candidates = expand_candidates(
            search_space
        )

    except (
        FileNotFoundError,
        SearchSpaceValidationError,
        json.JSONDecodeError,
    ) as error:
        argument_parser.error(str(error))

    output = {
        "schema_version": (
            search_space["schema_version"]
        ),
        "search_space_id": (
            search_space["search_space_id"]
        ),
        "search_type": (
            search_space["search"]["type"]
        ),
        "strategy": (
            search_space["search"]["strategy"]
        ),
        "experiment_budget": (
            search_space[
                "search"
            ]["experiment_budget"]
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()