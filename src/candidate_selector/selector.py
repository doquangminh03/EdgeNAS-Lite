import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
SUPPORTED_OPTIMIZATION_GOALS = {
    "accuracy",
    "latency",
    "model_size",
    "balanced",
}

CHECK_SPECS = {
    "minimum_map50_95": {
        "metric_path": "accuracy.map50_95",
        "operator": ">=",
        "metric_name": "map50_95",
    },
    "maximum_median_latency_ms": {
        "metric_path": "benchmark.median_latency_ms",
        "operator": "<=",
        "metric_name": "median_latency_ms",
    },
    "maximum_model_size_mb": {
        "metric_path": "model.model_size_mb",
        "operator": "<=",
        "metric_name": "model_size_mb",
    },
}


class CandidateSelectorError(ValueError):
    """Raised when evaluation inputs cannot be selected safely."""


def require_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise CandidateSelectorError(
            f"'{field_name}' must be a mapping/object."
        )
    return value


def require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateSelectorError(
            f"'{field_name}' must be a non-empty string."
        )
    return value.strip()


def require_boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise CandidateSelectorError(
            f"'{field_name}' must be true or false."
        )
    return value


def require_finite_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise CandidateSelectorError(
            f"'{field_name}' must be a finite number."
        )
    return float(value)


def load_evaluation(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

    if path.suffix.lower() != ".json":
        raise CandidateSelectorError(
            "Evaluation file must use the .json extension."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise CandidateSelectorError(
            f"Invalid evaluation JSON syntax: {error}"
        ) from error

    return require_mapping(data, "evaluation root")


def load_request_evaluations(
    evaluations_directory: Union[str, Path],
    request_id: str,
) -> List[Dict[str, Any]]:
    directory = Path(evaluations_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Evaluations directory not found: {directory}"
        )

    if not directory.is_dir():
        raise CandidateSelectorError(
            f"Evaluations path must be a directory: {directory}"
        )

    paths = sorted(directory.glob(f"{request_id}__*.json"))

    if not paths:
        raise CandidateSelectorError(
            f"No evaluations found for request '{request_id}'."
        )

    return [load_evaluation(path) for path in paths]


def _requirement_constraint_values(requirement: Any) -> Dict[str, float]:
    constraints = requirement.constraints
    return {
        "minimum_map50_95": float(
            constraints.minimum_map50_95
        ),
        "maximum_median_latency_ms": float(
            constraints.maximum_median_latency_ms
        ),
        "maximum_model_size_mb": float(
            constraints.maximum_model_size_mb
        ),
    }


def _expected_passed(operator: str, actual: float, required: float) -> bool:
    if operator == ">=":
        return actual >= required
    if operator == "<=":
        return actual <= required
    raise CandidateSelectorError(
        f"Unsupported evaluation operator: {operator}"
    )


def validate_evaluation(
    evaluation: Dict[str, Any],
    requirement: Any,
) -> Dict[str, Any]:
    required_root_fields = {
        "schema_version",
        "evaluation_id",
        "request_id",
        "candidate_id",
        "target",
        "checks",
        "constraints_satisfied",
    }
    missing_root_fields = required_root_fields - set(evaluation)
    if missing_root_fields:
        raise CandidateSelectorError(
            "Evaluation is missing field(s): "
            + ", ".join(sorted(missing_root_fields))
        )

    schema_version = require_non_empty_string(
        evaluation["schema_version"],
        "schema_version",
    )
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise CandidateSelectorError(
            f"Unsupported evaluation schema_version: {schema_version}"
        )

    request_id = require_non_empty_string(
        evaluation["request_id"],
        "request_id",
    )
    if request_id != requirement.request_id:
        raise CandidateSelectorError(
            "Evaluation request mismatch: expected "
            f"'{requirement.request_id}', got '{request_id}'."
        )

    candidate_id = require_non_empty_string(
        evaluation["candidate_id"],
        "candidate_id",
    )
    expected_evaluation_id = f"{request_id}__{candidate_id}"
    evaluation_id = require_non_empty_string(
        evaluation["evaluation_id"],
        "evaluation_id",
    )
    if evaluation_id != expected_evaluation_id:
        raise CandidateSelectorError(
            "Evaluation ID does not match its request and candidate."
        )

    target = require_mapping(evaluation["target"], "target")
    expected_target = {
        "task": requirement.target.task,
        "device": requirement.target.device,
        "dataset": requirement.target.dataset,
    }
    if target != expected_target:
        raise CandidateSelectorError(
            f"Evaluation target mismatch for '{candidate_id}'."
        )

    checks = require_mapping(evaluation["checks"], "checks")
    if set(checks) != set(CHECK_SPECS):
        raise CandidateSelectorError(
            f"Evaluation checks do not match the supported contract for '{candidate_id}'."
        )

    required_values = _requirement_constraint_values(requirement)
    metrics: Dict[str, float] = {}
    failed_constraints = []
    recomputed_results = []

    for constraint_name, spec in CHECK_SPECS.items():
        check = require_mapping(
            checks[constraint_name],
            f"checks.{constraint_name}",
        )

        metric_path = require_non_empty_string(
            check.get("metric_path"),
            f"checks.{constraint_name}.metric_path",
        )
        operator = require_non_empty_string(
            check.get("operator"),
            f"checks.{constraint_name}.operator",
        )
        required = require_finite_number(
            check.get("required"),
            f"checks.{constraint_name}.required",
        )
        actual = require_finite_number(
            check.get("actual"),
            f"checks.{constraint_name}.actual",
        )
        stored_passed = require_boolean(
            check.get("passed"),
            f"checks.{constraint_name}.passed",
        )

        if metric_path != spec["metric_path"]:
            raise CandidateSelectorError(
                f"Unexpected metric path for '{constraint_name}'."
            )
        if operator != spec["operator"]:
            raise CandidateSelectorError(
                f"Unexpected operator for '{constraint_name}'."
            )
        if not math.isclose(
            required,
            required_values[constraint_name],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise CandidateSelectorError(
                f"Evaluation for '{candidate_id}' is stale: "
                f"'{constraint_name}' does not match the current requirement."
            )

        recomputed_passed = _expected_passed(
            operator,
            actual,
            required,
        )
        if stored_passed != recomputed_passed:
            raise CandidateSelectorError(
                f"Stored pass/fail value is inconsistent for '{candidate_id}' "
                f"and constraint '{constraint_name}'."
            )

        metrics[spec["metric_name"]] = actual
        recomputed_results.append(recomputed_passed)
        if not recomputed_passed:
            failed_constraints.append(constraint_name)

    if not 0.0 <= metrics["map50_95"] <= 1.0:
        raise CandidateSelectorError(
            f"Evaluation accuracy is outside [0, 1] for '{candidate_id}'."
        )
    if metrics["median_latency_ms"] <= 0.0:
        raise CandidateSelectorError(
            f"Evaluation latency must be greater than 0 for '{candidate_id}'."
        )
    if metrics["model_size_mb"] <= 0.0:
        raise CandidateSelectorError(
            f"Evaluation model size must be greater than 0 for '{candidate_id}'."
        )

    stored_satisfied = require_boolean(
        evaluation["constraints_satisfied"],
        "constraints_satisfied",
    )
    recomputed_satisfied = all(recomputed_results)
    if stored_satisfied != recomputed_satisfied:
        raise CandidateSelectorError(
            f"Overall constraint result is inconsistent for '{candidate_id}'."
        )

    return {
        "candidate_id": candidate_id,
        "constraints_satisfied": recomputed_satisfied,
        "metrics": metrics,
        "failed_constraints": failed_constraints,
    }


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_components(
    metrics: Dict[str, float],
    requirement: Any,
) -> Dict[str, float]:
    constraints = requirement.constraints
    minimum_accuracy = float(constraints.minimum_map50_95)
    maximum_latency = float(constraints.maximum_median_latency_ms)
    maximum_size = float(constraints.maximum_model_size_mb)

    if minimum_accuracy >= 1.0:
        accuracy_headroom = 1.0
    else:
        accuracy_headroom = (
            metrics["map50_95"] - minimum_accuracy
        ) / (1.0 - minimum_accuracy)

    latency_headroom = (
        maximum_latency - metrics["median_latency_ms"]
    ) / maximum_latency
    size_headroom = (
        maximum_size - metrics["model_size_mb"]
    ) / maximum_size

    return {
        "accuracy_headroom": round(
            _clamp_unit(accuracy_headroom), 6
        ),
        "latency_headroom": round(
            _clamp_unit(latency_headroom), 6
        ),
        "model_size_headroom": round(
            _clamp_unit(size_headroom), 6
        ),
    }


def candidate_score(
    metrics: Dict[str, float],
    components: Dict[str, float],
    optimization_goal: str,
) -> float:
    if optimization_goal == "accuracy":
        return metrics["map50_95"]
    if optimization_goal == "latency":
        return components["latency_headroom"]
    if optimization_goal == "model_size":
        return components["model_size_headroom"]
    if optimization_goal == "balanced":
        return sum(components.values()) / len(components)
    raise CandidateSelectorError(
        f"Unsupported optimization goal: {optimization_goal}"
    )


def _ranking_key(
    item: Dict[str, Any],
    optimization_goal: str,
) -> Tuple[Any, ...]:
    metrics = item["metrics"]

    if optimization_goal == "accuracy":
        return (
            -item["score"],
            metrics["median_latency_ms"],
            metrics["model_size_mb"],
            item["candidate_id"],
        )
    if optimization_goal == "latency":
        return (
            -item["score"],
            -metrics["map50_95"],
            metrics["model_size_mb"],
            item["candidate_id"],
        )
    if optimization_goal == "model_size":
        return (
            -item["score"],
            -metrics["map50_95"],
            metrics["median_latency_ms"],
            item["candidate_id"],
        )
    return (
        -item["score"],
        -metrics["map50_95"],
        metrics["median_latency_ms"],
        metrics["model_size_mb"],
        item["candidate_id"],
    )


def rank_feasible_candidates(
    feasible_candidates: List[Dict[str, Any]],
    requirement: Any,
) -> List[Dict[str, Any]]:
    optimization_goal = requirement.preferences.optimization_goal

    if optimization_goal not in SUPPORTED_OPTIMIZATION_GOALS:
        raise CandidateSelectorError(
            f"Unsupported optimization goal: {optimization_goal}"
        )

    scored = []
    for candidate in feasible_candidates:
        components = score_components(
            candidate["metrics"], requirement
        )
        score = candidate_score(
            candidate["metrics"],
            components,
            optimization_goal,
        )
        scored.append(
            {
                "candidate_id": candidate["candidate_id"],
                "score": round(score, 6),
                "score_components": components,
                "metrics": candidate["metrics"],
            }
        )

    ranked = sorted(
        scored,
        key=lambda item: _ranking_key(
            item, optimization_goal
        ),
    )

    return [
        {"rank": index, **item}
        for index, item in enumerate(ranked, start=1)
    ]


def select_candidate(
    requirement: Any,
    evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not evaluations:
        raise CandidateSelectorError(
            "At least one evaluation is required."
        )

    validated = [
        validate_evaluation(evaluation, requirement)
        for evaluation in evaluations
    ]
    candidate_ids = [item["candidate_id"] for item in validated]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CandidateSelectorError(
            "Evaluation candidate IDs must be unique."
        )

    feasible = [
        item
        for item in validated
        if item["constraints_satisfied"]
    ]
    rejected = [
        {
            "candidate_id": item["candidate_id"],
            "failed_constraints": item["failed_constraints"],
            "metrics": item["metrics"],
        }
        for item in validated
        if not item["constraints_satisfied"]
    ]
    optimization_goal = requirement.preferences.optimization_goal

    base_result = {
        "schema_version": "1.0",
        "selection_id": f"{requirement.request_id}__selection",
        "request_id": requirement.request_id,
        "target": {
            "task": requirement.target.task,
            "device": requirement.target.device,
            "dataset": requirement.target.dataset,
        },
        "optimization_goal": optimization_goal,
        "evaluated_candidate_count": len(validated),
        "feasible_candidate_count": len(feasible),
        "rejected_candidates": sorted(
            rejected,
            key=lambda item: item["candidate_id"],
        ),
    }

    if not feasible:
        return {
            **base_result,
            "selection_status": "no_feasible_candidate",
            "selection_method": None,
            "selected_candidate_id": None,
            "selected_metrics": None,
            "reason": "No candidate satisfies all hard constraints.",
            "ranking": [],
        }

    ranking = rank_feasible_candidates(feasible, requirement)
    selected = ranking[0]

    if len(feasible) == 1:
        selection_method = "only_feasible_candidate"
        reason = "Only one candidate satisfies all hard constraints."
    else:
        selection_method = f"optimization_goal_{optimization_goal}"
        reason = (
            "Selected the highest-ranked feasible candidate "
            f"for optimization goal '{optimization_goal}'."
        )

    return {
        **base_result,
        "selection_status": "selected",
        "selection_method": selection_method,
        "selected_candidate_id": selected["candidate_id"],
        "selected_metrics": selected["metrics"],
        "reason": reason,
        "ranking": ranking,
    }


def save_selection(
    selection: Dict[str, Any],
    output_file: Union[str, Path],
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            selection,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Select the best feasible EdgeNAS-Lite candidate "
            "from request-specific evaluations."
        )
    )
    argument_parser.add_argument(
        "request_file",
        help="Path to the requirement YAML or JSON file.",
    )
    argument_parser.add_argument(
        "evaluations_directory",
        help="Directory containing request-candidate evaluation JSON files.",
    )
    argument_parser.add_argument(
        "--output",
        help="Optional output path for the selection JSON file.",
    )
    arguments = argument_parser.parse_args()

    try:
        from src.requirement_parser.parser import (
            RequirementValidationError,
            load_requirement,
        )

        requirement = load_requirement(arguments.request_file)
        evaluations = load_request_evaluations(
            arguments.evaluations_directory,
            requirement.request_id,
        )
        selection = select_candidate(requirement, evaluations)

        if arguments.output:
            save_selection(selection, arguments.output)

    except (
        FileNotFoundError,
        CandidateSelectorError,
        RequirementValidationError,
    ) as error:
        argument_parser.error(str(error))

    print(
        json.dumps(
            selection,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
