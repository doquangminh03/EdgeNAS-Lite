import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.candidate_runner.runner import (
    CandidateRunnerError,
    candidate_record_path,
    project_root_from_config,
    run_candidates,
)
from src.candidate_selector.selector import (
    CandidateSelectorError,
    save_selection,
    select_candidate,
)
from src.constraint_checker.checker import (
    CandidateValidationError,
    evaluate_candidate,
    load_candidate,
    save_evaluation,
)
from src.requirement_parser.parser import (
    RequirementValidationError,
    load_requirement,
)
from src.search_space.parser import (
    SearchSpaceValidationError,
    expand_candidates,
    load_search_space,
)


class SearchControllerError(ValueError):
    """Raised when a complete search cannot be executed safely."""


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def save_json(data: Dict[str, Any], output_file: Union[str, Path]) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def validate_requirement_matches_search_space(
    requirement: Any,
    search_space: Dict[str, Any],
) -> None:
    fixed = search_space["fixed_configuration"]
    expected = {
        "task": fixed["task"],
        "device": fixed["deployment"]["device"],
        "dataset": fixed["dataset"]["name"],
    }
    actual = {
        "task": requirement.target.task,
        "device": requirement.target.device,
        "dataset": requirement.target.dataset,
    }

    if actual != expected:
        raise SearchControllerError(
            "Requirement target does not match the search space: "
            f"expected {expected}, got {actual}."
        )


def validate_candidate_matches_search_space(
    candidate_record: Dict[str, Any],
    candidate_spec: Dict[str, Any],
    search_space: Dict[str, Any],
) -> None:
    expected_id = candidate_spec["candidate_id"]
    if candidate_record.get("candidate_id") != expected_id:
        raise SearchControllerError(
            f"Candidate record ID does not match '{expected_id}'."
        )

    model = candidate_record.get("model")
    if not isinstance(model, dict):
        raise SearchControllerError(
            f"Candidate '{expected_id}' has no model metadata."
        )

    expected_checkpoint = search_space[
        "fixed_configuration"
    ]["model"]["checkpoint"]
    if model.get("checkpoint") != expected_checkpoint:
        raise SearchControllerError(
            f"Candidate '{expected_id}' uses a different checkpoint."
        )

    benchmark = candidate_record.get("benchmark")
    if not isinstance(benchmark, dict):
        raise SearchControllerError(
            f"Candidate '{expected_id}' has no benchmark metadata."
        )

    expected_image_size = candidate_spec["image_size"]
    if benchmark.get("image_size") != expected_image_size:
        raise SearchControllerError(
            f"Candidate '{expected_id}' benchmark image size does not "
            f"match {expected_image_size}."
        )

    if benchmark.get("protocol_status") != "standardized":
        raise SearchControllerError(
            f"Candidate '{expected_id}' does not use a standardized "
            "benchmark."
        )

    accuracy = candidate_record.get("accuracy")
    if not isinstance(accuracy, dict):
        raise SearchControllerError(
            f"Candidate '{expected_id}' has no accuracy metadata."
        )

    recorded_accuracy_size = accuracy.get("image_size")
    if (
        recorded_accuracy_size is not None
        and recorded_accuracy_size != expected_image_size
    ):
        raise SearchControllerError(
            f"Candidate '{expected_id}' accuracy image size does not "
            f"match {expected_image_size}."
        )


def build_candidate_plan(
    candidates: List[Dict[str, Any]],
    project_root: Path,
    overwrite_candidates: bool,
) -> List[Dict[str, Any]]:
    plan = []

    for candidate in candidates:
        record_path = candidate_record_path(
            project_root,
            candidate["candidate_id"],
        )

        if record_path.exists():
            if (
                overwrite_candidates
                and candidate["status"] != "reuse_existing"
            ):
                action = "regenerate"
            else:
                action = "reuse_record"
        elif candidate["status"] == "reuse_existing":
            raise SearchControllerError(
                "Reusable candidate record is missing: "
                f"{record_path}"
            )
        else:
            action = "generate"

        plan.append(
            {
                "candidate_id": candidate["candidate_id"],
                "image_size": candidate["image_size"],
                "action": action,
                "candidate_record": _relative_path(
                    record_path,
                    project_root,
                ),
            }
        )

    return plan


def _candidate_spec_by_id(
    candidates: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        candidate["candidate_id"]: candidate
        for candidate in candidates
    }


def run_search(
    requirement_file: Union[str, Path],
    search_space_file: Union[str, Path],
    dry_run: bool = False,
    overwrite_candidates: bool = False,
    output_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    requirement_path = Path(requirement_file)
    search_space_path = Path(search_space_file)
    project_root = project_root_from_config(search_space_path)

    requirement = load_requirement(requirement_path)
    search_space = load_search_space(
        search_space_path,
        check_paths=True,
    )
    validate_requirement_matches_search_space(
        requirement,
        search_space,
    )

    candidates = expand_candidates(search_space)
    candidate_specs = _candidate_spec_by_id(candidates)
    candidate_plan = build_candidate_plan(
        candidates,
        project_root,
        overwrite_candidates,
    )

    run_id = (
        f"{requirement.request_id}__"
        f"{search_space['search_space_id']}"
    )

    if dry_run:
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "request_id": requirement.request_id,
            "search_space_id": search_space["search_space_id"],
            "dry_run": True,
            "candidate_plan": candidate_plan,
        }

    evaluations = []
    evaluation_files = []

    for plan_item in candidate_plan:
        candidate_id = plan_item["candidate_id"]
        action = plan_item["action"]
        record_path = candidate_record_path(
            project_root,
            candidate_id,
        )

        if action in {"generate", "regenerate"}:
            run_candidates(
                search_space_path,
                candidate_id,
                False,
                action == "regenerate",
            )

        if not record_path.exists():
            raise SearchControllerError(
                "Candidate Runner did not create the expected record: "
                f"{record_path}"
            )

        candidate_record = load_candidate(record_path)
        validate_candidate_matches_search_space(
            candidate_record,
            candidate_specs[candidate_id],
            search_space,
        )

        evaluation = evaluate_candidate(
            requirement,
            candidate_record,
        )
        evaluation_path = (
            project_root
            / "results"
            / "evaluations"
            / f"{requirement.request_id}__{candidate_id}.json"
        )
        save_evaluation(evaluation, evaluation_path)
        evaluations.append(evaluation)
        evaluation_files.append(
            _relative_path(evaluation_path, project_root)
        )

    selection = select_candidate(requirement, evaluations)
    selection_path = (
        project_root
        / "results"
        / "selections"
        / f"{requirement.request_id}.json"
    )
    save_selection(selection, selection_path)

    result = {
        "schema_version": "1.0",
        "run_id": run_id,
        "request_id": requirement.request_id,
        "search_space_id": search_space["search_space_id"],
        "dry_run": False,
        "candidate_plan": candidate_plan,
        "evaluation_files": evaluation_files,
        "selection_file": _relative_path(
            selection_path,
            project_root,
        ),
        "selection_status": selection["selection_status"],
        "selected_candidate_id": selection[
            "selected_candidate_id"
        ],
    }

    if output_file is None:
        output_path = (
            project_root
            / "results"
            / "search_runs"
            / f"{run_id}.json"
        )
    else:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = project_root / output_path

    save_json(result, output_path)
    result["run_file"] = _relative_path(output_path, project_root)
    return result


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Run the complete EdgeNAS-Lite candidate search, "
            "evaluation, and selection pipeline."
        )
    )
    argument_parser.add_argument(
        "requirement_file",
        help="Path to the requirement YAML or JSON file.",
    )
    argument_parser.add_argument(
        "search_space_file",
        help="Path to the search-space YAML or JSON file.",
    )
    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without running or writing results.",
    )
    argument_parser.add_argument(
        "--overwrite-candidates",
        action="store_true",
        help=(
            "Regenerate non-reusable candidate records even when "
            "they already exist."
        ),
    )
    argument_parser.add_argument(
        "--output",
        help="Optional path for the search-run JSON manifest.",
    )
    arguments = argument_parser.parse_args()

    try:
        result = run_search(
            arguments.requirement_file,
            arguments.search_space_file,
            dry_run=arguments.dry_run,
            overwrite_candidates=arguments.overwrite_candidates,
            output_file=arguments.output,
        )
    except (
        CandidateRunnerError,
        CandidateSelectorError,
        CandidateValidationError,
        FileNotFoundError,
        RequirementValidationError,
        SearchControllerError,
        SearchSpaceValidationError,
        json.JSONDecodeError,
    ) as error:
        argument_parser.error(str(error))

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

