from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.candidate_selector.selector import select_candidate
from src.constraint_checker.checker import evaluate_candidate
from src.knowledge_database.loader import load_knowledge_base
from src.knowledge_database.query import query_candidates
from src.proposal.compatibility import check_candidate_compatibility
from src.requirement_parser.parser import (
    load_requirement,
    parse_requirement_data,
)


def retrieve_candidates_for_requirement(
    requirement_file: Union[str, Path],
    *,
    index_file: Union[str, Path] = "knowledge/index.yaml",
    project_root: Union[str, Path] = ".",
    model_family: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(project_root).resolve()

    requirement_path = Path(requirement_file)
    if not requirement_path.is_absolute():
        requirement_path = root / requirement_path

    requirement = load_requirement(requirement_path)

    knowledge = load_knowledge_base(
        index_file=index_file,
        project_root=root,
    )

    candidates = query_candidates(
        knowledge,
        dataset=requirement.target.dataset,
        device=requirement.target.device,
        model_family=model_family,
    )

    return {
        "request_id": requirement.request_id,
        "requirement": requirement.to_dict(),
        "knowledge_base_id": knowledge["knowledge_base_id"],
        "retrieval_filters": {
            "dataset": requirement.target.dataset,
            "device": requirement.target.device,
            "model_family": model_family,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def evaluate_retrieved_candidates(
    retrieval: Dict[str, Any],
    *,
    expected_hardware_id: Optional[str] = None,
) -> Dict[str, Any]:
    requirement = parse_requirement_data(retrieval["requirement"])

    compatibility_reports = []
    compatible_ids = []
    excluded_ids = []

    evaluations = []
    feasible_ids = []
    rejected_ids = []

    for item in retrieval["candidates"]:
        record = item["record"]

        report = check_candidate_compatibility(
            record,
            expected_hardware_id=expected_hardware_id,
        )

        compatibility_reports.append({
            **report,
            "record_path": item["record_path"],
        })

        candidate_id = record["candidate_id"]

        # Only compatible records may enter constraint evaluation.
        if report["status"] != "compatible":
            excluded_ids.append(candidate_id)
            continue

        compatible_ids.append(candidate_id)

        evaluation = evaluate_candidate(
            requirement=requirement,
            candidate=record,
        )
        evaluations.append(evaluation)

        if evaluation["constraints_satisfied"]:
            feasible_ids.append(candidate_id)
        else:
            rejected_ids.append(candidate_id)

    return {
        **retrieval,
        "expected_hardware_id": expected_hardware_id,
        "compatibility_reports": compatibility_reports,
        "compatible_candidate_ids": compatible_ids,
        "excluded_candidate_ids": excluded_ids,
        "evaluations": evaluations,
        "feasible_candidate_ids": feasible_ids,
        "rejected_candidate_ids": rejected_ids,
    }


def propose_from_knowledge(
    requirement_file: Union[str, Path],
    *,
    index_file: Union[str, Path] = "knowledge/index.yaml",
    project_root: Union[str, Path] = ".",
    model_family: Optional[str] = None,
    expected_hardware_id: Optional[str] = None,
) -> Dict[str, Any]:
    retrieval = retrieve_candidates_for_requirement(
        requirement_file,
        index_file=index_file,
        project_root=project_root,
        model_family=model_family,
    )

    evaluated = evaluate_retrieved_candidates(
        retrieval,
        expected_hardware_id=expected_hardware_id,
    )

    if not evaluated["candidates"]:
        return {
            **evaluated,
            "proposal_status": "no_matching_candidates",
            "selection": None,
            "selected_source": None,
        }

    if not evaluated["compatible_candidate_ids"]:
        return {
            **evaluated,
            "proposal_status": "no_compatible_candidates",
            "selection": None,
            "selected_source": None,
        }

    requirement = parse_requirement_data(evaluated["requirement"])

    selection = select_candidate(
        requirement=requirement,
        evaluations=evaluated["evaluations"],
    )

    selected_id = selection["selected_candidate_id"]
    selected_source = None

    if selected_id is not None:
        selected_item = next(
            item
            for item in evaluated["candidates"]
            if item["record"]["candidate_id"] == selected_id
        )

        selected_source = {
            "candidate_id": selected_id,
            "record_path": selected_item["record_path"],
        }

    return {
        **evaluated,
        "proposal_status": selection["selection_status"],
        "selection": selection,
        "selected_source": selected_source,
    }