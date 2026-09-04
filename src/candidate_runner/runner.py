import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from src.benchmarking.cpu_benchmark import (
    load_config as load_benchmark_config,
    summarize_latencies,
)
from src.search_space.parser import (
    expand_candidates,
    load_search_space,
)


TOTAL_BENCHMARK_SESSIONS = 5
STABILIZATION_SESSIONS = 2
POOLED_BENCHMARK_SESSIONS = 3


class CandidateRunnerError(ValueError):
    """Raised when a candidate cannot be evaluated safely."""


def load_json(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise CandidateRunnerError(
            f"JSON root must be an object: {file_path}"
        )

    return data


def save_json(
    data: Dict[str, Any],
    file_path: Path,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")


def project_root_from_config(
    search_space_path: Path,
) -> Path:
    return search_space_path.resolve().parent.parent


def candidate_record_path(
    project_root: Path,
    candidate_id: str,
) -> Path:
    return (
        project_root
        / "results"
        / "candidates"
        / f"{candidate_id}.json"
    )


def benchmark_result_paths(
    project_root: Path,
    candidate_id: str,
) -> List[Path]:
    return [
        (
            project_root
            / "results"
            / "benchmarks"
            / f"{candidate_id}_cpu_run{session}.json"
        )
        for session in range(
            1,
            TOTAL_BENCHMARK_SESSIONS + 1,
        )
    ]


def select_candidates(
    candidates: List[Dict[str, Any]],
    requested_candidate_id: Optional[str],
) -> List[Dict[str, Any]]:
    if requested_candidate_id is None:
        return candidates

    selected = [
        candidate
        for candidate in candidates
        if candidate["candidate_id"]
        == requested_candidate_id
    ]

    if not selected:
        raise CandidateRunnerError(
            "Candidate is not in the search space: "
            f"{requested_candidate_id}"
        )

    return selected


def build_validation_kwargs(
    search_space: Dict[str, Any],
    candidate: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    fixed = search_space["fixed_configuration"]
    dataset = fixed["dataset"]
    deployment = fixed["deployment"]
    prediction = fixed["prediction"]
    evaluation = search_space["evaluation"]

    validation_project = Path(
        evaluation["validation_output_dir"]
    )
    if not validation_project.is_absolute():
        validation_project = (
            project_root / validation_project
        )

    # Do not pass prediction.confidence_threshold here. AP validation
    # must retain Ultralytics' low validation confidence threshold.
    return {
        "data": dataset["data_config"],
        "split": dataset["validation_split"],
        "imgsz": candidate["image_size"],
        "batch": deployment["batch_size"],
        "device": deployment["device"],
        "classes": dataset["class_ids"],
        "iou": prediction["iou_threshold"],
        "max_det": prediction["max_detections"],
        "project": str(validation_project),
        "name": candidate["candidate_id"],
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }


def _class_result_positions(
    box_metrics: Any,
    class_ids: Sequence[int],
) -> Dict[int, int]:
    raw_indices = getattr(
        box_metrics,
        "ap_class_index",
        None,
    )

    if raw_indices is None:
        return {
            class_id: position
            for position, class_id in enumerate(class_ids)
        }

    indices = [int(index) for index in raw_indices]

    return {
        class_id: indices.index(class_id)
        for class_id in class_ids
        if class_id in indices
    }


def _count_validation_instances(
    metrics: Any,
    class_ids: Sequence[int],
) -> Optional[int]:
    counts = getattr(metrics, "nt_per_class", None)

    if counts is None:
        counts = getattr(
            metrics.box,
            "nt_per_class",
            None,
        )

    if counts is None:
        return None

    normalized_counts = [int(value) for value in counts]

    if not normalized_counts:
        return 0

    if max(class_ids) < len(normalized_counts):
        return sum(
            normalized_counts[class_id]
            for class_id in class_ids
        )

    if len(normalized_counts) == len(class_ids):
        return sum(normalized_counts)

    return None


def extract_accuracy_result(
    metrics: Any,
    search_space: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    dataset = search_space[
        "fixed_configuration"
    ]["dataset"]
    class_ids = dataset["class_ids"]
    class_names = dataset["classes"]
    box_metrics = getattr(metrics, "box", None)

    if box_metrics is None:
        raise CandidateRunnerError(
            "Validation did not return detection metrics."
        )

    positions = _class_result_positions(
        box_metrics,
        class_ids,
    )

    missing_ids = [
        class_id
        for class_id in class_ids
        if class_id not in positions
    ]
    if missing_ids:
        raise CandidateRunnerError(
            "Validation metrics are missing class IDs: "
            + ", ".join(str(value) for value in missing_ids)
        )

    per_class = {}

    for class_id, class_name in zip(
        class_ids,
        class_names,
    ):
        position = positions[class_id]
        class_result = box_metrics.class_result(position)

        if len(class_result) < 4:
            raise CandidateRunnerError(
                "Unexpected per-class metric format for "
                f"class ID {class_id}."
            )

        precision, recall, map50, map50_95 = (
            class_result[:4]
        )

        per_class[class_name] = {
            "class_id": class_id,
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "map50": round(float(map50), 6),
            "map50_95": round(float(map50_95), 6),
        }

    return {
        "dataset": dataset["name"],
        "split": dataset["validation_split"],
        "image_size": candidate["image_size"],
        "validation_images": dataset[
            "validation_images"
        ],
        "validation_instances": (
            _count_validation_instances(
                metrics,
                class_ids,
            )
        ),
        "precision": round(float(box_metrics.mp), 6),
        "recall": round(float(box_metrics.mr), 6),
        "map50": round(float(box_metrics.map50), 6),
        "map50_95": round(float(box_metrics.map), 6),
        "per_class": per_class,
    }


def run_validation(
    search_space: Dict[str, Any],
    candidate: Dict[str, Any],
    project_root: Path,
) -> Dict[str, Any]:
    checkpoint = Path(
        search_space["fixed_configuration"]
        ["model"]["checkpoint"]
    )
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise CandidateRunnerError(
            "Ultralytics is required to run validation."
        ) from error

    print(
        "Running full validation: "
        f"{candidate['candidate_id']} "
        f"(imgsz={candidate['image_size']})"
    )

    model = YOLO(str(checkpoint))
    metrics = model.val(
        **build_validation_kwargs(
            search_space,
            candidate,
            project_root,
        )
    )

    return extract_accuracy_result(
        metrics,
        search_space,
        candidate,
    )


def build_session_benchmark_config(
    base_config: Dict[str, Any],
    search_space: Dict[str, Any],
    candidate: Dict[str, Any],
    session_number: int,
    result_path: Path,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    checkpoint = search_space[
        "fixed_configuration"
    ]["model"]["checkpoint"]

    config["benchmark_id"] = (
        f"{candidate['candidate_id']}_cpu_v1_"
        f"run{session_number}"
    )
    config["candidate_id"] = candidate["candidate_id"]
    config["model"]["checkpoint"] = checkpoint
    config["protocol"]["image_size"] = (
        candidate["image_size"]
    )
    config["output"]["result_file"] = str(result_path)

    return config


def _relative_path_string(
    file_path: Path,
    project_root: Path,
) -> str:
    try:
        return str(file_path.relative_to(project_root))
    except ValueError:
        return str(file_path)


def validate_benchmark_sessions(
    results: List[Dict[str, Any]],
    candidate_id: str,
    image_size: int,
) -> None:
    if len(results) != POOLED_BENCHMARK_SESSIONS:
        raise CandidateRunnerError(
            "Exactly three measured benchmark sessions "
            "are required for pooling."
        )

    reference_protocol = results[0].get("protocol")
    reference_images = results[0].get(
        "selected_image_files"
    )

    if not isinstance(reference_protocol, dict):
        raise CandidateRunnerError(
            "Benchmark result is missing its protocol."
        )

    protocol_keys = {
        "device",
        "image_size",
        "batch_size",
        "warmup_runs",
        "repetitions_per_image",
        "confidence_threshold",
        "iou_threshold",
        "max_detections",
        "timing_scope",
        "disk_io_included",
    }

    reference_values = {
        key: reference_protocol.get(key)
        for key in protocol_keys
    }

    for result in results:
        if result.get("candidate_id") != candidate_id:
            raise CandidateRunnerError(
                "Benchmark candidate ID does not match "
                f"'{candidate_id}'."
            )

        if result.get("protocol_status") != "standardized":
            raise CandidateRunnerError(
                "Only standardized benchmark results "
                "can be pooled."
            )

        protocol = result.get("protocol", {})
        if protocol.get("image_size") != image_size:
            raise CandidateRunnerError(
                "Benchmark image size does not match "
                f"{image_size}."
            )

        values = {
            key: protocol.get(key)
            for key in protocol_keys
        }
        if values != reference_values:
            raise CandidateRunnerError(
                "Benchmark sessions use different protocols."
            )

        if result.get("selected_image_files") != reference_images:
            raise CandidateRunnerError(
                "Benchmark sessions selected different images."
            )

        samples = result.get("raw_samples")
        if not isinstance(samples, list) or not samples:
            raise CandidateRunnerError(
                "Benchmark result has no raw latency samples."
            )

        reported_count = result.get(
            "summary",
            {},
        ).get("sample_count")
        if reported_count != len(samples):
            raise CandidateRunnerError(
                "Benchmark raw sample count does not match "
                "its summary."
            )


def pool_benchmark_results(
    included_results: List[Dict[str, Any]],
    included_paths: List[Path],
    excluded_paths: List[Path],
    candidate_id: str,
    image_size: int,
    project_root: Path,
) -> Dict[str, Any]:
    validate_benchmark_sessions(
        included_results,
        candidate_id,
        image_size,
    )

    pooled_samples = [
        sample
        for result in included_results
        for sample in result["raw_samples"]
    ]
    summary = summarize_latencies(pooled_samples)
    first_result = included_results[0]
    protocol = first_result["protocol"]
    dataset = first_result["dataset"]
    environment = first_result["environment"]

    return {
        "device": protocol["device"],
        "system": environment["system"],
        "architecture": environment["architecture"],
        "image_size": image_size,
        "batch_size": protocol["batch_size"],
        "dataset": dataset["name"],
        "split": dataset["split"],
        "selected_images": dataset["selected_images"],
        "selection_seed": dataset["selection_seed"],
        "session_count": len(included_results),
        "warmup_runs_per_session": protocol["warmup_runs"],
        "repetitions_per_image": protocol[
            "repetitions_per_image"
        ],
        "total_latency_samples": summary["sample_count"],
        "average_latency_ms": summary[
            "average_latency_ms"
        ],
        "median_latency_ms": summary[
            "median_latency_ms"
        ],
        "p95_latency_ms": summary["p95_latency_ms"],
        "standard_deviation_ms": summary[
            "standard_deviation_ms"
        ],
        "min_latency_ms": summary["min_latency_ms"],
        "max_latency_ms": summary["max_latency_ms"],
        "fps_from_median": summary["fps_from_median"],
        "timing_scope": protocol["timing_scope"],
        "disk_io_included": protocol["disk_io_included"],
        "protocol_status": "standardized",
        "protocol_version": "cpu_v1",
        "raw_samples_saved": True,
        "source_results": [
            _relative_path_string(path, project_root)
            for path in included_paths
        ],
        "excluded_stabilization_runs": [
            _relative_path_string(path, project_root)
            for path in excluded_paths
        ],
    }


def run_benchmark_sessions(
    search_space: Dict[str, Any],
    candidate: Dict[str, Any],
    project_root: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    benchmark_config_path = Path(
        search_space["evaluation"]["benchmark_config"]
    )
    if not benchmark_config_path.is_absolute():
        benchmark_config_path = (
            project_root / benchmark_config_path
        )

    base_config = load_benchmark_config(
        benchmark_config_path
    )
    result_paths = benchmark_result_paths(
        project_root,
        candidate["candidate_id"],
    )

    if not overwrite:
        existing_paths = [
            path
            for path in result_paths
            if path.exists()
        ]
        if existing_paths:
            raise CandidateRunnerError(
                "Benchmark result already exists; use "
                "--overwrite to replace generated results: "
                f"{existing_paths[0]}"
            )

    results = []

    with tempfile.TemporaryDirectory(
        prefix="edgenas_candidate_runner_"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)

        for session_number, result_path in enumerate(
            result_paths,
            start=1,
        ):
            session_config = build_session_benchmark_config(
                base_config,
                search_space,
                candidate,
                session_number,
                result_path,
            )
            temporary_config_path = (
                temporary_root
                / f"benchmark_session_{session_number}.yaml"
            )

            with temporary_config_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                yaml.safe_dump(
                    session_config,
                    file,
                    sort_keys=False,
                )

            print(
                "Running CPU benchmark session "
                f"{session_number}/{TOTAL_BENCHMARK_SESSIONS}: "
                f"{candidate['candidate_id']}"
            )

            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.benchmarking.cpu_benchmark",
                        str(temporary_config_path),
                    ],
                    cwd=project_root,
                    check=True,
                )
            except subprocess.CalledProcessError as error:
                raise CandidateRunnerError(
                    "CPU benchmark session failed: "
                    f"{session_number}"
                ) from error

            results.append(load_json(result_path))

    included_results = results[STABILIZATION_SESSIONS:]
    included_paths = result_paths[STABILIZATION_SESSIONS:]
    excluded_paths = result_paths[:STABILIZATION_SESSIONS]

    return pool_benchmark_results(
        included_results,
        included_paths,
        excluded_paths,
        candidate["candidate_id"],
        candidate["image_size"],
        project_root,
    )


def load_template_candidate(
    search_space: Dict[str, Any],
    project_root: Path,
) -> Tuple[str, Dict[str, Any]]:
    reusable = [
        item
        for item in search_space["existing_candidates"]
        if item["reuse_accuracy"]
        and item["reuse_benchmark"]
    ]

    if not reusable:
        raise CandidateRunnerError(
            "At least one complete existing candidate is "
            "required as the model/training template."
        )

    source_candidate_id = reusable[0]["candidate_id"]
    source_path = candidate_record_path(
        project_root,
        source_candidate_id,
    )
    source_candidate = load_json(source_path)

    if source_candidate.get("candidate_id") != source_candidate_id:
        raise CandidateRunnerError(
            "Template candidate ID does not match its file."
        )

    for required_section in (
        "model",
        "training",
    ):
        if not isinstance(
            source_candidate.get(required_section),
            dict,
        ):
            raise CandidateRunnerError(
                "Template candidate is missing section: "
                f"{required_section}"
            )

    return source_candidate_id, source_candidate


def build_candidate_record(
    source_candidate: Dict[str, Any],
    candidate: Dict[str, Any],
    accuracy: Dict[str, Any],
    benchmark: Dict[str, Any],
) -> Dict[str, Any]:
    model = copy.deepcopy(source_candidate["model"])
    training = copy.deepcopy(source_candidate["training"])

    return {
        "schema_version": "1.0",
        "candidate_id": candidate["candidate_id"],
        "model": model,
        "training": training,
        "accuracy": accuracy,
        "benchmark": benchmark,
    }


def build_dry_run_plan(
    search_space: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    project_root: Path,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "search_space_id": search_space[
            "search_space_id"
        ],
        "dry_run": True,
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "image_size": candidate["image_size"],
                "action": (
                    "reuse_existing"
                    if candidate["status"] == "reuse_existing"
                    else "validate_and_benchmark"
                ),
                "candidate_record": str(
                    candidate_record_path(
                        project_root,
                        candidate["candidate_id"],
                    )
                ),
                "benchmark_sessions": (
                    0
                    if candidate["status"] == "reuse_existing"
                    else TOTAL_BENCHMARK_SESSIONS
                ),
            }
            for candidate in candidates
        ],
    }


def run_candidates(
    search_space_path: Path,
    requested_candidate_id: Optional[str],
    dry_run: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    project_root = project_root_from_config(
        search_space_path
    )
    search_space = load_search_space(
        search_space_path,
        check_paths=True,
    )
    candidates = select_candidates(
        expand_candidates(search_space),
        requested_candidate_id,
    )

    if dry_run:
        return build_dry_run_plan(
            search_space,
            candidates,
            project_root,
        )

    _, source_candidate = (
        load_template_candidate(
            search_space,
            project_root,
        )
    )
    completed = []

    for candidate in candidates:
        output_path = candidate_record_path(
            project_root,
            candidate["candidate_id"],
        )

        if candidate["status"] == "reuse_existing":
            completed.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "status": "reused",
                    "candidate_record": str(output_path),
                }
            )
            continue

        if output_path.exists() and not overwrite:
            raise CandidateRunnerError(
                "Candidate record already exists; use "
                f"--overwrite to replace it: {output_path}"
            )

        accuracy = run_validation(
            search_space,
            candidate,
            project_root,
        )
        benchmark = run_benchmark_sessions(
            search_space,
            candidate,
            project_root,
            overwrite,
        )
        candidate_record = build_candidate_record(
            source_candidate,
            candidate,
            accuracy,
            benchmark,
        )
        save_json(candidate_record, output_path)

        completed.append(
            {
                "candidate_id": candidate["candidate_id"],
                "status": "completed",
                "candidate_record": str(output_path),
                "map50_95": accuracy["map50_95"],
                "median_latency_ms": benchmark[
                    "median_latency_ms"
                ],
            }
        )

    return {
        "schema_version": "1.0",
        "search_space_id": search_space[
            "search_space_id"
        ],
        "completed": completed,
    }


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Validate, benchmark, and record "
            "EdgeNAS-Lite search candidates."
        )
    )
    argument_parser.add_argument(
        "search_space_file",
        help="Path to the validated search-space YAML.",
    )
    argument_parser.add_argument(
        "--candidate-id",
        help=(
            "Run one candidate. Omit to process every "
            "candidate in the search space."
        ),
    )
    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running models.",
    )
    argument_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of generated candidate results.",
    )
    arguments = argument_parser.parse_args()

    try:
        result = run_candidates(
            Path(arguments.search_space_file),
            arguments.candidate_id,
            arguments.dry_run,
            arguments.overwrite,
        )
    except (
        CandidateRunnerError,
        FileNotFoundError,
        json.JSONDecodeError,
    ) as error:
        argument_parser.error(str(error))

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
