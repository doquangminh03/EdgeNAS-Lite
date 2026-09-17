from typing import Any, Dict, Optional


# Explicit policy for the current KITTI CPU experiment.
# These values are not a universal policy for other experiments.
PILOT_POLICY = {
    "accuracy.dataset": "KITTI",
    "accuracy.split": "val",
    "accuracy.validation_images": 1496,
    "accuracy.validation_instances": 6989,
    "benchmark.dataset": "KITTI",
    "benchmark.split": "val",
    "benchmark.device": "cpu",
    "benchmark.batch_size": 1,
    "benchmark.protocol_version": "cpu_v1",
    "benchmark.protocol_status": "standardized",
    "benchmark.timing_scope": "preprocess_inference_postprocess",
    "benchmark.disk_io_included": False,
    "benchmark.selected_images": 100,
    "benchmark.selection_seed": 42,
    "benchmark.session_count": 3,
    "benchmark.warmup_runs_per_session": 10,
    "benchmark.repetitions_per_image": 3,
}

EXPECTED_CLASSES = {"car", "pedestrian", "cyclist"}


def read_field(record: Dict[str, Any], path: str) -> Any:
    value = record

    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]

    return value


def is_missing(value: Any) -> bool:
    return (
        value is None
        or (
            isinstance(value, str)
            and value.strip() in {"", "<MISSING>"}
        )
    )


def check_candidate_compatibility(
    record: Dict[str, Any],
    *,
    expected_hardware_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Check a record against the current pilot metadata policy."""

    if expected_hardware_id is not None:
        if (
            not isinstance(expected_hardware_id, str)
            or is_missing(expected_hardware_id)
        ):
            raise ValueError(
                "expected_hardware_id must be a non-empty identifier."
            )

    mismatches = []
    missing_fields = []
    unverified = []

    for path, expected in PILOT_POLICY.items():
        actual = read_field(record, path)

        if is_missing(actual):
            missing_fields.append(path)
        elif type(actual) is not type(expected) or actual != expected:
            mismatches.append({
                "field": path,
                "expected": expected,
                "actual": actual,
            })

    # Check evaluated class names, rather than assuming that
    # training classes prove which classes were evaluated.
    per_class = read_field(record, "accuracy.per_class")

    if per_class is None or per_class == {}:
        missing_fields.append("accuracy.per_class")
    elif not isinstance(per_class, dict):
        mismatches.append({
            "field": "accuracy.per_class",
            "expected": "object with evaluated class names",
            "actual": per_class,
        })
    elif set(per_class) != EXPECTED_CLASSES:
        mismatches.append({
            "field": "accuracy.per_class",
            "expected": sorted(EXPECTED_CLASSES),
            "actual": sorted(per_class),
        })

    # Resolutions may differ between candidates.
    # Within one candidate, accuracy and latency must refer
    # to the same input resolution.
    sizes = {}

    for path in ("accuracy.image_size", "benchmark.image_size"):
        value = read_field(record, path)

        if is_missing(value):
            missing_fields.append(path)
        elif type(value) is not int or value <= 0 or value % 32 != 0:
            mismatches.append({
                "field": path,
                "expected": "positive integer divisible by 32",
                "actual": value,
            })
        else:
            sizes[path] = value

    if len(sizes) == 2:
        if sizes["accuracy.image_size"] != sizes["benchmark.image_size"]:
            mismatches.append({
                "field": "accuracy.image_size",
                "expected": sizes["benchmark.image_size"],
                "actual": sizes["accuracy.image_size"],
            })

    hardware_id = read_field(record, "benchmark.hardware_id")

    if is_missing(hardware_id):
        missing_fields.append("benchmark.hardware_id")
    elif not isinstance(hardware_id, str):
        mismatches.append({
            "field": "benchmark.hardware_id",
            "expected": "non-empty string",
            "actual": hardware_id,
        })
    elif expected_hardware_id is not None:
        if hardware_id != expected_hardware_id:
            mismatches.append({
                "field": "benchmark.hardware_id",
                "expected": expected_hardware_id,
                "actual": hardware_id,
            })

    if expected_hardware_id is None:
        unverified.append("Target hardware identity has not been specified.")

    if mismatches:
        status = "incompatible"
    elif missing_fields or unverified:
        status = "insufficient_metadata"
    else:
        status = "compatible"

    return {
        "candidate_id": record.get("candidate_id"),
        "policy_id": "kitti_cpu_pilot_v1",
        "status": status,
        "mismatches": mismatches,
        "missing_fields": sorted(set(missing_fields)),
        "unverified": unverified,
    }