import copy
import unittest

from src.constraint_checker.checker import (
    CandidateValidationError,
    evaluate_candidate,
)
from src.requirement_parser.schema import (
    Constraints,
    Preferences,
    Requirement,
    Target,
)


VALID_REQUIREMENT = Requirement(
    schema_version="1.0",
    request_id="edge_cpu_demo",
    target=Target(
        task="object_detection",
        device="cpu",
        dataset="KITTI",
    ),
    constraints=Constraints(
        minimum_map50_95=0.25,
        maximum_median_latency_ms=35.0,
        maximum_model_size_mb=6.0,
    ),
    preferences=Preferences(
        optimization_goal="balanced",
    ),
)


VALID_CANDIDATE = {
    "candidate_id": "yolo26n_kitti_pilot",
    "model": {
        "model_size_mb": 5.102,
    },
    "accuracy": {
        "dataset": "KITTI",
        "map50_95": 0.273,
    },
    "benchmark": {
        "device": "cpu",
        "median_latency_ms": 30.615,
    },
}


class TestConstraintChecker(unittest.TestCase):
    def test_candidate_passes_all_constraints(self) -> None:
        evaluation = evaluate_candidate(
            VALID_REQUIREMENT,
            copy.deepcopy(VALID_CANDIDATE),
        )

        self.assertTrue(
            evaluation["constraints_satisfied"]
        )

        for check in evaluation["checks"].values():
            self.assertTrue(check["passed"])

    def test_fails_accuracy_constraint(self) -> None:
        candidate = copy.deepcopy(VALID_CANDIDATE)
        candidate["accuracy"]["map50_95"] = 0.20

        evaluation = evaluate_candidate(
            VALID_REQUIREMENT,
            candidate,
        )

        self.assertFalse(
            evaluation["checks"][
                "minimum_map50_95"
            ]["passed"]
        )
        self.assertFalse(
            evaluation["constraints_satisfied"]
        )

    def test_fails_latency_constraint(self) -> None:
        candidate = copy.deepcopy(VALID_CANDIDATE)
        candidate["benchmark"][
            "median_latency_ms"
        ] = 40.0

        evaluation = evaluate_candidate(
            VALID_REQUIREMENT,
            candidate,
        )

        self.assertFalse(
            evaluation["checks"][
                "maximum_median_latency_ms"
            ]["passed"]
        )
        self.assertFalse(
            evaluation["constraints_satisfied"]
        )

    def test_fails_model_size_constraint(self) -> None:
        candidate = copy.deepcopy(VALID_CANDIDATE)
        candidate["model"]["model_size_mb"] = 7.0

        evaluation = evaluate_candidate(
            VALID_REQUIREMENT,
            candidate,
        )

        self.assertFalse(
            evaluation["checks"][
                "maximum_model_size_mb"
            ]["passed"]
        )
        self.assertFalse(
            evaluation["constraints_satisfied"]
        )

    def test_rejects_dataset_mismatch(self) -> None:
        candidate = copy.deepcopy(VALID_CANDIDATE)
        candidate["accuracy"]["dataset"] = "COCO"

        with self.assertRaisesRegex(
            CandidateValidationError,
            "Dataset mismatch",
        ):
            evaluate_candidate(
                VALID_REQUIREMENT,
                candidate,
            )

    def test_rejects_device_mismatch(self) -> None:
        candidate = copy.deepcopy(VALID_CANDIDATE)
        candidate["benchmark"]["device"] = "mps"

        with self.assertRaisesRegex(
            CandidateValidationError,
            "Device mismatch",
        ):
            evaluate_candidate(
                VALID_REQUIREMENT,
                candidate,
            )

    def test_rejects_missing_metric(self) -> None:
        candidate = copy.deepcopy(VALID_CANDIDATE)
        del candidate["benchmark"]["median_latency_ms"]

        with self.assertRaisesRegex(
            CandidateValidationError,
            "Missing candidate field",
        ):
            evaluate_candidate(
                VALID_REQUIREMENT,
                candidate,
            )

    def test_rejects_invalid_accuracy_value(self) -> None:
        candidate = copy.deepcopy(VALID_CANDIDATE)
        candidate["accuracy"]["map50_95"] = 1.5

        with self.assertRaisesRegex(
            CandidateValidationError,
            "between 0 and 1",
        ):
            evaluate_candidate(
                VALID_REQUIREMENT,
                candidate,
            )


if __name__ == "__main__":
    unittest.main()