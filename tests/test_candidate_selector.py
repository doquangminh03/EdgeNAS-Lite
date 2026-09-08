import copy
import unittest
from types import SimpleNamespace

from src.candidate_selector.selector import (
    CandidateSelectorError,
    select_candidate,
)


def make_requirement(optimization_goal="balanced"):
    return SimpleNamespace(
        schema_version="1.0",
        request_id="edge_cpu_demo",
        target=SimpleNamespace(
            task="object_detection",
            device="cpu",
            dataset="KITTI",
        ),
        constraints=SimpleNamespace(
            minimum_map50_95=0.25,
            maximum_median_latency_ms=35.0,
            maximum_model_size_mb=6.0,
        ),
        preferences=SimpleNamespace(
            optimization_goal=optimization_goal,
        ),
    )


def make_evaluation(
    candidate_id,
    map50_95,
    latency_ms,
    model_size_mb,
):
    accuracy_passed = map50_95 >= 0.25
    latency_passed = latency_ms <= 35.0
    size_passed = model_size_mb <= 6.0
    overall = accuracy_passed and latency_passed and size_passed

    return {
        "schema_version": "1.0",
        "evaluation_id": f"edge_cpu_demo__{candidate_id}",
        "request_id": "edge_cpu_demo",
        "candidate_id": candidate_id,
        "target": {
            "task": "object_detection",
            "device": "cpu",
            "dataset": "KITTI",
        },
        "checks": {
            "minimum_map50_95": {
                "metric_path": "accuracy.map50_95",
                "operator": ">=",
                "required": 0.25,
                "actual": map50_95,
                "passed": accuracy_passed,
            },
            "maximum_median_latency_ms": {
                "metric_path": "benchmark.median_latency_ms",
                "operator": "<=",
                "required": 35.0,
                "actual": latency_ms,
                "passed": latency_passed,
            },
            "maximum_model_size_mb": {
                "metric_path": "model.model_size_mb",
                "operator": "<=",
                "required": 6.0,
                "actual": model_size_mb,
                "passed": size_passed,
            },
        },
        "constraints_satisfied": overall,
    }


CURRENT_EVALUATIONS = [
    make_evaluation(
        "yolo26n_kitti_pilot_imgsz416_cpu",
        0.207384,
        8.754,
        5.102,
    ),
    make_evaluation(
        "yolo26n_kitti_pilot_imgsz512_cpu",
        0.242931,
        11.279,
        5.102,
    ),
    make_evaluation(
        "yolo26n_kitti_pilot",
        0.273,
        16.429,
        5.102,
    ),
]


class TestCandidateSelector(unittest.TestCase):
    def test_selects_only_feasible_current_candidate(self):
        result = select_candidate(
            make_requirement(),
            copy.deepcopy(CURRENT_EVALUATIONS),
        )

        self.assertEqual(result["selection_status"], "selected")
        self.assertEqual(result["feasible_candidate_count"], 1)
        self.assertEqual(
            result["selected_candidate_id"],
            "yolo26n_kitti_pilot",
        )
        self.assertEqual(
            result["selection_method"],
            "only_feasible_candidate",
        )

    def test_returns_no_selection_when_every_candidate_fails(self):
        evaluations = [
            make_evaluation("candidate_a", 0.20, 10.0, 5.0),
            make_evaluation("candidate_b", 0.24, 12.0, 5.0),
        ]

        result = select_candidate(
            make_requirement(), evaluations
        )

        self.assertEqual(
            result["selection_status"],
            "no_feasible_candidate",
        )
        self.assertIsNone(result["selected_candidate_id"])

    def test_balanced_goal_uses_equal_constraint_headroom(self):
        evaluations = [
            make_evaluation("accuracy_heavy", 0.30, 30.0, 5.0),
            make_evaluation("balanced_fast", 0.26, 10.0, 3.0),
        ]

        result = select_candidate(
            make_requirement("balanced"), evaluations
        )

        self.assertEqual(
            result["selected_candidate_id"],
            "balanced_fast",
        )

    def test_accuracy_goal_selects_highest_accuracy(self):
        evaluations = [
            make_evaluation("accurate", 0.40, 30.0, 5.0),
            make_evaluation("fast", 0.26, 8.0, 3.0),
        ]

        result = select_candidate(
            make_requirement("accuracy"), evaluations
        )

        self.assertEqual(result["selected_candidate_id"], "accurate")

    def test_latency_goal_selects_lowest_latency(self):
        evaluations = [
            make_evaluation("accurate", 0.40, 30.0, 5.0),
            make_evaluation("fast", 0.26, 8.0, 5.0),
        ]

        result = select_candidate(
            make_requirement("latency"), evaluations
        )

        self.assertEqual(result["selected_candidate_id"], "fast")

    def test_model_size_goal_selects_smallest_model(self):
        evaluations = [
            make_evaluation("large", 0.40, 10.0, 5.0),
            make_evaluation("small", 0.26, 10.0, 2.0),
        ]

        result = select_candidate(
            make_requirement("model_size"), evaluations
        )

        self.assertEqual(result["selected_candidate_id"], "small")

    def test_rejects_duplicate_candidate_ids(self):
        evaluation = make_evaluation("duplicate", 0.30, 10.0, 5.0)

        with self.assertRaisesRegex(
            CandidateSelectorError,
            "must be unique",
        ):
            select_candidate(
                make_requirement(),
                [evaluation, copy.deepcopy(evaluation)],
            )

    def test_rejects_inconsistent_pass_value(self):
        evaluation = make_evaluation("tampered", 0.20, 10.0, 5.0)
        evaluation["checks"]["minimum_map50_95"]["passed"] = True
        evaluation["constraints_satisfied"] = True

        with self.assertRaisesRegex(
            CandidateSelectorError,
            "Stored pass/fail value is inconsistent",
        ):
            select_candidate(make_requirement(), [evaluation])

    def test_rejects_stale_constraint_value(self):
        evaluation = make_evaluation("stale", 0.30, 10.0, 5.0)
        evaluation["checks"]["minimum_map50_95"]["required"] = 0.30

        with self.assertRaisesRegex(
            CandidateSelectorError,
            "is stale",
        ):
            select_candidate(make_requirement(), [evaluation])

    def test_rejects_non_positive_latency(self):
        evaluation = make_evaluation("invalid", 0.30, -1.0, 5.0)

        with self.assertRaisesRegex(
            CandidateSelectorError,
            "latency must be greater than 0",
        ):
            select_candidate(make_requirement(), [evaluation])


if __name__ == "__main__":
    unittest.main()
