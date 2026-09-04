import copy
import unittest
from pathlib import Path

from ultralytics import data

from src.search_space.parser import (
    SearchSpaceValidationError,
    expand_candidates,
    load_search_space,
    parse_search_space_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEARCH_SPACE_FILE = (
    PROJECT_ROOT
    / "configs"
    / "search_space.yaml"
)


class TestSearchSpaceParser(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_search_space = load_search_space(
            SEARCH_SPACE_FILE,
            check_paths=False,
        )

    def test_valid_search_space_expands_candidates(
        self,
    ) -> None:
        candidates = expand_candidates(
            copy.deepcopy(
                self.valid_search_space
            )
        )

        self.assertEqual(len(candidates), 3)

        self.assertEqual(
            [
                candidate["image_size"]
                for candidate in candidates
            ],
            [416, 512, 640],
        )

        self.assertEqual(
            candidates[0]["status"],
            "pending_evaluation",
        )
        self.assertEqual(
            candidates[1]["status"],
            "pending_evaluation",
        )
        self.assertEqual(
            candidates[2]["status"],
            "reuse_existing",
        )
        self.assertEqual(
            candidates[2]["candidate_id"],
            "yolo26n_kitti_pilot",
        )

    def test_rejects_duplicate_image_sizes(
        self,
    ) -> None:
        data = copy.deepcopy(
            self.valid_search_space
        )

        data["variable_dimensions"][
            "image_size"
        ]["values"] = [416, 416, 640]

        with self.assertRaisesRegex(
            SearchSpaceValidationError,
            "must be unique",
        ):
            parse_search_space_data(data)

    def test_rejects_image_size_not_divisible_by_32(
        self,
    ) -> None:
        data = copy.deepcopy(
            self.valid_search_space
        )

        data["variable_dimensions"][
            "image_size"
        ]["values"] = [416, 500, 640]

        with self.assertRaisesRegex(
            SearchSpaceValidationError,
            "divisible by 32",
        ):
            parse_search_space_data(data)

    def test_rejects_budget_mismatch(self) -> None:
        data = copy.deepcopy(
            self.valid_search_space
        )

        data["search"]["experiment_budget"] = 2

        with self.assertRaisesRegex(
            SearchSpaceValidationError,
            "must equal",
        ):
            parse_search_space_data(data)

    def test_rejects_retraining(self) -> None:
        data = copy.deepcopy(
            self.valid_search_space
        )

        data["fixed_configuration"][
            "model"
        ]["retrain"] = True

        with self.assertRaisesRegex(
            SearchSpaceValidationError,
            "retrain: false",
        ):
            parse_search_space_data(data)

    def test_rejects_unsupported_device(
        self,
    ) -> None:
        data = copy.deepcopy(
            self.valid_search_space
        )

        data["fixed_configuration"][
            "deployment"
        ]["device"] = "cuda"

        with self.assertRaisesRegex(
            SearchSpaceValidationError,
            "Unsupported value",
        ):
            parse_search_space_data(data)

    def test_rejects_incomplete_metric_contract(
        self,
    ) -> None:
        data = copy.deepcopy(
            self.valid_search_space
        )

        data["evaluation"][
            "required_metrics"
        ].remove(
            "benchmark.p95_latency_ms"
        )

        with self.assertRaisesRegex(
            SearchSpaceValidationError,
            "metric contract",
        ):
            parse_search_space_data(data)

    def test_rejects_existing_candidate_outside_space(
        self,
    ) -> None:
        data = copy.deepcopy(
            self.valid_search_space
        )

        data["existing_candidates"][0][
            "matches"
        ]["image_size"] = 320

        with self.assertRaisesRegex(
            SearchSpaceValidationError,
            "not in the search space",
        ):
            parse_search_space_data(data)

    def test_rejects_mismatched_classes_and_ids(self) -> None:
            data = copy.deepcopy(self.valid_search_space)
            data["fixed_configuration"]["dataset"]["class_ids"] = [0, 3]

            with self.assertRaisesRegex(
            SearchSpaceValidationError,
            "must have the same length",
        ):
                parse_search_space_data(data)
        


if __name__ == "__main__":
    unittest.main()
