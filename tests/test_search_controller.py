import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.search_controller.controller import (
    SearchControllerError,
    build_candidate_plan,
    run_search,
    validate_candidate_matches_search_space,
    validate_requirement_matches_search_space,
)


def make_requirement(
    request_id="edge_cpu_demo",
    device="cpu",
):
    return SimpleNamespace(
        request_id=request_id,
        target=SimpleNamespace(
            task="object_detection",
            device=device,
            dataset="KITTI",
        ),
        constraints=SimpleNamespace(
            minimum_map50_95=0.25,
            maximum_median_latency_ms=35.0,
            maximum_model_size_mb=6.0,
        ),
        preferences=SimpleNamespace(
            optimization_goal="balanced",
        ),
    )


SEARCH_SPACE = {
    "search_space_id": "resolution_search_v1",
    "fixed_configuration": {
        "task": "object_detection",
        "model": {"checkpoint": "runs/best.pt"},
        "dataset": {"name": "KITTI"},
        "deployment": {"device": "cpu"},
    },
}


CANDIDATES = [
    {
        "candidate_id": "candidate_416",
        "image_size": 416,
        "status": "pending_evaluation",
    },
    {
        "candidate_id": "candidate_640",
        "image_size": 640,
        "status": "reuse_existing",
    },
]


def candidate_record(candidate_id, image_size, map50_95):
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "model": {
            "checkpoint": "runs/best.pt",
            "model_size_mb": 5.102,
        },
        "accuracy": {
            "dataset": "KITTI",
            "image_size": image_size,
            "map50_95": map50_95,
        },
        "benchmark": {
            "device": "cpu",
            "image_size": image_size,
            "median_latency_ms": 10.0,
            "protocol_status": "standardized",
        },
    }


class TestSearchController(unittest.TestCase):
    def test_accepts_matching_requirement_target(self):
        validate_requirement_matches_search_space(
            make_requirement(),
            SEARCH_SPACE,
        )

    def test_rejects_mismatched_requirement_target(self):
        with self.assertRaisesRegex(
            SearchControllerError,
            "does not match",
        ):
            validate_requirement_matches_search_space(
                make_requirement(device="cuda"),
                SEARCH_SPACE,
            )

    def test_rejects_candidate_with_wrong_image_size(self):
        record = candidate_record("candidate_416", 512, 0.30)

        with self.assertRaisesRegex(
            SearchControllerError,
            "image size",
        ):
            validate_candidate_matches_search_space(
                record,
                CANDIDATES[0],
                SEARCH_SPACE,
            )

    def test_plan_reuses_existing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "results" / "candidates"
            records.mkdir(parents=True)
            (records / "candidate_416.json").write_text("{}")
            (records / "candidate_640.json").write_text("{}")

            plan = build_candidate_plan(
                CANDIDATES,
                root,
                overwrite_candidates=False,
            )

        self.assertEqual(
            [item["action"] for item in plan],
            ["reuse_record", "reuse_record"],
        )

    def test_plan_rejects_missing_reusable_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                SearchControllerError,
                "Reusable candidate record is missing",
            ):
                build_candidate_plan(
                    [CANDIDATES[1]],
                    Path(directory),
                    overwrite_candidates=False,
                )

    def test_dry_run_does_not_write_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / "configs"
            configs.mkdir()
            search_path = configs / "search_space.yaml"
            search_path.write_text("placeholder")
            records = root / "results" / "candidates"
            records.mkdir(parents=True)
            for candidate in CANDIDATES:
                path = records / f"{candidate['candidate_id']}.json"
                path.write_text("{}")

            with (
                patch(
                    "src.search_controller.controller.load_requirement",
                    return_value=make_requirement(),
                ),
                patch(
                    "src.search_controller.controller.load_search_space",
                    return_value=SEARCH_SPACE,
                ),
                patch(
                    "src.search_controller.controller.expand_candidates",
                    return_value=CANDIDATES,
                ),
            ):
                result = run_search(
                    configs / "request.yaml",
                    search_path,
                    dry_run=True,
                )

            self.assertTrue(result["dry_run"])
            self.assertFalse((root / "results" / "search_runs").exists())

    def test_complete_run_reuses_evaluates_and_selects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / "configs"
            configs.mkdir()
            search_path = configs / "search_space.yaml"
            search_path.write_text("placeholder")
            records = root / "results" / "candidates"
            records.mkdir(parents=True)

            data = [
                candidate_record("candidate_416", 416, 0.20),
                candidate_record("candidate_640", 640, 0.30),
            ]
            for record in data:
                path = records / f"{record['candidate_id']}.json"
                path.write_text(json.dumps(record))

            with (
                patch(
                    "src.search_controller.controller.load_requirement",
                    return_value=make_requirement(),
                ),
                patch(
                    "src.search_controller.controller.load_search_space",
                    return_value=SEARCH_SPACE,
                ),
                patch(
                    "src.search_controller.controller.expand_candidates",
                    return_value=CANDIDATES,
                ),
                patch(
                    "src.search_controller.controller.run_candidates"
                ) as runner,
            ):
                result = run_search(
                    configs / "request.yaml",
                    search_path,
                )

            runner.assert_not_called()
            self.assertEqual(
                result["selected_candidate_id"],
                "candidate_640",
            )
            self.assertTrue(
                (root / "results" / "selections" / "edge_cpu_demo.json").exists()
            )
            self.assertTrue(
                (root / "results" / "search_runs" / (
                    "edge_cpu_demo__resolution_search_v1.json"
                )).exists()
            )

    def test_complete_run_generates_missing_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / "configs"
            configs.mkdir()
            search_path = configs / "search_space.yaml"
            search_path.write_text("placeholder")
            pending = [CANDIDATES[0]]
            record_path = (
                root
                / "results"
                / "candidates"
                / "candidate_416.json"
            )

            def create_candidate(*args, **kwargs):
                record_path.parent.mkdir(parents=True)
                record_path.write_text(
                    json.dumps(
                        candidate_record(
                            "candidate_416",
                            416,
                            0.30,
                        )
                    )
                )

            with (
                patch(
                    "src.search_controller.controller.load_requirement",
                    return_value=make_requirement(),
                ),
                patch(
                    "src.search_controller.controller.load_search_space",
                    return_value=SEARCH_SPACE,
                ),
                patch(
                    "src.search_controller.controller.expand_candidates",
                    return_value=pending,
                ),
                patch(
                    "src.search_controller.controller.run_candidates",
                    side_effect=create_candidate,
                ) as runner,
            ):
                result = run_search(
                    configs / "request.yaml",
                    search_path,
                )

            runner.assert_called_once()
            self.assertEqual(
                result["candidate_plan"][0]["action"],
                "generate",
            )
            self.assertEqual(
                result["selected_candidate_id"],
                "candidate_416",
            )


if __name__ == "__main__":
    unittest.main()
