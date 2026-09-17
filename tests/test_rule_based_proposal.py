import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from src.proposal.rule_based import propose_from_knowledge


class TestRuleBasedProposal(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

        self.root = Path(self.temporary_directory.name)
        self.hardware_id = "test_machine_01"

        self.requirement = {
            "schema_version": "1.0",
            "request_id": "proposal_test",
            "target": {
                "task": "object_detection",
                "device": "cpu",
                "dataset": "KITTI",
            },
            "constraints": {
                "minimum_map50_95": 0.20,
                "maximum_median_latency_ms": 12.0,
                "maximum_model_size_mb": 6.0,
            },
            "preferences": {
                "optimization_goal": "balanced",
            },
        }

        self.records = {}

        specifications = [
            ("candidate_416", 416, 0.207384, 8.754),
            ("candidate_512", 512, 0.242931, 11.279),
            ("candidate_640", 640, 0.273, 16.429),
        ]

        for candidate_id, image_size, accuracy, latency in specifications:
            self.records[candidate_id] = {
                "schema_version": "1.0",
                "candidate_id": candidate_id,
                "model": {
                    "family": "YOLO26",
                    "model_size_mb": 5.102,
                },
                "accuracy": {
                    "dataset": "KITTI",
                    "split": "val",
                    "image_size": image_size,
                    "validation_images": 1496,
                    "validation_instances": 6989,
                    "map50_95": accuracy,
                    "per_class": {
                        "car": {},
                        "pedestrian": {},
                        "cyclist": {},
                    },
                },
                "benchmark": {
                    "dataset": "KITTI",
                    "split": "val",
                    "device": "cpu",
                    "hardware_id": self.hardware_id,
                    "image_size": image_size,
                    "median_latency_ms": latency,
                    "batch_size": 1,
                    "protocol_version": "cpu_v1",
                    "protocol_status": "standardized",
                    "timing_scope": "preprocess_inference_postprocess",
                    "disk_io_included": False,
                    "selected_images": 100,
                    "selection_seed": 42,
                    "session_count": 3,
                    "warmup_runs_per_session": 10,
                    "repetitions_per_image": 3,
                },
            }

        index = {
            "schema_version": "1.0",
            "knowledge_base_id": "test_kb",
            "candidate_records": [
                {
                    "candidate_id": candidate_id,
                    "record_path": (
                        f"results/candidates/{candidate_id}.json"
                    ),
                }
                for candidate_id in self.records
            ],
        }

        index_path = self.root / "knowledge/index.yaml"
        index_path.parent.mkdir(parents=True)
        index_path.write_text(
            yaml.safe_dump(index, sort_keys=False),
            encoding="utf-8",
        )

        self.write_records()
        self.write_requirement()

    def write_records(self):
        for candidate_id, record in self.records.items():
            path = (
                self.root
                / "results"
                / "candidates"
                / f"{candidate_id}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(record, indent=2, allow_nan=False),
                encoding="utf-8",
            )

    def write_requirement(self):
        path = self.root / "request.yaml"
        path.write_text(
            yaml.safe_dump(self.requirement, sort_keys=False),
            encoding="utf-8",
        )

    def propose(self, **overrides):
        options = {
            "project_root": self.root,
            "model_family": "YOLO26",
            "expected_hardware_id": self.hardware_id,
        }
        options.update(overrides)

        return propose_from_knowledge(
            "request.yaml",
            **options,
        )

    def test_selects_416_from_multiple_feasible_candidates(self):
        result = self.propose()

        self.assertEqual(result["proposal_status"], "selected")
        self.assertEqual(
            result["compatible_candidate_ids"],
            ["candidate_416", "candidate_512", "candidate_640"],
        )
        self.assertEqual(result["excluded_candidate_ids"], [])
        self.assertEqual(
            result["rejected_candidate_ids"],
            ["candidate_640"],
        )

        selection = result["selection"]
        self.assertEqual(
            selection["selected_candidate_id"],
            "candidate_416",
        )
        self.assertEqual(
            [item["candidate_id"] for item in selection["ranking"]],
            ["candidate_416", "candidate_512"],
        )

        for item, expected_score in zip(
            selection["ranking"],
            [0.143132, 0.087805],
        ):
            self.assertAlmostEqual(
                item["score"],
                expected_score,
                places=6,
            )

        self.assertEqual(
            result["selected_source"],
            {
                "candidate_id": "candidate_416",
                "record_path": "results/candidates/candidate_416.json",
            },
        )

    def test_selects_640_as_only_feasible_candidate(self):
        self.requirement["constraints"].update({
            "minimum_map50_95": 0.25,
            "maximum_median_latency_ms": 35.0,
        })
        self.write_requirement()

        result = self.propose()

        self.assertEqual(result["proposal_status"], "selected")
        self.assertEqual(
            result["feasible_candidate_ids"],
            ["candidate_640"],
        )
        self.assertEqual(
            result["selection"]["selected_candidate_id"],
            "candidate_640",
        )

    def test_reports_no_feasible_candidate(self):
        self.requirement["constraints"]["minimum_map50_95"] = 0.99
        self.write_requirement()

        result = self.propose()

        self.assertEqual(
            result["proposal_status"],
            "no_feasible_candidate",
        )
        self.assertEqual(len(result["evaluations"]), 3)
        self.assertEqual(result["excluded_candidate_ids"], [])
        self.assertEqual(result["feasible_candidate_ids"], [])
        self.assertEqual(result["selection"]["ranking"], [])
        self.assertIsNone(
            result["selection"]["selected_candidate_id"]
        )
        self.assertIsNone(result["selected_source"])

    def test_reports_no_matching_candidates(self):
        result = self.propose(model_family="unknown_model")

        self.assertEqual(
            result["proposal_status"],
            "no_matching_candidates",
        )
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["compatibility_reports"], [])
        self.assertEqual(result["evaluations"], [])
        self.assertIsNone(result["selection"])
        self.assertIsNone(result["selected_source"])

    def test_excludes_all_candidates_for_different_hardware(self):
        result = self.propose(
            expected_hardware_id="another_machine",
        )

        self.assertEqual(
            result["proposal_status"],
            "no_compatible_candidates",
        )
        self.assertEqual(result["compatible_candidate_ids"], [])
        self.assertEqual(
            result["excluded_candidate_ids"],
            ["candidate_416", "candidate_512", "candidate_640"],
        )
        self.assertEqual(result["evaluations"], [])
        self.assertIsNone(result["selection"])
        self.assertIsNone(result["selected_source"])

        for report in result["compatibility_reports"]:
            self.assertEqual(report["status"], "incompatible")
            self.assertIn(
                "benchmark.hardware_id",
                [item["field"] for item in report["mismatches"]],
            )

    def test_blocks_selection_without_target_hardware(self):
        # Omit the argument to check the public function's default.
        result = propose_from_knowledge(
            "request.yaml",
            project_root=self.root,
            model_family="YOLO26",
        )

        self.assertEqual(
            result["proposal_status"],
            "no_compatible_candidates",
        )
        self.assertEqual(len(result["excluded_candidate_ids"]), 3)
        self.assertEqual(result["evaluations"], [])
        self.assertIsNone(result["selection"])

        for report in result["compatibility_reports"]:
            self.assertEqual(
                report["status"],
                "insufficient_metadata",
            )
            self.assertTrue(report["unverified"])

    def test_excludes_missing_metadata_before_ranking(self):
        # 416 normally wins, but cannot compete without hardware metadata.
        del self.records["candidate_416"]["benchmark"]["hardware_id"]
        self.write_records()

        result = self.propose()

        self.assertEqual(result["proposal_status"], "selected")
        self.assertEqual(
            result["excluded_candidate_ids"],
            ["candidate_416"],
        )
        self.assertEqual(
            result["selection"]["selected_candidate_id"],
            "candidate_512",
        )
        self.assertEqual(
            [item["candidate_id"] for item in result["evaluations"]],
            ["candidate_512", "candidate_640"],
        )
        self.assertEqual(
            [
                item["candidate_id"]
                for item in result["selection"]["ranking"]
            ],
            ["candidate_512"],
        )

        report = next(
            item
            for item in result["compatibility_reports"]
            if item["candidate_id"] == "candidate_416"
        )
        self.assertEqual(report["status"], "insufficient_metadata")
        self.assertIn(
            "benchmark.hardware_id",
            report["missing_fields"],
        )
        self.assertEqual(
            report["record_path"],
            "results/candidates/candidate_416.json",
        )

    def test_excludes_incompatible_protocol_before_ranking(self):
        self.records["candidate_416"]["benchmark"][
            "timing_scope"
        ] = "inference_only"
        self.write_records()

        result = self.propose()

        self.assertEqual(result["proposal_status"], "selected")
        self.assertEqual(
            result["excluded_candidate_ids"],
            ["candidate_416"],
        )
        self.assertEqual(
            result["selection"]["selected_candidate_id"],
            "candidate_512",
        )
        self.assertEqual(
            [item["candidate_id"] for item in result["evaluations"]],
            ["candidate_512", "candidate_640"],
        )

        report = next(
            item
            for item in result["compatibility_reports"]
            if item["candidate_id"] == "candidate_416"
        )
        self.assertEqual(report["status"], "incompatible")
        self.assertIn(
            "benchmark.timing_scope",
            [item["field"] for item in report["mismatches"]],
        )


if __name__ == "__main__":
    unittest.main()