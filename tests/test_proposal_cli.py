import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestProposalCLI(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

        self.requirement = {
            "schema_version": "1.0",
            "request_id": "cli_test",
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

        record = {
            "schema_version": "1.0",
            "candidate_id": "candidate_416",
            "model": {
                "family": "YOLO26",
                "model_size_mb": 5.102,
            },
            "accuracy": {
                "dataset": "KITTI",
                "split": "val",
                "image_size": 416,
                "validation_images": 1496,
                "validation_instances": 6989,
                "map50_95": 0.207384,
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
                "hardware_id": "test_machine_01",
                "image_size": 416,
                "median_latency_ms": 8.754,
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

        self.record_path = (
            self.root / "results/candidates/candidate_416.json"
        )
        self.record_path.parent.mkdir(parents=True)
        self.record_path.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

        index = {
            "schema_version": "1.0",
            "knowledge_base_id": "cli_test_kb",
            "candidate_records": [
                {
                    "candidate_id": "candidate_416",
                    "record_path": (
                        "results/candidates/candidate_416.json"
                    ),
                },
            ],
        }

        self.index_path = self.root / "knowledge/index.yaml"
        self.index_path.parent.mkdir(parents=True)
        self.index_path.write_text(
            yaml.safe_dump(index),
            encoding="utf-8",
        )

        self.requirement_path = self.root / "request.yaml"
        self.output_path = self.root / "results/proposals/demo.json"
        self.write_requirement()

    def write_requirement(self):
        self.requirement_path.write_text(
            yaml.safe_dump(self.requirement),
            encoding="utf-8",
        )

    def run_cli(
        self,
        *,
        hardware_id="test_machine_01",
        model_family="YOLO26",
        requirement="request.yaml",
        output="results/proposals/demo.json",
    ):
        command = [
            sys.executable,
            "-m",
            "src.proposal.cli",
            requirement,
            "--project-root",
            str(self.root),
            "--model-family",
            model_family,
            "--output",
            output,
        ]

        if hardware_id is not None:
            command.extend(["--hardware-id", hardware_id])

        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def read_output(self):
        return json.loads(
            self.output_path.read_text(encoding="utf-8")
        )

    def test_selects_candidate_and_saves_json_with_exit_zero(self):
        result = self.run_cli()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Status: selected", result.stdout)
        self.assertIn("Selected: candidate_416", result.stdout)

        saved = self.read_output()
        self.assertEqual(saved["request_id"], "cli_test")
        self.assertEqual(
            saved["expected_hardware_id"],
            "test_machine_01",
        )
        self.assertEqual(
            saved["selection"]["selected_candidate_id"],
            "candidate_416",
        )
        self.assertEqual(
            saved["compatibility_reports"][0]["status"],
            "compatible",
        )

    def test_different_hardware_saves_report_with_exit_one(self):
        result = self.run_cli(hardware_id="another_machine")

        self.assertEqual(result.returncode, 1, result.stderr)
        saved = self.read_output()

        self.assertEqual(
            saved["proposal_status"],
            "no_compatible_candidates",
        )
        self.assertEqual(
            saved["excluded_candidate_ids"],
            ["candidate_416"],
        )
        self.assertEqual(saved["evaluations"], [])
        self.assertIsNone(saved["selection"])

    def test_no_matching_candidates_saves_report_with_exit_one(self):
        result = self.run_cli(model_family="unknown_model")

        self.assertEqual(result.returncode, 1, result.stderr)
        saved = self.read_output()

        self.assertEqual(
            saved["proposal_status"],
            "no_matching_candidates",
        )
        self.assertEqual(saved["candidate_count"], 0)
        self.assertIsNone(saved["selection"])

    def test_no_feasible_candidate_saves_report_with_exit_one(self):
        self.requirement["constraints"]["minimum_map50_95"] = 0.99
        self.write_requirement()

        result = self.run_cli()

        self.assertEqual(result.returncode, 1, result.stderr)
        saved = self.read_output()

        self.assertEqual(
            saved["proposal_status"],
            "no_feasible_candidate",
        )
        self.assertEqual(
            saved["compatible_candidate_ids"],
            ["candidate_416"],
        )
        self.assertEqual(saved["selection"]["ranking"], [])
        self.assertIsNone(saved["selected_source"])

    def test_missing_hardware_argument_returns_two_without_output(self):
        result = self.run_cli(hardware_id=None)

        self.assertEqual(result.returncode, 2)
        self.assertIn("--hardware-id", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_invalid_hardware_argument_returns_two_without_output(self):
        for value in ("", "   ", "<MISSING>"):
            with self.subTest(value=value):
                result = self.run_cli(hardware_id=value)

                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "Hardware ID must be a non-empty identifier.",
                    result.stderr,
                )
                self.assertFalse(self.output_path.exists())

    def test_missing_requirement_returns_two_without_output(self):
        result = self.run_cli(requirement="does_not_exist.yaml")

        self.assertEqual(result.returncode, 2)
        self.assertIn("Error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_output_cannot_replace_inputs(self):
        paths = [
            self.requirement_path,
            self.index_path,
            self.record_path,
        ]

        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()

                result = self.run_cli(output=str(path))

                self.assertEqual(result.returncode, 2)
                self.assertIn("Error:", result.stderr)
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()