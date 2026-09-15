import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.proposal.rule_based import propose_from_knowledge


class TestRuleBasedProposal(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

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

        # Test fixtures reproduce the three measured metric combinations.
        candidate_metrics = [
            ("candidate_416", 0.207384, 8.754),
            ("candidate_512", 0.242931, 11.279),
            ("candidate_640", 0.273, 16.429),
        ]

        entries = []

        for candidate_id, accuracy, latency in candidate_metrics:
            record_path = f"results/candidates/{candidate_id}.json"

            record = {
                "schema_version": "1.0",
                "candidate_id": candidate_id,
                "model": {
                    "family": "YOLO26",
                    "model_size_mb": 5.102,
                },
                "accuracy": {
                    "dataset": "KITTI",
                    "map50_95": accuracy,
                },
                "benchmark": {
                    "device": "cpu",
                    "median_latency_ms": latency,
                },
            }

            path = self.root / record_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record), encoding="utf-8")

            entries.append({
                "candidate_id": candidate_id,
                "record_path": record_path,
            })

        index = {
            "schema_version": "1.0",
            "knowledge_base_id": "test_kb",
            "candidate_records": entries,
        }

        index_path = self.root / "knowledge/index.yaml"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            yaml.safe_dump(index),
            encoding="utf-8",
        )

        self.write_requirement()

    def write_requirement(self):
        path = self.root / "request.yaml"
        path.write_text(
            yaml.safe_dump(self.requirement),
            encoding="utf-8",
        )

    def propose(self, model_family="YOLO26"):
        return propose_from_knowledge(
            "request.yaml",
            project_root=self.root,
            model_family=model_family,
        )

    def test_selects_416_from_multiple_feasible_candidates(self):
        result = self.propose()
        selection = result["selection"]

        self.assertEqual(result["proposal_status"], "selected")
        self.assertEqual(
            selection["selected_candidate_id"],
            "candidate_416",
        )
        self.assertEqual(
            [
                item["candidate_id"]
                for item in selection["ranking"]
            ],
            ["candidate_416", "candidate_512"],
        )
        self.assertEqual(
            [item["score"] for item in selection["ranking"]],
            [0.143132, 0.087805],
        )
        self.assertEqual(
            result["selected_source"],
            {
                "candidate_id": "candidate_416",
                "record_path": "results/candidates/candidate_416.json",
            },
        )
        self.assertEqual(
            result["rejected_candidate_ids"],
            ["candidate_640"],
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
            result["selection"]["selected_candidate_id"],
            "candidate_640",
        )
        self.assertEqual(
            result["feasible_candidate_ids"],
            ["candidate_640"],
        )
        self.assertEqual(
            result["selected_source"]["record_path"],
            "results/candidates/candidate_640.json",
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
        self.assertEqual(result["feasible_candidate_ids"], [])
        self.assertEqual(len(result["rejected_candidate_ids"]), 3)
        self.assertIsNone(
            result["selection"]["selected_candidate_id"]
        )
        self.assertEqual(result["selection"]["ranking"], [])
        self.assertIsNone(result["selected_source"])

    def test_reports_no_matching_candidates(self):
        result = self.propose(model_family="UNKNOWN_MODEL")

        self.assertEqual(
            result["proposal_status"],
            "no_matching_candidates",
        )
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["evaluations"], [])
        self.assertIsNone(result["selection"])
        self.assertIsNone(result["selected_source"])


if __name__ == "__main__":
    unittest.main()