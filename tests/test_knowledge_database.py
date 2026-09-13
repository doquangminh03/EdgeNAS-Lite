import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.knowledge_database.loader import (
    KnowledgeValidationError,
    load_knowledge_base,
    validate_candidate_metrics,
)


class TestCandidateMetricValidation(unittest.TestCase):
    def make_record(self):
        return {
            "candidate_id": "test_candidate",
            "accuracy": {"map50_95": 0.25},
            "benchmark": {"median_latency_ms": 10.0},
            "model": {"model_size_mb": 5.102},
        }

    def test_accepts_valid_metrics_and_accuracy_boundaries(self):
        for accuracy in (0, 0.25, 1):
            with self.subTest(accuracy=accuracy):
                record = self.make_record()
                record["accuracy"]["map50_95"] = accuracy

                validate_candidate_metrics(record)

    def test_rejects_invalid_accuracy(self):
        invalid_values = (
            -0.01,
            1.01,
            True,
            "0.25",
            None,
            float("nan"),
            float("inf"),
        )

        for value in invalid_values:
            with self.subTest(value=value):
                record = self.make_record()
                record["accuracy"]["map50_95"] = value

                with self.assertRaises(KnowledgeValidationError):
                    validate_candidate_metrics(record)

    def test_rejects_invalid_latency(self):
        invalid_values = (
            0,
            -1,
            True,
            "10.0",
            None,
            float("nan"),
            float("inf"),
        )

        for value in invalid_values:
            with self.subTest(value=value):
                record = self.make_record()
                record["benchmark"]["median_latency_ms"] = value

                with self.assertRaises(KnowledgeValidationError):
                    validate_candidate_metrics(record)

    def test_rejects_missing_metric(self):
        fields = (
            ("accuracy", "map50_95"),
            ("benchmark", "median_latency_ms"),
            ("model", "model_size_mb"),
        )

        for section, field in fields:
            with self.subTest(section=section, field=field):
                record = self.make_record()
                del record[section][field]

                with self.assertRaises(KnowledgeValidationError):
                    validate_candidate_metrics(record)

    def test_rejects_missing_or_invalid_section(self):
        for section in ("accuracy", "benchmark", "model"):
            with self.subTest(section=section, case="missing"):
                record = self.make_record()
                del record[section]

                with self.assertRaises(KnowledgeValidationError):
                    validate_candidate_metrics(record)

            for value in (None, [], "invalid"):
                with self.subTest(section=section, value=value):
                    record = self.make_record()
                    record[section] = value

                    with self.assertRaises(KnowledgeValidationError):
                        validate_candidate_metrics(record)
    def test_rejects_invalid_model_size(self):
        invalid_values = (
            0,
            -1,
            True,
            "5.102",
            None,
            float("nan"),
            float("inf"),
        )

        for value in invalid_values:
            with self.subTest(value=value):
                record = self.make_record()
                record["model"]["model_size_mb"] = value

                with self.assertRaises(KnowledgeValidationError):
                    validate_candidate_metrics(record)

class TestKnowledgeLoader(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

        self.record = {
            "candidate_id": "candidate_a",
            "accuracy": {"map50_95": 0.25},
            "benchmark": {"median_latency_ms": 10.0},
            "model": {"model_size_mb": 5.102},
        }

        self.index = {
            "schema_version": "1.0",
            "knowledge_base_id": "test_kb",
            "candidate_records": [
                {
                    "candidate_id": "candidate_a",
                    "record_path": "results/candidates/candidate_a.json",
                }
            ],
        }

        self.write_record()
        self.write_index()

    def write_record(self):
        path = self.root / "results/candidates/candidate_a.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.record),
            encoding="utf-8",
        )

    def write_index(self):
        path = self.root / "knowledge/index.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.index),
            encoding="utf-8",
        )

    def load(self):
        return load_knowledge_base("knowledge/index.yaml", self.root)

    def test_loads_record_and_preserves_source_path(self):
        knowledge = self.load()

        self.assertEqual(knowledge["knowledge_base_id"], "test_kb")
        self.assertEqual(len(knowledge["candidates"]), 1)

        candidate = knowledge["candidates"][0]
        self.assertEqual(candidate["record"], self.record)
        self.assertEqual(
            candidate["record_path"],
            "results/candidates/candidate_a.json",
        )

    def test_rejects_duplicate_candidate_id(self):
        entry = self.index["candidate_records"][0]
        self.index["candidate_records"].append(dict(entry))
        self.write_index()

        with self.assertRaisesRegex(
            KnowledgeValidationError, "Duplicate candidate ID"
        ):
            self.load()

    def test_rejects_candidate_id_mismatch(self):
        self.record["candidate_id"] = "candidate_b"
        self.write_record()

        with self.assertRaisesRegex(
            KnowledgeValidationError, "Candidate ID mismatch"
        ):
            self.load()

    def test_reports_missing_record_file(self):
        path = self.root / "results/candidates/candidate_a.json"
        path.unlink()

        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_rejects_record_path_outside_project(self):
        self.index["candidate_records"][0]["record_path"] = (
            "../outside_candidate.json"
        )
        self.write_index()

        with self.assertRaisesRegex(
            KnowledgeValidationError, "outside the project"
        ):
            self.load()

    def test_rejects_invalid_metrics_when_loading(self):
        self.record["accuracy"]["map50_95"] = 1.2
        self.write_record()

        with self.assertRaisesRegex(
            KnowledgeValidationError, "accuracy.map50_95"
        ):
            self.load()
if __name__ == "__main__":
    unittest.main()