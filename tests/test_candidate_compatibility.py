import unittest
from copy import deepcopy

from src.proposal.compatibility import check_candidate_compatibility


class TestCandidateCompatibility(unittest.TestCase):
    def setUp(self):
        # Synthetic fixture; does not modify real candidate records.
        self.record = {
            "candidate_id": "test_candidate",
            "accuracy": {
                "dataset": "KITTI",
                "split": "val",
                "image_size": 416,
                "validation_images": 1496,
                "validation_instances": 6989,
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

    def check(self, record):
        return check_candidate_compatibility(
            record,
            expected_hardware_id="test_machine_01",
        )

    def test_accepts_each_supported_resolution(self):
        for size in (416, 512, 640):
            with self.subTest(size=size):
                record = deepcopy(self.record)
                record["accuracy"]["image_size"] = size
                record["benchmark"]["image_size"] = size

                report = self.check(record)

                self.assertEqual(report["status"], "compatible")
                self.assertEqual(report["missing_fields"], [])
                self.assertEqual(report["mismatches"], [])
                self.assertEqual(report["unverified"], [])

    def test_reports_missing_hardware_id(self):
        del self.record["benchmark"]["hardware_id"]

        report = self.check(self.record)

        self.assertEqual(report["status"], "insufficient_metadata")
        self.assertIn(
            "benchmark.hardware_id",
            report["missing_fields"],
        )

    def test_requires_target_hardware_identity(self):
        report = check_candidate_compatibility(self.record)

        self.assertEqual(report["status"], "insufficient_metadata")
        self.assertEqual(report["missing_fields"], [])
        self.assertTrue(report["unverified"])

    def test_rejects_different_hardware(self):
        self.record["benchmark"]["hardware_id"] = "another_machine"

        report = self.check(self.record)

        self.assertEqual(report["status"], "incompatible")
        self.assertIn(
            "benchmark.hardware_id",
            [item["field"] for item in report["mismatches"]],
        )

    def test_reports_missing_accuracy_image_size(self):
        del self.record["accuracy"]["image_size"]

        report = self.check(self.record)

        self.assertEqual(report["status"], "insufficient_metadata")
        self.assertIn(
            "accuracy.image_size",
            report["missing_fields"],
        )

    def test_rejects_accuracy_benchmark_resolution_mismatch(self):
        self.record["accuracy"]["image_size"] = 640

        report = self.check(self.record)

        self.assertEqual(report["status"], "incompatible")
        self.assertIn(
            "accuracy.image_size",
            [item["field"] for item in report["mismatches"]],
        )

    def test_rejects_incompatible_benchmark_settings(self):
        changes = {
            "batch_size": 2,
            "protocol_version": "cpu_v2",
            "timing_scope": "inference_only",
            "disk_io_included": True,
        }

        for field, value in changes.items():
            with self.subTest(field=field):
                record = deepcopy(self.record)
                record["benchmark"][field] = value

                report = self.check(record)

                self.assertEqual(report["status"], "incompatible")
                self.assertIn(
                    f"benchmark.{field}",
                    [item["field"] for item in report["mismatches"]],
                )

    def test_rejects_different_evaluated_classes(self):
        del self.record["accuracy"]["per_class"]["cyclist"]

        report = self.check(self.record)

        self.assertEqual(report["status"], "incompatible")
        self.assertIn(
            "accuracy.per_class",
            [item["field"] for item in report["mismatches"]],
        )

    def test_does_not_modify_candidate(self):
        original = deepcopy(self.record)

        self.check(self.record)

        self.assertEqual(self.record, original)


if __name__ == "__main__":
    unittest.main()